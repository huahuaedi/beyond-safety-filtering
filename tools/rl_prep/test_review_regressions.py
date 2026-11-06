"""Regression tests for the 2026-09-23 correctness review (NumPy only)."""
import ast
from pathlib import Path
import tempfile
import unittest

import numpy as np

from policy_optimization import compute_reference_anchored_advantage
from risk_ranker_splits import inner_scene_split
from evaluate_world_action_mining import load_oof

KEYS = ('no_at_fault_collisions', 'drivable_area_compliance',
        'time_to_collision_within_bound', 'ego_progress',
        'driving_direction_compliance')


class HardSignAdvantageTest(unittest.TestCase):
    def scores(self, n):
        return {key: np.ones(n, dtype=np.float32) for key in KEYS}

    def test_reported_counterexample_and_training_entrypoint(self):
        # Exercise the actual training wrapper without importing the large model.
        source = Path(__file__).with_name('rl_stage1_prototype.py').read_text()
        node = next(n for n in ast.parse(source).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'compute_reference_anchored_advantage')
        scope = dict(reference_anchored_advantage=compute_reference_anchored_advantage,
                     ADVANTAGE_NOISE_FLOOR=0.02, ADVANTAGE_VERSION='current')
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<training-wrapper>', 'exec'), scope)
        scores = self.scores(16)
        scores[KEYS[0]][2:] = 0
        for key in KEYS[1:3]:
            scores[key][3:] = 0
        adv, feasible, violation = scope['compute_reference_anchored_advantage'](scores, dict.fromkeys(KEYS, 1))
        self.assertFalse(feasible[2])
        self.assertEqual(violation[2], 1)
        self.assertLess(adv[2], 0)
        self.assertGreater(adv[2], adv[3])
        self.assertGreater(adv[feasible].min(), adv[~feasible].max())

    def test_sizes_imbalance_singleton_all_bad_and_zero_variance(self):
        for n in (1, 2, 16, 64, 257):
            for n_safe in sorted({0, 1, n // 2, n - 1, n}):
                for constant in (False, True):
                    with self.subTest(n=n, safe=n_safe, constant=constant):
                        scores = self.scores(n)
                        scores[KEYS[0]][n_safe:] = 0
                        if not constant:
                            scores['ego_progress'] = np.linspace(0, 1, n, dtype=np.float32) ** 8
                        adv, feasible, _ = compute_reference_anchored_advantage(scores, dict.fromkeys(KEYS, 1))
                        self.assertTrue(np.isfinite(adv).all())
                        self.assertTrue((adv[~feasible] < 0).all())
                        self.assertTrue((np.abs(adv[feasible]) <= 0.5).all())
                        if 0 < n_safe < n:
                            self.assertGreater(adv[feasible].min(), adv[~feasible].max())

    def test_zero_weights_do_not_disable_hard_gate(self):
        scores = self.scores(2)
        scores[KEYS[0]][1] = 0
        adv, feasible, _ = compute_reference_anchored_advantage(
            scores, dict.fromkeys(KEYS, 1), dict(nc=0, dac=0, ttc=0))
        self.assertFalse(feasible[1])
        self.assertLess(adv[1], 0)

    def test_progress_is_explicitly_group_relative(self):
        scores = self.scores(4)
        scores['ego_progress'] = np.array([0.1, 0.2, 0.3, 0.4])
        reference = dict.fromkeys(KEYS, 1)
        a = compute_reference_anchored_advantage(scores, reference)[0]
        reference['ego_progress'] = 0
        b = compute_reference_anchored_advantage(scores, reference)[0]
        np.testing.assert_array_equal(a, b)
        self.assertTrue((np.diff(a) > 0).all())


class InnerSplitTest(unittest.TestCase):
    def test_partition_determinism_and_whole_scene_isolation(self):
        paths = [Path(f'train/scene_{i}.npz') for i in range(20)]
        train, val = inner_scene_split(paths, seed=5)
        self.assertEqual((train, val), inner_scene_split(paths[::-1], seed=5))
        self.assertFalse(set(train) & set(val))
        self.assertEqual(set(train) | set(val), set(paths))
        self.assertEqual(len(val), 4)

    def test_invalid_or_duplicate_scenes_fail(self):
        for paths in ([], [Path('a.npz')], [Path('train/a.npz'), Path('val/a.npz')]):
            with self.assertRaises(ValueError):
                inner_scene_split(paths)


class OOFProvenanceTest(unittest.TestCase):
    def test_legacy_and_overlap_rejected_valid_predictions_load(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'action/seed_0/fold_0/oof_predictions.npz'
            path.parent.mkdir(parents=True)
            payload = dict(scene_names=np.array(['outer']),
                           probabilities=np.zeros((1, 64, 4)), labels=np.zeros((1, 64, 4)))
            np.savez(path, **payload)
            with self.assertRaisesRegex(ValueError, 'regenerate'):
                load_oof(root, 'action', 1)
            payload.update(selection_protocol=np.array('inner_scene_holdout_v1'),
                           training_scene_names=np.array(['train']), selection_scene_names=np.array(['outer']))
            np.savez(path, **payload)
            with self.assertRaisesRegex(ValueError, 'overlapping'):
                load_oof(root, 'action', 1)
            payload['selection_scene_names'] = np.array(['inner'])
            np.savez(path, **payload)
            names, probabilities, labels = load_oof(root, 'action', 1)
            self.assertEqual(names, ['outer'])
            self.assertEqual(probabilities.shape, (1, 64, 4))


class DiagnosticTransformTest(unittest.TestCase):
    def test_shuffle_preserves_labels_and_joint_layer_pairing(self):
        from run_wm_diagnostics import transform_features
        l1 = np.arange(24).reshape(2, 6, 2).astype(np.float32)
        action, labels = l1.copy(), np.zeros((2, 6, 4))
        arrays = (action, l1, l1 + 100, labels, ['a', 'b'])
        result = transform_features(arrays, 'shuffled_fusion', 0)
        np.testing.assert_array_equal(result[0], action)
        np.testing.assert_array_equal(result[3], labels)
        np.testing.assert_array_equal(result[2] - result[1], np.full_like(l1, 100))
        self.assertFalse(np.array_equal(result[1], l1))
        for before, after in zip(l1, result[1]):
            np.testing.assert_array_equal(np.sort(before, axis=0), np.sort(after, axis=0))
        np.testing.assert_array_equal(result[1], transform_features(arrays, 'shuffled_fusion', 0)[1])


if __name__ == '__main__':
    unittest.main()
