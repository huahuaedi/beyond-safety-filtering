import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from offline_verified_data import (topk_safe_labels, build_schedule, pdms, read_scene,
                                   POSITIVE, NEGATIVE, SAFETY, absolute_from_relative)
import build_log_disjoint_manifest as mf

METRICS = (*SAFETY, 'ego_progress', 'comfort')


def scores(**over):
    n = 6
    s = {k: np.ones(n) for k in METRICS}
    s['ego_progress'] = np.array([.5, .6, .7, .8, .9, .95])
    for k, v in over.items():
        s[k] = np.asarray(v, float)
    return s


def ref(**over):
    r = {k: np.ones(2) for k in METRICS}
    r['ego_progress'] = np.array([.5, .5])
    for k, v in over.items():
        r[k] = np.asarray(v, float)
    return r


class TopKTests(unittest.TestCase):
    def test_picks_k_highest_pdms(self):
        m = topk_safe_labels(scores(), ref(), k=2)
        self.assertEqual(np.flatnonzero(m['labels'] == POSITIVE).tolist(), [4, 5])

    def test_unsafe_candidate_cannot_win_even_with_best_progress(self):
        s = scores(drivable_area_compliance=[1, 1, 1, 1, 1, 0])
        m = topk_safe_labels(s, ref(), k=2)
        self.assertEqual(np.flatnonzero(m['labels'] == POSITIVE).tolist(), [3, 4])

    def test_no_forced_positive(self):
        m = topk_safe_labels(scores(), ref(ego_progress=[1., 1.]), k=2)
        self.assertEqual(m['labels'].tolist(), [NEGATIVE] * 6)

    def test_failed_reference_scene_is_repairable(self):
        r = ref(drivable_area_compliance=[0., 0.])
        s = scores(drivable_area_compliance=[0, 0, 1, 1, 1, 1])
        m = topk_safe_labels(s, r, k=1)
        self.assertEqual(np.flatnonzero(m['labels'] == POSITIVE).tolist(), [5])

    def test_candidate_worse_than_worse_probe_is_excluded(self):
        r = ref(time_to_collision_within_bound=[1., .5])
        s = scores(time_to_collision_within_bound=[1, 1, 1, 1, 1, .4])
        m = topk_safe_labels(s, r, k=1)
        self.assertEqual(np.flatnonzero(m['labels'] == POSITIVE).tolist(), [4])

    def test_missing_or_bad_inputs_fail_closed(self):
        s = scores(); del s['comfort']
        with self.assertRaises(ValueError): topk_safe_labels(s, ref())
        s = scores(ego_progress=[np.nan] + [.5] * 5)
        with self.assertRaises(ValueError): topk_safe_labels(s, ref())

    def test_pdms_matches_scorer_formula(self):
        self.assertAlmostEqual(pdms(1, 1, .5, 1, 1), (2.5 + 5 + 2) / 12)


class ScheduleTests(unittest.TestCase):
    def setUp(self):
        self.records = [{'token': f't{i}', 'candidates': 10} for i in range(5)]
        self.labels = {r['token']: np.where(np.arange(10) < (i % 3), POSITIVE, NEGATIVE)
                       for i, r in enumerate(self.records)}
        self.fn = lambda r: self.labels[r['token']]

    def build(self, mode, **kw):
        return build_schedule(self.records, self.fn, mode=mode, candidates_per_scene=4,
                              epochs=1, seed=3, **kw)

    def test_filtered_uses_only_positives_and_skips_scenes_without(self):
        sched = self.build('filtered')
        self.assertTrue(sched)
        for r, c, _ in sched:
            self.assertEqual(self.labels[r['token']][c], POSITIVE)
        self.assertFalse({r['token'] for r, _, _ in sched} & {'t0', 't3'})

    def test_conditioned_and_unconditional_share_schedule(self):
        a = self.build('conditioned', positive_fraction=.5)
        b = self.build('unconditional', positive_fraction=.5)
        self.assertEqual([(r['token'], c) for r, c, _ in a], [(r['token'], c) for r, c, _ in b])

    def test_positive_fraction_includes_positives_when_available(self):
        sched = self.build('conditioned', positive_fraction=.5)
        got = {}
        for r, c, _ in sched:
            got.setdefault(r['token'], []).append(self.labels[r['token']][c])
        for tok, labs in got.items():
            has_pos = (self.labels[tok] == POSITIVE).any()
            self.assertEqual(any(l == POSITIVE for l in labs), bool(has_pos))
            self.assertEqual(len(labs), 4)

    def test_legacy_schedule_is_unchanged_without_fraction(self):
        sched = self.build('conditioned')
        self.assertEqual(len(sched), 20)
        self.assertEqual(len({(r['token'], c) for r, c, _ in sched}), 20)


def write_scene(path, token, split, n=4, ep=None):
    rel = np.zeros((n, 8, 3), np.float32); rel[:, :, 0] = 1
    d = dict(scene_token=token, split=split, relative_trajectories=rel,
             absolute_trajectories=absolute_from_relative(rel))
    for k in METRICS:
        d['score__' + k] = np.ones(n, np.float32)
        d['reference_probe__' + k] = np.ones(2, np.float32)
    d['score__ego_progress'] = np.asarray(ep if ep is not None else [.6, .7, .8, .9], np.float32)
    d['reference_probe__ego_progress'] = np.array([.5, .5], np.float32)
    np.savez(path, **d)


class ManifestTests(unittest.TestCase):
    def make(self, root):
        # logs: A (train), B (train), C (holds final-eval token), D (holds legacy val token)
        cache = root / 'cache'
        layout = {'a1': 'A', 'a2': 'A', 'b1': 'B', 'c1': 'C', 'c2': 'C', 'd1': 'D'}
        for tok, log in layout.items():
            f = cache / log / 'unknown' / tok
            f.mkdir(parents=True); (f / 'metric_cache.pkl').touch()
        data = root / 'data'
        split_of = {'a1': 'train', 'a2': 'val', 'b1': 'test', 'c2': 'train', 'd1': 'val'}
        for tok, split in split_of.items():
            (data / split).mkdir(parents=True, exist_ok=True)
            write_scene(data / split / f'{tok}.npz', tok, split)
        (root / 'final.json').write_text(json.dumps(['c1']))
        (root / 'val.json').write_text(json.dumps(['d1']))
        return data, cache

    def run_main(self, root, data, cache, *extra):
        import sys
        out = root / 'm.json'
        argv = ['x', '--root', str(data), '--metric-cache-dirs', str(cache),
                '--final-eval-tokens', str(root / 'final.json'), '--val-tokens', str(root / 'val.json'),
                '--output', str(out), '--extra-heldout-log-fraction', '0', *extra]
        old = sys.argv; sys.argv = argv
        try:
            mf.main()
        finally:
            sys.argv = old
        return json.loads(out.read_text())

    def test_log_isolation_ignores_original_split_dirs(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); data, cache = self.make(root)
            m = self.run_main(root, data, cache)
        train = {r['token'] for r in m['train']}
        # a2 is in the val directory and b1 in test, but their logs are not held out, so they train
        self.assertEqual(train, {'a1', 'a2', 'b1'})
        # c2 (log C, via the final-eval token) and d1 (log D, legacy val) are held out
        self.assertEqual(set(m['heldout_eval_tokens']), {'c2', 'd1'})
        self.assertEqual({r['token']: r['split'] for r in m['train']}['a2'], 'val')
        self.assertTrue(all(r['positive'] == 2 for r in m['train']))

    def test_unmapped_scene_fails_closed(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); data, cache = self.make(root)
            write_scene(data / 'train' / 'zz.npz', 'zz', 'train')
            with self.assertRaisesRegex(ValueError, 'lack a log mapping'):
                self.run_main(root, data, cache)

    def test_forced_token_without_cache_fails_closed(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); data, cache = self.make(root)
            (root / 'final.json').write_text(json.dumps(['nope']))
            with self.assertRaisesRegex(ValueError, 'no metric cache'):
                self.run_main(root, data, cache)


if __name__ == '__main__':
    unittest.main()
