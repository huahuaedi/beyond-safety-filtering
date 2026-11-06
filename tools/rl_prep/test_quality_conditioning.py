import tempfile
import unittest
from pathlib import Path

import numpy as np

from quality_conditioned_dataset import QualityConditionedWorldActionDataset
from quality_conditioning import (
    CONDITION_DIM,
    build_quality_metadata,
    desired_high_quality_condition,
)


class QualityConditioningTest(unittest.TestCase):
    def scores(self):
        return {
            "no_at_fault_collisions": np.array([1, 1, 0, 1], np.float32),
            "drivable_area_compliance": np.array([1, 1, 1, 0], np.float32),
            "time_to_collision_within_bound": np.array([1, 1, 1, 1], np.float32),
            "ego_progress": np.array([1.0, 0.5, 1.0, 1.0], np.float32),
            "driving_direction_compliance": np.ones(4, np.float32),
            "comfort": np.ones(4, np.float32),
        }

    def reference(self):
        return {
            "no_at_fault_collisions": 1.0,
            "drivable_area_compliance": 1.0,
            "time_to_collision_within_bound": 1.0,
            "ego_progress": 0.75,
            "driving_direction_compliance": 1.0,
            "comfort": 1.0,
        }

    def test_safety_is_lexicographic_and_vector_valued(self):
        metadata = build_quality_metadata(self.scores(), self.reference())
        self.assertEqual(metadata["condition_vector"].shape, (4, CONDITION_DIM))
        self.assertTrue(metadata["risk_mask"][2, 0])
        self.assertTrue(metadata["risk_mask"][3, 1])
        self.assertFalse(metadata["risk_mask"][0].any())
        self.assertGreaterEqual(metadata["quality_tier"][0], 2)
        self.assertLessEqual(metadata["quality_tier"][2], 1)

    def test_dynamic_weight_changes_severity_not_risk_identity(self):
        static = build_quality_metadata(self.scores(), self.reference())
        dynamic = build_quality_metadata(
            self.scores(), self.reference(), {"nc": 2.0, "dac": 0.5, "ttc": 1.0}
        )
        np.testing.assert_array_equal(static["risk_mask"], dynamic["risk_mask"])
        self.assertGreater(dynamic["weighted_violation"][2], static["weighted_violation"][2])
        self.assertLess(dynamic["weighted_violation"][3], static["weighted_violation"][3])

    def test_desired_condition_requests_high_quality_and_feasible(self):
        desired = desired_high_quality_condition(2)
        self.assertEqual(desired.shape, (2, CONDITION_DIM))
        self.assertTrue((desired[:, 3] == 1).all())
        self.assertTrue((desired[:, 7] == 1).all())
        self.assertTrue((desired[:, 4:7] == 0).all())

    def test_lazy_dataset_pairs_sidecar_and_latent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("data", "quality", "latent"):
                (root / name / "train").mkdir(parents=True)
            np.savez_compressed(
                root / "data/train/a.npz",
                scene_token=np.asarray("a"), split=np.asarray("train"),
                relative_trajectories=np.zeros((2, 8, 3), np.float32),
            )
            np.savez_compressed(
                root / "quality/train/a.npz",
                quality_tier=np.array([3, 0]), advantage=np.array([1.0, -1.0]),
                advantage_bin=np.array([4, 0]), risk_mask=np.zeros((2, 3), bool),
                condition_vector=np.zeros((2, CONDITION_DIM), np.float32),
            )
            np.savez_compressed(
                root / "latent/train/a.npz",
                eoa_latents=np.zeros((2, 4, 16), np.float32),
            )
            dataset = QualityConditionedWorldActionDataset(
                root / "data", root / "quality", root / "latent"
            )
            self.assertEqual(len(dataset), 2)
            self.assertEqual(dataset[0]["quality_tier"], 3)
            self.assertEqual(dataset[0]["wm_latent"].shape, (4, 16))


if __name__ == "__main__":
    unittest.main()
