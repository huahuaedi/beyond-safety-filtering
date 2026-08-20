import random
import unittest

from risk_curriculum import (
    parse_source_mix,
    select_curriculum_memory,
    tempered_priority_weights,
)


class RiskCurriculumTest(unittest.TestCase):
    def test_tempered_priority_endpoints(self):
        self.assertEqual(tempered_priority_weights([0.25, 1.0], 0), [1.0, 1.0])
        self.assertEqual(tempered_priority_weights([0.25, 1.0], 1), [0.25, 1.0])
        self.assertEqual(tempered_priority_weights([0.25, 1.0], 0.5), [0.5, 1.0])

    def test_parse_mix_normalizes(self):
        self.assertEqual(
            parse_source_mix("manifest:2,coverage:1,online:1"),
            {"manifest": 0.5, "coverage": 0.25, "online": 0.25},
        )

    def test_source_quota_is_approximately_respected(self):
        memory = {
            "m": {"sources": {"manifest"}, "priority": 1, "label_counts": {}},
            "c": {"sources": {"coverage"}, "priority": 1, "label_counts": {}},
            "o": {"sources": {"online"}, "priority": 1, "label_counts": {}},
        }
        rng = random.Random(7)
        counts = {key: 0 for key in "mco"}
        for _ in range(4000):
            token, _ = select_curriculum_memory(
                memory, rng, {"manifest": 0.5, "coverage": 0.25, "online": 0.25}
            )
            counts[token] += 1
        self.assertAlmostEqual(counts["m"] / 4000, 0.5, delta=0.03)
        self.assertAlmostEqual(counts["c"] / 4000, 0.25, delta=0.03)

    def test_missing_source_redistributes_quota(self):
        memory = {
            "m": {"sources": {"manifest"}, "priority": 1, "label_counts": {}},
            "o": {"sources": {"online"}, "priority": 1, "label_counts": {}},
        }
        token, source = select_curriculum_memory(
            memory,
            random.Random(0),
            {"manifest": 0.5, "coverage": 0.25, "online": 0.25},
        )
        self.assertIn(token, memory)
        self.assertIn(source, {"manifest", "online"})

    def test_replayed_item_is_downweighted(self):
        memory = {
            "fresh": {"sources": {"manifest"}, "priority": 1, "replay_count": 0},
            "stale": {"sources": {"manifest"}, "priority": 1, "replay_count": 20},
        }
        rng = random.Random(11)
        selected = [
            select_curriculum_memory(
                memory,
                rng,
                {"manifest": 1.0},
                type_balance=False,
                redundancy_coef=0.5,
            )[0]
            for _ in range(1000)
        ]
        self.assertGreater(selected.count("fresh"), 5 * selected.count("stale"))

    def test_rare_risk_type_receives_more_samples(self):
        memory = {
            "common": {
                "sources": {"manifest"},
                "priority": 1,
                "label_counts": {"unsafe": 100},
            },
            "rare": {
                "sources": {"manifest"},
                "priority": 1,
                "label_counts": {"nc_failure": 1},
            },
        }
        rng = random.Random(19)
        selected = [
            select_curriculum_memory(memory, rng, {"manifest": 1.0})[0]
            for _ in range(1000)
        ]
        self.assertGreater(selected.count("rare"), 4 * selected.count("common"))


if __name__ == "__main__":
    unittest.main()
