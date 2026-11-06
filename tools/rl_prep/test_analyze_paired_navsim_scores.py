import csv
import tempfile
import unittest
from pathlib import Path

from analyze_paired_navsim_scores import analyze, find_csv, load_scores


class PairedNavsimAnalysisTest(unittest.TestCase):
    def test_load_discards_average_and_analysis_is_paired(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, scores in (("base", (0.2, 0.4)), ("candidate", (0.3, 0.6))):
                path = root / name / "nested" / "scores.csv"
                path.parent.mkdir(parents=True)
                with path.open("w", newline="") as stream:
                    writer = csv.DictWriter(
                        stream, fieldnames=["", "token", "valid", "score", "comfort"]
                    )
                    writer.writeheader()
                    writer.writerow(
                        {"": 0, "token": "a", "valid": True, "score": scores[0], "comfort": 1}
                    )
                    writer.writerow(
                        {"": 1, "token": "b", "valid": True, "score": scores[1], "comfort": 0}
                    )
                    writer.writerow(
                        {"": 2, "token": "average", "valid": True, "score": 999, "comfort": 999}
                    )

            _, baseline = load_scores(root / "base")
            _, candidate = load_scores(root / "candidate")
            result = analyze(baseline, candidate, draws=1000, seed=7)
            self.assertEqual(result["scenes"], 2)
            self.assertAlmostEqual(result["metrics"]["score"]["mean_delta"], 0.15)
            self.assertEqual(result["metrics"]["score"]["wins"], 2)
            self.assertEqual(result["metrics"]["comfort"]["ties"], 2)

    def test_find_csv_rejects_ambiguous_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.csv").touch()
            (root / "b.csv").touch()
            with self.assertRaises(RuntimeError):
                find_csv(root)


if __name__ == "__main__":
    unittest.main()
