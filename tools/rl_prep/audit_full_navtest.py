"""Read-only audit of the full official navtest records: how many scenes could serve as additional evaluation?

Reports the number of scenes, whether the processed image-token files exist, the log of every scene, and how
many scenes lie in logs that occur neither in the frozen cohort nor in the RL training pool.
Run with the drivevla environment; nothing is written except an optional JSON report.
"""
import argparse
import json
import pickle
import sys
from collections import Counter
from pathlib import Path


class Compat(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "numpy._core" or module.startswith("numpy._core."):
            module = "numpy.core" + module[len("numpy._core"):]
        return super().find_class(module, name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-pkl", type=Path, required=True, help="navsim_emu_vla_256_144_test_pre_1s.pkl")
    ap.add_argument("--raw-log-dir", type=Path, required=True, help="navsim_logs/test directory with the raw log pickles")
    ap.add_argument("--frozen-tokens", type=Path, required=True)
    ap.add_argument("--pool-cache", type=Path, required=True, help="metric_cache_trainval_rl (log directories = pool logs)")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    with args.full_pkl.open("rb") as f:
        records = Compat(f).load()
    tokens = [r["token"] for r in records]
    print(f"records: {len(records)}  unique tokens: {len(set(tokens))}  keys: {sorted(records[0])[:12]}")
    missing_img = [r["token"] for r in records
                   if not Path(r["image"][3]).is_file() or not Path(r["pre_1s_image"][3]).is_file()]
    print(f"records whose image-token files are missing: {len(missing_img)}")

    need = set(tokens)
    log_of = {}
    for path in sorted(args.raw_log_dir.glob("*.pkl")):
        with path.open("rb") as f:
            frames = Compat(f).load()
        for fr in frames:
            t = str(fr.get("token", ""))
            if t in need:
                log_of[t] = str(fr["log_name"])
        del frames
    print(f"scenes with a log name: {len(log_of)} / {len(need)}  (raw log files: {len(list(args.raw_log_dir.glob('*.pkl')))})")

    frozen = set(json.loads(args.frozen_tokens.read_text()))
    frozen_logs = {log_of[t] for t in frozen if t in log_of}
    pool_logs = {p.name for p in args.pool_cache.iterdir() if p.is_dir()}
    per_log = Counter(log_of[t] for t in tokens if t in log_of)
    fresh = [l for l in per_log if l not in frozen_logs and l not in pool_logs]
    out = {"records": len(records), "missing_image_token_files": len(missing_img), "logs": len(per_log),
           "frozen_cohort_logs": len(frozen_logs), "pool_logs": len(pool_logs),
           "pool_logs_overlapping_navtest": len(set(per_log) & pool_logs),
           "fresh_logs": len(fresh), "fresh_log_scenes": int(sum(per_log[l] for l in fresh)),
           "scenes_in_frozen_logs_but_not_frozen": int(sum(c for l, c in per_log.items() if l in frozen_logs) - len(frozen & set(log_of))),
           "fresh_scenes_with_missing_image_tokens": len([t for t in missing_img if log_of.get(t) in set(fresh)])}
    print(json.dumps(out, indent=2))
    if args.output:
        args.output.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
