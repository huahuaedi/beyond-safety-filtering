"""Write {token: "log_name:<log>"} for a token list from a NAVSIM metric cache layout <cache>/<log>/<type>/<token>/."""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens-json", type=Path, required=True)
    ap.add_argument("--metric-cache", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    tokens = set(json.loads(args.tokens_json.read_text()))
    found = {}
    for pkl in args.metric_cache.glob("*/*/*/metric_cache.pkl"):
        if pkl.parent.name in tokens:
            found[pkl.parent.name] = "log_name:" + pkl.parents[2].name
    missing = tokens - set(found)
    if missing:
        raise SystemExit(f"{len(missing)} tokens have no metric cache entry, e.g. {sorted(missing)[0]}")
    args.output.write_text(json.dumps(found, indent=1) + "\n")
    print(f"{len(found)} tokens in {len(set(found.values()))} logs -> {args.output}")


if __name__ == "__main__":
    main()
