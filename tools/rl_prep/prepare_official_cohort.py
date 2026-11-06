"""Build a 'cohort' directory for the FULL official navtest split, in the layout the navtest selector pipeline expects.

Layout written to <out>:
  test.pkl            the released full navtest pickle with a "token" field added to every record (the released file has none;
                      the collection code builds its token->index map from scene["token"]). Everything else is unchanged.
  test_tokens.json    navtest tokens in pickle order (taken from the stock navtest scene filter; length must match the pickle)
  cache               symlink to the official metric cache built with run_metric_caching.py
  token_log_map.json  {token: "log_name:<log>"} read off the cache layout <cache>/<log>/<type>/<token>/metric_cache.pkl
The pickle order equals the token-list order: the released inference writes <token>.json by that index and its scores match
the published numbers (see the paired comparison of the authors' outputs).
"""
import argparse
import json
import pickle
from pathlib import Path

import yaml


def navtest_tokens(yaml_path):
    tokens = yaml.safe_load(Path(yaml_path).read_text())["tokens"]
    if len(set(tokens)) != len(tokens):
        raise ValueError("duplicate tokens in the navtest filter")
    return [str(t) for t in tokens]


def pickle_length(path):
    obj = pickle.load(open(path, "rb"))
    return len(obj["data"]) if isinstance(obj, dict) and "data" in obj else len(obj)


def add_tokens(records, tokens):
    """Copy of the pickle's record list with record["token"] = tokens[i]; refuses a record that already has another token."""
    if isinstance(records, dict) and "data" in records:
        out = dict(records)
        out["data"] = add_tokens(records["data"], tokens)
        return out
    if len(records) != len(tokens):
        raise ValueError(f"{len(records)} records but {len(tokens)} tokens")
    out = []
    for rec, tok in zip(records, tokens):
        if rec.get("token", tok) != tok:
            raise ValueError("record already carries a different token")
        new = dict(rec)
        new["token"] = tok
        out.append(new)
    return out


def token_log_map(cache, tokens):
    found = {}
    wanted = set(tokens)
    for pkl in Path(cache).glob("*/*/*/metric_cache.pkl"):
        if pkl.parent.name in wanted:
            found[pkl.parent.name] = "log_name:" + pkl.parents[2].name
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-pkl", type=Path, required=True)
    ap.add_argument("--token-yaml", type=Path, required=True)
    ap.add_argument("--metric-cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing existing {args.out}")
    tokens = navtest_tokens(args.token_yaml)
    n = pickle_length(args.test_pkl)
    if n != len(tokens):
        raise SystemExit(f"pickle has {n} scenes but the navtest filter lists {len(tokens)} tokens")
    mapping = token_log_map(args.metric_cache, tokens)
    missing = [t for t in tokens if t not in mapping]
    if missing:
        raise SystemExit(f"{len(missing)} navtest tokens have no metric cache entry, e.g. {missing[0]}")
    args.out.mkdir(parents=True)
    with open(args.test_pkl, "rb") as f:
        records = pickle.load(f)
    with open(args.out / "test.pkl", "wb") as f:
        pickle.dump(add_tokens(records, tokens), f)
    (args.out / "cache").symlink_to(args.metric_cache.resolve())
    (args.out / "test_tokens.json").write_text(json.dumps(tokens) + "\n")
    (args.out / "token_log_map.json").write_text(json.dumps(mapping) + "\n")
    print(f"{len(tokens)} scenes, {len(set(mapping.values()))} logs -> {args.out}")


if __name__ == "__main__":
    main()
