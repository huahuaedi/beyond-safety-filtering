#!/usr/bin/env python3
"""Download released DriveVLA-W0 checkpoints (top-level files of each folder only) and optionally the authors' 87.2 json outputs.

The folders contain json_output/ subtrees with ~10k small files; recursive globbing hits the anonymous rate limit,
so every folder is listed non-recursively and only its direct files are fetched.
Usage: fetch_released_models.py DEST [--models Emu3_Query_based_Action_Expert_PDMS_88.4 Qwen_PDMS_85.3] [--json-zip] [--dry-run]
"""
import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO = "liyingyan/DriveVLA-W0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dest", type=Path)
    ap.add_argument("--models", nargs="*", default=["Emu3_Query_based_Action_Expert_PDMS_88.4", "Qwen_PDMS_85.3"])
    ap.add_argument("--json-zip", action="store_true", help="also fetch flow_matching_json_output_PDMS_87.2.zip (the authors' outputs)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    api = HfApi()
    plan = []
    for model in args.models:
        entries = list(api.list_repo_tree(REPO, path_in_repo=model, recursive=False))
        files = [e for e in entries if getattr(e, "size", None) is not None and not e.path.endswith("/")]
        skipped = [e.path for e in entries if getattr(e, "size", None) is None]
        print(f"{model}: {len(files)} files, {sum(f.size for f in files) / 1e9:.1f} GB; skipped directories: {skipped}")
        plan += [f.path for f in files]
    if args.json_zip:
        plan.append("flow_matching_json_output_PDMS_87.2.zip")
    if args.dry_run:
        print("\n".join(plan))
        return 0
    args.dest.mkdir(parents=True, exist_ok=True)
    for path in plan:
        target = args.dest / path
        if target.exists():
            print(f"exists {path}")
            continue
        print(f"fetching {path}", flush=True)
        hf_hub_download(REPO, path, local_dir=str(args.dest))
    print("FETCH_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
