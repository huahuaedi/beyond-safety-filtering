# Beyond Safety Filtering: How Training Targets Shape Driving Policy Improvement

Code, analysis scripts, and paper sources for a controlled study of the *winner rule* among safety-qualified
trajectories (NC = DAC = TTC = 1) when post-training the frozen-backbone flow-matching policy
[DriveVLA-W0](https://github.com/BraveGroup/DriveVLA-W0) on NAVSIM navtest (12,146 scenes, 136 logs).
Candidates, safety gates, loser construction, and fitting are held fixed; only the winner rule changes
(highest PDMS, maximum obstacle clearance, nearest to a reference, uniformly random among eligible, progress-capped).

## Layout

| Path | Contents |
|---|---|
| `tools/rl_prep/` | Pair construction, offline fitting, evaluation sharding/merging, paired log-cluster bootstrap (`paired_scores_by_log.py`), repair/new-failure ledger (`build_ledger.py`), target statistics, fit-error measurement, privileged rankers, unit tests |
| `scripts/hpc/` | Slurm/bash drivers used on the HPC cluster (training, 12-shard navtest evaluation, NAVSIM v2 re-scoring, controls, audit recomputation) |
| `docs/` | `paper_ral.tex`, `paper_ral_supplement.tex`, figure sources and plotting scripts, `results_master.md` (number-to-output map), `reproduction_index.md` |
| `results_summary/` | Small summary outputs and figures |
| `patches/` | Minimal changes to upstream DriveVLA-W0 files (see below) |

## Setup

This repository contains only our additions. It runs on top of an unmodified DriveVLA-W0 checkout:

```bash
git clone https://github.com/BraveGroup/DriveVLA-W0 && cd DriveVLA-W0
git checkout a55766c                      # upstream commit the patch was made against
git apply /path/to/this/repo/patches/drivevla_w0_a55766c.diff
cp -r /path/to/this/repo/{tools,scripts,configs,docs,results_summary} .
```

Follow the upstream instructions for the environment, released weights, NAVSIM data, and a NAVSIM v1.1 checkout.
The Slurm scripts hard-code our cluster's account, partitions, and data paths; edit the variables at the top of
each script (`OFFLINE_REPO_ROOT`, `DRIVEVLA_REPO_ROOT`, data roots) before use.

## Reproducing the main results

- Training and evaluation: `scripts/hpc/submit_navtest_policy_eval.sh` (sharded navtest evaluation), merged with `tools/rl_prep/merge_policy_eval_shards.py --expect-scenes 12146`.
- Paired comparisons: `tools/rl_prep/paired_scores_by_log.py`; ledgers: `tools/rl_prep/build_ledger.py`.
- Random-winner control: `scripts/hpc/submit_random_winner.sh`, `scripts/hpc/run_randwin_navtest.sh`.
- NAVSIM v2 re-scoring: `scripts/hpc/run_v2_rescore_all.sh`.
- Recomputing reported tables from the score CSVs: `scripts/hpc/audit_recompute_tables.py`.
- Number-to-output mapping: `docs/results_master.md`, `docs/reproduction_index.md`.

## Acknowledgments

Built on DriveVLA-W0 (Li et al., ICLR 2026) and NAVSIM. Please cite and follow the licenses of those projects.
