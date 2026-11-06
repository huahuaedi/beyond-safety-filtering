#!/bin/bash
# Mac launcher. submit: tar this package, copy it to the HPC and run its submit.sh (CPU audit -> SFT smoke -> 23 evaluations -> report).
# fetch : download logs and per-scene score tables once jobs finished.
set -euo pipefail
task_dir=$(cd "$(dirname "$0")" && pwd)
task_receipt="$task_dir/submission.txt"
case "${1:-submit}" in
 submit)
 [[ ! -e "$task_receipt" ]] || { echo 'Receipt exists; refusing duplicate submission'; exit 1; }
 task_remote="/home/huamin/orig_five_navtest_$(date -u +%Y%m%d_%H%M%S)_$$"
 printf -v task_q '%q' "$task_remote"
 echo "package=$task_remote" > "$task_receipt"
 tar -czf - -C "$task_dir" --exclude='__pycache__' --exclude='submission.txt' --exclude='run.sh' . \
   | ssh "${HPC_USER}@${HPC_HOST}" "set -e; mkdir $task_q; tar -xzf - -C $task_q; cd $task_q; bash submit.sh" | tee -a "$task_receipt"
 echo 'Submitted: CPU audit -> SFT smoke -> evaluations (max 5 GPUs) -> report. No training.'
 ;;
 fetch)
 task_remote=$(sed -n 's/^package=//p' "$task_receipt" | head -1)
 [[ "$task_remote" == /home/huamin/orig_five_navtest_* ]] || exit 1
 printf -v task_q '%q' "$task_remote"
 mkdir -p "$HOME/Downloads/risk_replay_input"
 task_target=$(mktemp -d "$HOME/Downloads/risk_replay_input/orig_five_navtest_results.XXXXXX")
 ssh "${HPC_USER}@${HPC_HOST}" "python3 $task_q/export.py $task_q" > "$task_target/results.tar.gz.part"
 tar -tzf "$task_target/results.tar.gz.part" >/dev/null
 mv "$task_target/results.tar.gz.part" "$task_target/results.tar.gz"
 tar -xzf "$task_target/results.tar.gz" -C "$task_target"
 echo "Downloaded results/logs to: $task_target"
 ;;
 *) echo 'Usage: bash run.sh [submit|fetch]'; exit 2 ;;
esac
