#!/bin/bash
# Runs a python script while watching system available memory; kills it if
# available memory drops below MIN_AVAIL_MB, instead of letting the kernel
# OOM-killer pick an arbitrary victim process on a shared machine.
set -u
MIN_AVAIL_MB=3000
POLL_SECONDS=2

PY_SCRIPT="$1"
shift

python "$PY_SCRIPT" "$@" &
PID=$!
echo "Launched PID=$PID, memory-guarding at ${MIN_AVAIL_MB}MB available, polling every ${POLL_SECONDS}s"

while kill -0 "$PID" 2>/dev/null; do
    AVAIL_MB=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
    if [ "$AVAIL_MB" -lt "$MIN_AVAIL_MB" ]; then
        echo "!!! Available memory ${AVAIL_MB}MB < ${MIN_AVAIL_MB}MB threshold — killing PID=$PID"
        kill -9 "$PID"
        wait "$PID" 2>/dev/null
        echo "MEM_GUARD_KILLED"
        exit 1
    fi
    sleep "$POLL_SECONDS"
done

wait "$PID"
EXIT_CODE=$?
echo "Process finished with exit code $EXIT_CODE"
exit $EXIT_CODE
