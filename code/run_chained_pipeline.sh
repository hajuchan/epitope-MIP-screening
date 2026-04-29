#!/bin/bash
# Chain: crosslinker sweep → Phase 5 extended
# Runs sequentially since both compete for the same GPU.

set -u
cd "$(dirname "$0")/.."

PYTHON=/home/chan/anaconda3/envs/MIPscreen/bin/python3
LOGDIR=results/logs
mkdir -p "$LOGDIR"

STAMP=$(date +%Y%m%d_%H%M)

# Wait for any previous crosslinker_sweep python to finish
echo "[$(date)] Waiting for any in-progress sweep python to finish..."
while pgrep -f "run_crosslinker_sweep.py" > /dev/null; do
  sleep 60
done

echo "[$(date)] Crosslinker sweep done."

# Phase 5 extended
echo "[$(date)] Starting Phase 5 extended (CD63+CD9, N=10, 50ns)..."
$PYTHON -u code/run_phase5_extended.py > "$LOGDIR/phase5_extended_${STAMP}.log" 2>&1
echo "[$(date)] Phase 5 extended done."
