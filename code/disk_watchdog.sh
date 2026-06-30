#!/usr/bin/env bash
# Disk watchdog — pauses the MIP pipeline if free disk drops below THRESHOLD_GB.
# Uses SIGTERM (graceful) → SIGKILL fallback. Preserves GROMACS checkpoints.
# Persists across pipeline restarts. Stop with: pkill -f disk_watchdog.sh

THRESHOLD_GB=50
POLL_S=120
ROOT=/home/chan/Research/Monomer_screening_in_Bio
LOG="$ROOT/results/logs/disk_watchdog.log"

mkdir -p "$ROOT/results/logs"
ts() { date '+%Y-%m-%d %H:%M:%S'; }

echo "[$(ts)] watchdog START  threshold=${THRESHOLD_GB}GB  poll=${POLL_S}s  pid=$$" >> "$LOG"

while true; do
    AVAIL_GB=$(df --output=avail -BG / | tail -1 | tr -dc '0-9')
    if [ -z "$AVAIL_GB" ]; then
        echo "[$(ts)] WARN: df parse failed, skip" >> "$LOG"
        sleep "$POLL_S"; continue
    fi

    if [ "$AVAIL_GB" -lt "$THRESHOLD_GB" ]; then
        # Only act if the pipeline is actually running.
        if pgrep -f "pipeline.run_pipeline" >/dev/null 2>&1 \
           || pgrep -f "gmx mdrun" >/dev/null 2>&1; then
            echo "[$(ts)] *** CRITICAL: ${AVAIL_GB}GB free < ${THRESHOLD_GB}GB \
— stopping pipeline ***" >> "$LOG"
            pkill -TERM -f "pipeline.run_pipeline" 2>>"$LOG"
            pkill -TERM -f "gmx mdrun" 2>>"$LOG"
            sleep 8
            pkill -KILL -f "pipeline.run_pipeline" 2>>"$LOG"
            pkill -KILL -f "gmx mdrun" 2>>"$LOG"
            echo "[$(ts)] pipeline stopped. Free up disk before resuming." >> "$LOG"
            # Don't exit — keep watching so we don't auto-resume into the same trap.
        else
            echo "[$(ts)] LOW: ${AVAIL_GB}GB free (pipeline idle, no action)" >> "$LOG"
        fi
    fi
    sleep "$POLL_S"
done
