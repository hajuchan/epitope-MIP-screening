#!/bin/bash
# Driver robustness checks for review findings 8, 10, 11, 12.
#
# These re-create each finding's exact failure scenario against the REAL driver.
# Trajectory budget: only CD63|dual|0| is ever selected — the two legs already
# exercised by the anchor gate. No other trajectory is opened.
#
#   bash code/test_driver_robustness.sh [8|10|11|12|all]
source ~/anaconda3/etc/profile.d/conda.sh
conda activate MIPscreen
cd /home/chan/Research/Monomer_screening_in_Bio || exit 1
BASE=/home/chan/Research/Monomer_screening_in_Bio/reports_v2/_robust
SEL='CD63|dual|0|'
WHICH=${1:-all}

hdr() { echo; echo "================================================================"; echo "$1"; echo "================================================================"; }

# ---------------------------------------------------------------- finding 8 --
if [ "$WHICH" = "8" ] || [ "$WHICH" = "all" ]; then
hdr "FINDING 8: kill -9 one worker mid-run; the sweep must still finish"
OUT=$BASE/f8; rm -rf $OUT; mkdir -p $OUT
python code/run_pcsi_star.py --out-dir $OUT --only "$SEL" --workers 2 \
    --skip-anchor-gate > $OUT/run.log 2>&1 &
PID=$!
sleep 5
CH=$(pgrep -P $PID | head -1)
echo "parent=$PID  killing ONE worker child=$CH  (simulating the OOM killer)"
kill -9 "$CH" 2>/dev/null
wait $PID; RC=$?
echo "PARENT EXIT CODE = $RC"
echo "--- pool recovery lines ---"
grep -E "worker pool died|rebuilding|compute wall" $OUT/run.log
echo "--- outputs produced? ---"
ls -la $OUT/pcsi_star_summary.json $OUT/pcsi_star_report.txt 2>&1 | sed 's/^/  /'
echo "--- checkpoint ---"
python - <<PY
import json,os
p="$OUT/pcsi_star_legs_v2.jsonl"
if os.path.exists(p):
    for l in open(p):
        r=json.loads(l); print("  ", r["key"], "ok=", r["ok"])
else: print("  NO CHECKPOINT FILE")
PY
fi

# --------------------------------------------------------------- finding 10 --
if [ "$WHICH" = "10" ] || [ "$WHICH" = "all" ]; then
hdr "FINDING 10: a torn checkpoint line must not make the driver unstartable"
OUT=$BASE/f10; rm -rf $OUT; mkdir -p $OUT
python code/run_pcsi_star.py --out-dir $OUT --only "$SEL" --workers 2 \
    --skip-anchor-gate > $OUT/first.log 2>&1
echo "seeded checkpoint: $(wc -l < $OUT/pcsi_star_legs_v2.jsonl) lines"
python - <<PY
p="$OUT/pcsi_star_legs_v2.jsonl"
ls=open(p).read().splitlines()
ls[0]=ls[0][:len(ls[0])//2]          # tear line 1 in half
open(p,"w").write("\n".join(ls)+"\n")
print(f"  tore line 1 to {len(ls[0])} chars; total lines {len(ls)}")
PY
echo "--- aggregate-only on the torn checkpoint ---"
python code/run_pcsi_star.py --out-dir $OUT --aggregate-only > $OUT/agg.log 2>&1
echo "EXIT=$?"
grep -E "WARNING: checkpoint line|checkpoint:" $OUT/agg.log
echo "--- normal resume on the torn checkpoint ---"
python code/run_pcsi_star.py --out-dir $OUT --only "$SEL" --workers 2 \
    --skip-anchor-gate > $OUT/resume.log 2>&1
echo "EXIT=$?"
grep -E "WARNING: checkpoint line|checkpoint:|legs present|compute wall" $OUT/resume.log
fi

# --------------------------------------------------------------- finding 11 --
if [ "$WHICH" = "11" ] || [ "$WHICH" = "all" ]; then
hdr "FINDING 11: two instances on one --out-dir must be refused, not duplicated"
OUT=$BASE/f11; rm -rf $OUT; mkdir -p $OUT
python code/run_pcsi_star.py --out-dir $OUT --only "$SEL" --workers 2 \
    --skip-anchor-gate > $OUT/a.log 2>&1 &
A=$!
sleep 2
python code/run_pcsi_star.py --out-dir $OUT --only "$SEL" --workers 2 \
    --skip-anchor-gate > $OUT/b.log 2>&1
echo "instance B exit=$?  (4 = refused by the lock)"
wait $A; echo "instance A exit=$?"
echo "--- B's message ---"; sed 's/^/  /' $OUT/b.log | head -6
echo "--- checkpoint: lines vs unique keys ---"
python - <<PY
import json
p="$OUT/pcsi_star_legs_v2.jsonl"
ok=bad=0; keys=[]
for i,l in enumerate(open(p),1):
    if not l.strip(): continue
    try: keys.append(json.loads(l)["key"]); ok+=1
    except Exception as e: bad+=1; print(f"  MALFORMED LINE {i}: {e}")
print(f"  parseable lines={ok} malformed={bad} unique keys={len(set(keys))} "
      f"duplicate records={len(keys)-len(set(keys))}")
PY
fi

# --------------------------------------------------------------- finding 12 --
if [ "$WHICH" = "12" ] || [ "$WHICH" = "all" ]; then
hdr "FINDING 12: --cutoff outside the grid must be refused up front"
OUT=$BASE/f12; rm -rf $OUT; mkdir -p $OUT
cp $BASE/f8/pcsi_star_legs_v2.jsonl $OUT/ 2>/dev/null
echo "--- --cutoff 8 (invalid) ---"
python code/run_pcsi_star.py --out-dir $OUT --aggregate-only --cutoff 8 2>&1 | tail -3
echo "EXIT=${PIPESTATUS[0]}"
echo "--- --persistence 0.55 (invalid) ---"
python code/run_pcsi_star.py --out-dir $OUT --aggregate-only --persistence 0.55 2>&1 | tail -2
echo "EXIT=${PIPESTATUS[0]}"
echo "--- --cutoff 7 (valid) ---"
python code/run_pcsi_star.py --out-dir $OUT --aggregate-only --cutoff 7 > $OUT/ok.log 2>&1
echo "EXIT=$?"
grep -E "^wrote" $OUT/ok.log
fi
