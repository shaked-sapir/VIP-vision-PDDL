#!/bin/zsh
# Final continuation: wait for step 4 (PID 50776), then the ROSAME-I arms with 5 workers, then the ep=5000 control cell.
cd /Users/shakedsapir/Documents/BGU/thesis/VIP-vision-PDDL-depot-rerun
source /Users/shakedsapir/Documents/BGU/thesis/VIP-vision-PDDL/venv11/bin/activate
export PYTHONUNBUFFERED=1
SUMMARY=logs/summary.txt
IMG=benchmark/running_results/depot/TO=600__depot_data_from_PV__groundfix
while kill -0 50776 2>/dev/null; do sleep 30; done
n=$(python3 -c "import json,glob; fs=glob.glob('benchmark/running_results/depot/*/testing/fold*/fold_result.json'); print(sum(any(r['algorithm']=='PISAM_MILP_LOOP__gt=none' for r in json.load(open(f))) for f in fs))")
if [ "$n" -eq 300 ]; then rc=0; else rc=1; fi
echo "$(date '+%F %T') END   4_backfill_milp_loop_gt_none rc=$rc ($n/300 rows present)" | tee -a "$SUMMARY"

step() {  # step <name> <command...>
  local name=$1; shift
  echo "$(date '+%F %T') START $name" | tee -a "$SUMMARY"
  "$@" > "logs/$name.log" 2>&1
  local rc=$?
  echo "$(date '+%F %T') END   $name rc=$rc" | tee -a "$SUMMARY"
}

step 5_backfill_rosame_i_arms      python -m benchmark.backfill_baseline --baselines rosame_i_24 rosame_i_26 rosame_i_milp_24 rosame_i_milp_26 --experiment-dir $IMG --learn-timeout 600 --workers 5
step 7_backfill_rosame_i_26_ep5000 python -m benchmark.backfill_baseline --baselines rosame_i_26 --epochs 5000 --ignore-budget --n-seeds 1 --cells fold0_numtrajs3_gtrate0 --experiment-dir $IMG --learn-timeout 600

echo "$(date '+%F %T') ALL DONE" | tee -a "$SUMMARY"
