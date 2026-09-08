#!/bin/zsh
# Continues the depot re-run after step 1 (whose runner, PID 9207, keeps running): image run, then the backfills.
cd /Users/shakedsapir/Documents/BGU/thesis/VIP-vision-PDDL-depot-rerun
source /Users/shakedsapir/Documents/BGU/thesis/VIP-vision-PDDL/venv11/bin/activate
export PYTHONUNBUFFERED=1
SUMMARY=logs/summary.txt
IMG=benchmark/running_results/depot/TO=600__depot_data_from_PV__groundfix
while kill -0 9207 2>/dev/null; do sleep 60; done
if ls benchmark/finished_run_configs/simulation-final-run/*.json >/dev/null 2>&1; then rc=0; else rc=1; fi
echo "$(date '+%F %T') END   1_sim_run rc=$rc (manifest check)" | tee -a "$SUMMARY"

step() {  # step <name> <command...>
  local name=$1; shift
  echo "$(date '+%F %T') START $name" | tee -a "$SUMMARY"
  "$@" > "logs/$name.log" 2>&1
  local rc=$?
  echo "$(date '+%F %T') END   $name rc=$rc" | tee -a "$SUMMARY"
}

step 2_image_run  python -m benchmark.benchmark_runner --config benchmark/run_config.depot-image.yaml

SIM_CELLS=(benchmark/running_results/depot/simulation-final-run__mask=*)
step 2b_backfill_cdps_anchored    python -m benchmark.backfill_cdps --algorithm cdps_anchored --experiment-dir $SIM_CELLS --workers 3
step 3_backfill_milp_sr_gt_none   python -m benchmark.backfill_cdps --algorithm pisam_milp_single_round --milp-config benchmark/pisam_milp.gt-none.yaml --experiment-dir $SIM_CELLS $IMG --workers 3
step 4_backfill_milp_loop_gt_none python -m benchmark.backfill_cdps --algorithm pisam_milp_loop         --milp-config benchmark/pisam_milp.gt-none.yaml --experiment-dir $SIM_CELLS $IMG --workers 3

step 5_backfill_rosame_i_arms     python -m benchmark.backfill_baseline --baselines rosame_i_24 rosame_i_26 rosame_i_milp_24 rosame_i_milp_26 --experiment-dir $IMG --learn-timeout 600
step 7_backfill_rosame_i_26_ep5000 python -m benchmark.backfill_baseline --baselines rosame_i_26 --epochs 5000 --ignore-budget --n-seeds 1 --cells fold0_numtrajs3_gtrate0 --experiment-dir $IMG --learn-timeout 600

echo "$(date '+%F %T') ALL DONE" | tee -a "$SUMMARY"
