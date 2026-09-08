#!/bin/zsh
# Depot re-run in the worktree: main runs, then the backfills that give every arm the originals have.
cd /Users/shakedsapir/Documents/BGU/thesis/VIP-vision-PDDL-depot-rerun
source /Users/shakedsapir/Documents/BGU/thesis/VIP-vision-PDDL/venv11/bin/activate
export PYTHONUNBUFFERED=1
SUMMARY=logs/summary.txt
: > "$SUMMARY"
IMG=benchmark/running_results/depot/TO=600__depot_data_from_PV__groundfix

step() {  # step <name> <command...>
  local name=$1; shift
  echo "$(date '+%F %T') START $name" | tee -a "$SUMMARY"
  "$@" > "logs/$name.log" 2>&1
  local rc=$?
  echo "$(date '+%F %T') END   $name rc=$rc" | tee -a "$SUMMARY"
}

step 1_sim_run    python -m benchmark.benchmark_runner --config benchmark/run_config.depot-sim.yaml
step 2_image_run  python -m benchmark.benchmark_runner --config benchmark/run_config.depot-image.yaml

SIM_CELLS=(benchmark/running_results/depot/simulation-final-run__mask=*)
step 2b_backfill_cdps_anchored    python -m benchmark.backfill_cdps --algorithm cdps_anchored --experiment-dir $SIM_CELLS --workers 3
step 3_backfill_milp_sr_gt_none   python -m benchmark.backfill_cdps --algorithm pisam_milp_single_round --milp-config benchmark/pisam_milp.gt-none.yaml --experiment-dir $SIM_CELLS $IMG --workers 3
step 4_backfill_milp_loop_gt_none python -m benchmark.backfill_cdps --algorithm pisam_milp_loop         --milp-config benchmark/pisam_milp.gt-none.yaml --experiment-dir $SIM_CELLS $IMG --workers 3

step 5_backfill_rosame_i_arms     python -m benchmark.backfill_baseline --baselines rosame_i_24 rosame_i_26 rosame_i_milp_24 rosame_i_milp_26 --experiment-dir $IMG --learn-timeout 600
step 6_backfill_rosame_i_24_res64 python -m benchmark.backfill_baseline --baselines rosame_i_24 --resize 64 --experiment-dir $IMG --learn-timeout 600
step 7_backfill_rosame_i_26_ep5000 python -m benchmark.backfill_baseline --baselines rosame_i_26 --epochs 5000 --ignore-budget --experiment-dir $IMG --learn-timeout 600

echo "$(date '+%F %T') ALL DONE" | tee -a "$SUMMARY"
