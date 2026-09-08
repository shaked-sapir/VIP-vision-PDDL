#!/bin/zsh
# Resume after the pause: re-seed the symbolic ROSAME arms (upstream's 8800), then the remaining backfills.
cd /Users/shakedsapir/Documents/BGU/thesis/VIP-vision-PDDL-depot-rerun
source /Users/shakedsapir/Documents/BGU/thesis/VIP-vision-PDDL/venv11/bin/activate
export PYTHONUNBUFFERED=1
SUMMARY=logs/summary.txt
IMG=benchmark/running_results/depot/TO=600__depot_data_from_PV__groundfix
SIM_CELLS=(benchmark/running_results/depot/simulation-final-run__mask=*)
SYMBOLIC_KNOBS=(--batch-size 0 --no-normalize-base-loss --rosame-seed 8800)

step() {  # step <name> <command...>
  local name=$1; shift
  echo "$(date '+%F %T') START $name" | tee -a "$SUMMARY"
  "$@" > "logs/$name.log" 2>&1
  local rc=$?
  echo "$(date '+%F %T') END   $name rc=$rc" | tee -a "$SUMMARY"
}
echo "$(date '+%F %T') RESUMED" | tee -a "$SUMMARY"

step 1b_reseed_symbolic_sim   python -m benchmark.backfill_baseline --baselines rosame_24 rosame_milp_24 rosame_milp_24_tag $SYMBOLIC_KNOBS --experiment-dir $SIM_CELLS --learn-timeout 300 --force --workers 3
step 1c_reseed_symbolic_image python -m benchmark.backfill_baseline --baselines rosame_24 $SYMBOLIC_KNOBS --experiment-dir $IMG --learn-timeout 600 --force --workers 3
step 1d_run_params_seed       python logs/fix_run_params_seed.py

step 2b_backfill_cdps_anchored    python -m benchmark.backfill_cdps --algorithm cdps_anchored --experiment-dir $SIM_CELLS --workers 3
step 3_backfill_milp_sr_gt_none   python -m benchmark.backfill_cdps --algorithm pisam_milp_single_round --milp-config benchmark/pisam_milp.gt-none.yaml --experiment-dir $SIM_CELLS $IMG --workers 3
step 4_backfill_milp_loop_gt_none python -m benchmark.backfill_cdps --algorithm pisam_milp_loop         --milp-config benchmark/pisam_milp.gt-none.yaml --experiment-dir $SIM_CELLS $IMG --workers 3

step 5_backfill_rosame_i_arms     python -m benchmark.backfill_baseline --baselines rosame_i_24 rosame_i_26 rosame_i_milp_24 rosame_i_milp_26 --experiment-dir $IMG --learn-timeout 600
step 7_backfill_rosame_i_26_ep5000 python -m benchmark.backfill_baseline --baselines rosame_i_26 --epochs 5000 --ignore-budget --n-seeds 1 --cells fold0_numtrajs3_gtrate0 --experiment-dir $IMG --learn-timeout 600

echo "$(date '+%F %T') ALL DONE" | tee -a "$SUMMARY"
