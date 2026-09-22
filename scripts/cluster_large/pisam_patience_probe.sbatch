#!/bin/bash
# Does the PI-SAM loop's auto patience stop it early at L = 2000?
# (docs/large-corpora-experiments-plan.md, sections 3.4 and 6.)
#
# The sweep stops the loop after ceil(7 * ln C(L, 4)) rounds without a V
# improvement: 191 at L = 2000. That schedule was fitted on folds of 5 to 8
# traces. This job runs both gt_anchoring variants of the loop at L = 2000 with
# the patience OFF and a 2 h budget, fold 0 of one mid-grid cell, so the round
# log (<fold>/pisam_milp_loop*/milp_loop_rounds.json) shows where improvements
# actually arrive: if the last one falls well inside the first 191 rounds the
# rule is not binding; if improvements keep arriving past it, it is.
#
#   sbatch --export=ALL,PROBE_DOMAIN=hanoi scripts/cluster_large/pisam_patience_probe.sbatch
#SBATCH --partition main
#SBATCH --job-name pisam-probe
#SBATCH --output logs/%x-%j.out
#SBATCH --time 0-12:00:00
#SBATCH --cpus-per-task 6
#SBATCH --constraint cpu256
#SBATCH --gpus=0
#SBATCH --mem 24G

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

DOM="${PROBE_DOMAIN:?set PROBE_DOMAIN}"
L="${PROBE_L:-2000}"
P_MASK="${PROBE_MASK:-0.1}"
P_NOISE="${PROBE_NOISE:-0.2}"
BUDGET="${PROBE_BUDGET:-7200}"
RUN_NAME="pisamprobe__L${L}__nopatience"

echo "SLURM_JOBID=${SLURM_JOB_ID:-?} node=${SLURM_JOB_NODELIST:-?}"
echo "pisam-probe: domain=$DOM L=$L mask=$P_MASK noise=$P_NOISE fold=0 patience=off budget=${BUDGET}s"
echo "cpu_model=$(awk -F': ' '/^model name/{print $2; exit}' /proc/cpuinfo)"

module load anaconda
source activate "${VIP_ENV:-vip_venv11}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

CFG="$(mktemp /tmp/pisamprobe_XXXX.yaml)"
python - "$L" "$BUDGET" "$RUN_NAME" "$CFG" <<'PY'
import sys, yaml
L, budget, run_name, dst = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]
c = yaml.safe_load(open("benchmark/run_config_large.yaml"))
s = c["shared"]
c["run_name"] = run_name
s["num_trajectories"] = [L]
s["algorithms"] = ["pisam_milp_loop"]          # the gt_anchoring ablation keeps both variants
s["learning_timeout_seconds"] = budget
s["pisam_milp"]["stop"]["no_improvement_rounds"] = None
s["pisam_milp"]["stop"]["budget_seconds"] = None  # inherit the budget above
s["resume"] = True
yaml.safe_dump(c, open(dst, "w"), sort_keys=False)
PY

RUN_RC=0
python -m benchmark.benchmark_runner --config "$CFG" \
    --domains "$DOM" --only-mask "$P_MASK" --only-noise "$P_NOISE" --only-fold 0 || RUN_RC=$?
rm -f "$CFG"

source scripts/cluster_large/pack_fold.sh
pack_fold_instances "benchmark/running_results/${DOM}/${RUN_NAME}__mask=${P_MASK}__noise=${P_NOISE}" 0

CGROUP_REL="$(awk -F: '{print $3}' /proc/self/cgroup | head -1)"
CGROUP="/sys/fs/cgroup${CGROUP_REL%/step_*}"
echo "cgroup_peak_mb=$(( $(cat "$CGROUP/memory.peak" 2>/dev/null || echo 0) / 1048576 )) of ${SLURM_MEM_PER_NODE:-?}MB"
exit "$RUN_RC"
