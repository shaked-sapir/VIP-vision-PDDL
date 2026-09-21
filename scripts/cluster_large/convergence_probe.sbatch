#!/bin/bash
# Calibration probe for the ROSAME loss-convergence rule
# (docs/rosame-training-convergence-fix.md, section 6).
#
# One job per domain: the DL-only arm (rosame_24) and the MILP arm
# (rosame_milp_24), pooled training, the convergence rule OFF and agreement not a
# stop rule, a FIXED number of epochs, at L = 10, 100 and 2000, fold 0 of one
# mid-grid cell. Each arm writes its per-epoch loss series (and, for the MILP
# arm, the base and CE terms and the first-solve epoch) to
# <fold>/rosame_training/<arm>.json, which is everything the calibration reads:
# where the curve plateaus, how large its epoch-to-epoch upticks are, the jump at
# the first solve, and seconds per epoch at L=2000. No model snapshots.
#
#   for d in blocksworld hanoi npuzzle depot gripper; do
#     sbatch --export=ALL,PROBE_DOMAIN=$d scripts/cluster_large/convergence_probe.sbatch
#   done
#
# The learning budget is raised well above the sweep's 1 h so the fixed epoch
# count is what ends training, not the timeout: the point is to see the whole
# curve, including how long an epoch takes when nothing cuts it short.
#SBATCH --partition main
#SBATCH --job-name conv-probe
#SBATCH --output logs/%x-%j.out
#SBATCH --time 1-12:00:00
#SBATCH --cpus-per-task 6
#SBATCH --constraint cpu256
#SBATCH --gpus=0
#SBATCH --mem 24G

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

DOM="${PROBE_DOMAIN:?set PROBE_DOMAIN}"
EPOCHS="${PROBE_EPOCHS:-500}"
P_MASK="${PROBE_MASK:-0.1}"
P_NOISE="${PROBE_NOISE:-0.2}"
LEARN_TIMEOUT="${PROBE_LEARN_TIMEOUT:-21600}"
RUN_NAME="convprobe__ep${EPOCHS}"

echo "SLURM_JOBID=${SLURM_JOB_ID:-?} node=${SLURM_JOB_NODELIST:-?}"
echo "conv-probe: domain=$DOM epochs=$EPOCHS L=10,100,2000 mask=$P_MASK noise=$P_NOISE fold=0 learn_timeout=${LEARN_TIMEOUT}s"
echo "cpu_model=$(awk -F': ' '/^model name/{print $2; exit}' /proc/cpuinfo)"

module load anaconda
source activate "${VIP_ENV:-vip_venv11}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

CFG="$(mktemp /tmp/convprobe_XXXX.yaml)"
python - "$EPOCHS" "$LEARN_TIMEOUT" "$RUN_NAME" "$CFG" <<'PY'
import sys, yaml
epochs, timeout, run_name, dst = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]
c = yaml.safe_load(open("benchmark/run_config_large.yaml"))
s = c["shared"]
c["run_name"] = run_name
s["num_trajectories"] = [10, 100, 2000]
s["algorithms"] = ["rosame_24", "rosame_milp_24"]
s["train_per_trajectory"] = False
s["epochs"] = epochs
s["pre_mip_epochs"] = 50
s["agreement_stop"] = None
s.pop("rosame_convergence", None)
s.pop("snapshot_interval", None)
s["learning_timeout_seconds"] = timeout
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
