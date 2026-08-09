#!/bin/bash
# Fold-level SLURM array template: each task = one (domain, p_mask, p_noise,
# num_trajs, fold) manifest row - a single fold instance as an independent job.
# Requires the --folds flag on benchmark/experiment_runner.py.
#
# Usage (from the project root on the cluster):
#   python scripts/cluster/make_manifest.py --per-fold   # writes manifest_folds.csv + run_flags.sh
#   mkdir -p logs
#   sbatch --array=0-<N-1>%30 scripts/cluster/run_fold.sbatch [manifest_folds.csv]
#
# Conda env: defaults to "vip_venv11"; override with  export VIP_ENV=my_env  before sbatch.
#
# Hardware pinning: see the rationale in run_cell.sbatch. One fold = one CPU-bound
# process, so 2 logical CPUs (= 1 full physical core, ThreadsPerCore=2) is right;
# the constraint is what matters here, to keep the wall-clock budget comparable.
#SBATCH --partition main
#SBATCH --job-name vip-fold
#SBATCH --output logs/%x-%A_%a.out
#SBATCH --time 0-02:00:00
#SBATCH --cpus-per-task 2
#SBATCH --constraint cpu256
#SBATCH --gpus=0
#SBATCH --mem 16G

set -euo pipefail
shopt -s failglob   # empty GT-trajectory glob must fail loudly, not fall back to ImageDataSource

cd "${SLURM_SUBMIT_DIR}"   # submit from the project root

MANIFEST="${1:-scripts/cluster/manifest_folds.csv}"
source "$(dirname "$MANIFEST")/run_flags.sh"   # defines COMMON_FLAGS

# Row for this task: +1 to skip the CSV header, +1 because sed is 1-based.
LINE="$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "$MANIFEST" | tr -d '\r')"
if [ -z "$LINE" ]; then
    echo "ERROR: no manifest row for array index ${SLURM_ARRAY_TASK_ID} in ${MANIFEST}" >&2
    exit 1
fi
IFS=, read -r DOMAIN DATA_DIR P_MASK P_NOISE NUM_TRAJS EXPERIMENT_NAME FOLD <<< "$LINE"

echo "SLURM_JOBID=${SLURM_JOB_ID:-?} node=${SLURM_JOB_NODELIST:-?}"
echo "task=${SLURM_ARRAY_TASK_ID}: domain=${DOMAIN} mask=${P_MASK} noise=${P_NOISE} numtrajs=${NUM_TRAJS} fold=${FOLD}"
echo "experiment_name=${EXPERIMENT_NAME}"

# Record the hardware -- a wall-clock search budget makes results hardware-dependent.
echo "cpu_model=$(awk -F': ' '/^model name/{print $2; exit}' /proc/cpuinfo)"
echo "cpus_allocated=${SLURM_CPUS_PER_TASK:-?} (logical; ThreadsPerCore=2 on this cluster)"

module load anaconda
source activate "${VIP_ENV:-vip_venv11}"

# Keep BLAS/OpenMP single-threaded so nothing oversubscribes the allocation.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# shellcheck disable=SC2086  # COMMON_FLAGS is intentionally word-split
python -m benchmark.experiment_runner \
    --domain "$DOMAIN" \
    --data-dir "$DATA_DIR" \
    --num-trajectories "$NUM_TRAJS" \
    --folds "$FOLD" \
    --experiment-name "$EXPERIMENT_NAME" \
    --simulated-gt-trajectories "$DATA_DIR"/gt_trajectories/problem*/*.trajectory \
    --simulated-masking-p "$P_MASK" \
    --simulated-noising-p "$P_NOISE" \
    ${COMMON_FLAGS}
