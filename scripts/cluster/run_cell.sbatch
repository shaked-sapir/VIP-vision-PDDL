#!/bin/bash
# Cell-level SLURM array template: each task = one (domain, p_mask, p_noise,
# num_trajs) manifest row, running all 5 folds in parallel in-process
# (experiment_runner's ProcessPoolExecutor) - hence 12 logical CPUs per task.
#
# Usage (from the project root on the cluster):
#   python scripts/cluster/make_manifest.py         # writes manifest.csv + run_flags.sh
#   mkdir -p logs
#   sbatch --array=0-<N-1>%15 scripts/cluster/run_cell.sbatch [manifest.csv]
#
# Conda env: defaults to "vip_venv11"; override with  export VIP_ENV=my_env  before sbatch.
#
# Hardware pinning (see the measurements in the header comment of submit.sh):
#   --constraint cpu256  pins to the AMD EPYC 7763 nodes, the fastest single-thread
#     class on this cluster and the only uniform one. The unconstrained `cpu` pool
#     spans a 1.9x single-thread range (Xeon E5-2680 v2 ~3160 iters/s down to
#     EPYC 7702P / Xeon E5-2680 v3 ~1990), which makes a wall-clock search budget
#     non-comparable across array tasks. Never use `cpu128` -- it is the slow EPYC.
#   --cpus-per-task 12   SLURM counts *logical* CPUs and every node here has
#     ThreadsPerCore=2, so 12 => 6 physical cores for the 5 fold processes plus
#     the parent. The previous value of 6 gave only 3 physical cores to 5 CPU-bound
#     processes, costing roughly another 1.5x on top of the raw clock gap.
#SBATCH --partition main
#SBATCH --job-name vip-cell
#SBATCH --output logs/%x-%A_%a.out
#SBATCH --time 0-06:00:00
#SBATCH --cpus-per-task 12
#SBATCH --constraint cpu256
#SBATCH --gpus=0
#SBATCH --mem 24G

set -euo pipefail
shopt -s failglob   # empty GT-trajectory glob must fail loudly, not fall back to ImageDataSource

cd "${SLURM_SUBMIT_DIR}"   # submit from the project root

MANIFEST="${1:-scripts/cluster/manifest.csv}"
source "$(dirname "$MANIFEST")/run_flags.sh"   # defines COMMON_FLAGS

# Row for this task: +1 to skip the CSV header, +1 because sed is 1-based.
LINE="$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "$MANIFEST" | tr -d '\r')"
if [ -z "$LINE" ]; then
    echo "ERROR: no manifest row for array index ${SLURM_ARRAY_TASK_ID} in ${MANIFEST}" >&2
    exit 1
fi
IFS=, read -r DOMAIN DATA_DIR P_MASK P_NOISE NUM_TRAJS EXPERIMENT_NAME <<< "$LINE"

echo "SLURM_JOBID=${SLURM_JOB_ID:-?} node=${SLURM_JOB_NODELIST:-?}"
echo "task=${SLURM_ARRAY_TASK_ID}: domain=${DOMAIN} mask=${P_MASK} noise=${P_NOISE} numtrajs=${NUM_TRAJS}"
echo "experiment_name=${EXPERIMENT_NAME}"

# Record the hardware. A wall-clock search budget makes results hardware-dependent,
# so the CPU model belongs in the log next to every result.
echo "cpu_model=$(awk -F': ' '/^model name/{print $2; exit}' /proc/cpuinfo)"
echo "cpus_allocated=${SLURM_CPUS_PER_TASK:-?} (logical; ThreadsPerCore=2 on this cluster)"

module load anaconda
source activate "${VIP_ENV:-vip_venv11}"

# One BLAS/OpenMP thread per fold process. Each of the 5 folds is its own process;
# without this, numpy can spawn one thread per visible core in each of them and
# oversubscribe the allocation many times over.
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
    --experiment-name "$EXPERIMENT_NAME" \
    --simulated-gt-trajectories "$DATA_DIR"/gt_trajectories/problem*/*.trajectory \
    --simulated-masking-p "$P_MASK" \
    --simulated-noising-p "$P_NOISE" \
    ${COMMON_FLAGS}
