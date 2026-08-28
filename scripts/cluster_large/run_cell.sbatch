#!/bin/bash
# Large-corpora array template: one task = one (domain, p_mask, p_noise) cell,
# running the WHOLE L sweep and every arm inside it.
#
# Usage (from the project root on the cluster):
#   python scripts/cluster_large/make_manifest.py
#   mkdir -p logs
#   sbatch --array=0-<N-1>%15 scripts/cluster_large/run_cell.sbatch [manifest.csv]
#
# Conda env: defaults to "vip_venv11"; override with  export VIP_ENV=my_env.
#
# Why this differs from scripts/cluster/run_cell.sbatch (the small-data sweep):
#
#   * It drives benchmark_runner, not experiment_runner. The arms here need the
#     `pisam_milp:` block and its gt_anchoring ablation, which only
#     benchmark_runner builds; experiment_runner's CLI cannot express them.
#     One config file therefore fixes the arms, the L list, patience and subset
#     size for every task.
#   * The L sweep stays INSIDE the task. All five training sizes and all five
#     arms of a cell run on one node, so a wall-clock learning budget compares
#     like with like across L.
#
# Resources, against measurements on the L<=500 local grid:
#   --time 24:00:00  measured 12.2 min/fold for 4 arms at L=500 (0.2/1.1/2.1 at
#     L=10/50/100). The two ROSAME arms scale linearly at ~0.72 s/trajectory, so
#     L=2000 projects to ~24 min/arm; a 5th arm and the 5-L total put a fold near
#     2.5 h. Folds run in parallel, so the cell is bounded by its slowest fold.
#   --mem 64G  the five L=2000 fold workers held ~1 GB combined when measured, so
#     this is deliberate headroom, not a fitted number: exhausting RAM is what
#     killed the local run, and a cell that swaps makes its own timing worthless.
#   --cpus-per-task 12  SLURM counts logical CPUs and ThreadsPerCore=2 here, so
#     12 => 6 physical cores for the 5 fold processes plus the parent.
#   --constraint cpu256  pins the AMD EPYC 7763 nodes: the unconstrained pool
#     spans a 1.9x single-thread range, which makes a wall-clock budget
#     non-comparable across tasks. Never cpu128 -- that is the slow EPYC.
#SBATCH --partition main
#SBATCH --job-name vip-large
#SBATCH --output logs/%x-%A_%a.out
#SBATCH --time 1-00:00:00
#SBATCH --cpus-per-task 12
#SBATCH --constraint cpu256
#SBATCH --gpus=0
#SBATCH --mem 64G

set -euo pipefail
shopt -s failglob   # an empty GT glob must fail loudly, not fall back to images

cd "${SLURM_SUBMIT_DIR}"   # submit from the project root

MANIFEST="${1:-scripts/cluster_large/manifest.csv}"
CONFIG="${VIP_LARGE_CONFIG:-benchmark/run_config_large.yaml}"

# Row for this task: +1 to skip the CSV header, +1 because sed is 1-based.
LINE="$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "$MANIFEST" | tr -d '\r')"
if [ -z "$LINE" ]; then
    echo "ERROR: no manifest row for array index ${SLURM_ARRAY_TASK_ID} in ${MANIFEST}" >&2
    exit 1
fi
IFS=, read -r DOMAIN DATA_DIR P_MASK P_NOISE RUN_NAME <<< "$LINE"

echo "SLURM_JOBID=${SLURM_JOB_ID:-?} node=${SLURM_JOB_NODELIST:-?}"
echo "task=${SLURM_ARRAY_TASK_ID}: domain=${DOMAIN} mask=${P_MASK} noise=${P_NOISE}"
echo "run_name=${RUN_NAME} config=${CONFIG}"

# A wall-clock budget makes results hardware-dependent, so the CPU model belongs
# in the log beside every result.
echo "cpu_model=$(awk -F': ' '/^model name/{print $2; exit}' /proc/cpuinfo)"
echo "cpus_allocated=${SLURM_CPUS_PER_TASK:-?} (logical; ThreadsPerCore=2 here)"
echo "mem_allocated=${SLURM_MEM_PER_NODE:-?} MB"

if [ ! -d "$DATA_DIR/gt_trajectories" ]; then
    echo "ERROR: no gt_trajectories/ under ${DATA_DIR}. Generate the corpus locally first." >&2
    exit 1
fi
echo "corpus_problems=$(ls "$DATA_DIR/gt_trajectories" | wc -l)"

module load anaconda
source activate "${VIP_ENV:-vip_venv11}"

# One BLAS/OpenMP thread per fold process: each fold is its own process, and
# without this numpy spawns a thread per visible core in every one of them.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

python -m benchmark.benchmark_runner \
    --config "$CONFIG" \
    --domains "$DOMAIN" \
    --only-mask "$P_MASK" \
    --only-noise "$P_NOISE"
