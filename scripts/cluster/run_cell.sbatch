#!/bin/bash
# Cell-level SLURM array template: each task = one (domain, p_mask, p_noise,
# num_trajs) manifest row, running all 5 folds in parallel in-process
# (experiment_runner's ProcessPoolExecutor) — hence 6 CPUs per task.
#
# Usage (from the project root on the cluster):
#   python scripts/cluster/make_manifest.py         # writes manifest.csv + run_flags.sh
#   mkdir -p logs
#   sbatch --array=0-<N-1>%15 scripts/cluster/run_cell.sbatch [manifest.csv]
#
# Conda env: defaults to "vip_venv11"; override with  export VIP_ENV=my_env  before sbatch.
#
#SBATCH --partition main
#SBATCH --job-name vip-cell
#SBATCH --output logs/%x-%A_%a.out
#SBATCH --time 0-06:00:00
#SBATCH --cpus-per-task 6
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

module load anaconda
source activate "${VIP_ENV:-vip_venv11}"

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
