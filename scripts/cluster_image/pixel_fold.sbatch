#!/bin/bash
### One task = one manifest row (domain, fold) of a small image experiment: the
### pixel arms named in $VIP_PX_BASELINES on that fold's frozen observations and
### frames (benchmark/backfill_baseline.py --out-root), over the fold's whole
### training-size sweep, on one GPU.
###
### Submit from the project root, picking the GPU type on the command line:
###   VIP_PX_BASELINES="rosame_i_24 rosame_i_milp_24 rosame_i_26 rosame_i_milp_26" \
###   VIP_PX_ARGS="--learn-timeout 300" \
###   sbatch --constraint rtx_4090 --array=0-24 scripts/cluster_image/pixel_fold.sbatch
###
### Reads $VIP_PX_SOURCE_ROOT/<domain>/<experiment>/ (default
### benchmark/running_results_image_src, the slim copy the README describes) and
### writes only under $VIP_PX_OUT (default benchmark/running_results_image_pixel).
### Two arrays that share cells must not share an out root: give each its own
### $VIP_PX_OUT.
###   VIP_PX_DEVICE   torch device (default cuda; the job fails at start-up
###                   when the node cannot provide it)
###   VIP_PX_ARGS     further backfill_baseline flags (--epochs, --budget-mode,
###                   --learn-timeout, --force, ...)
###   VIP_ENV         conda env (default vip_venv11)
#SBATCH --partition main
#SBATCH --job-name vip-px
#SBATCH --output logs/%x-%A_%a.out
#SBATCH --time 12:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task 4
#SBATCH --mem 24G

set -euo pipefail
shopt -s failglob

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

MANIFEST="${1:-scripts/cluster_image/pixel_fold_manifest.csv}"
OUT_ROOT="${VIP_PX_OUT:-benchmark/running_results_image_pixel}"
SOURCE_ROOT="${VIP_PX_SOURCE_ROOT:-benchmark/running_results_image_src}"
DEVICE="${VIP_PX_DEVICE:-cuda}"
if [ -z "${VIP_PX_BASELINES:-}" ]; then
    echo "ERROR: VIP_PX_BASELINES is not set" >&2
    exit 1
fi

TASK="${SLURM_ARRAY_TASK_ID:-0}"
LINE="$(sed -n "$((TASK + 2))p" "$MANIFEST" | tr -d '\r')"
if [ -z "$LINE" ]; then
    echo "ERROR: no manifest row for array index ${TASK} in ${MANIFEST}" >&2
    exit 1
fi
IFS=, read -r DOMAIN DATA_DIR EXPERIMENT FOLD <<< "$LINE"
SOURCE_EXP="${SOURCE_ROOT}/${DOMAIN}/${EXPERIMENT}"

echo "SLURM_JOBID=${SLURM_JOB_ID:-?} node=${SLURM_JOB_NODELIST:-?} commit=$(cat PINNED_COMMIT 2>/dev/null || echo unpinned)"
echo "task=${TASK}: domain=${DOMAIN} fold=${FOLD} experiment=${EXPERIMENT}"
echo "source=${SOURCE_EXP} out_root=${OUT_ROOT} baselines=${VIP_PX_BASELINES} device=${DEVICE} extra_args=${VIP_PX_ARGS:-}"

if [ ! -d "$SOURCE_EXP/testing" ]; then
    echo "ERROR: no source experiment at ${SOURCE_EXP}" >&2
    exit 1
fi
if [ ! -d "$DATA_DIR/gt_trajectories" ]; then
    echo "ERROR: no gt_trajectories/ under ${DATA_DIR}" >&2
    exit 1
fi

module load anaconda
source activate "${VIP_ENV:-vip_venv11}"

GPU_LOG="logs/${SLURM_JOB_NAME:-vip-px}-gpu-${SLURM_JOB_ID:-0}.csv"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu \
        --format=csv,noheader -l 15 > "$GPU_LOG" &
    GPU_LOG_PID=$!
    trap 'kill "$GPU_LOG_PID" 2>/dev/null || true' EXIT
fi

RUN_RC=0
python -m benchmark.backfill_baseline \
    --experiment-dir "$SOURCE_EXP" \
    --data-dir "$DATA_DIR" \
    --baselines $VIP_PX_BASELINES \
    --cells "fold${FOLD}_numtrajs" \
    --out-root "$OUT_ROOT" \
    --device "$DEVICE" \
    --workers 1 ${VIP_PX_ARGS:-} || RUN_RC=$?

# The ROSAME arms leave a copy of the fold's trajectories in each cell.
find "${OUT_ROOT}/${DOMAIN}/${EXPERIMENT}/testing" -maxdepth 2 -type d -name temp_rosame_workspace -exec rm -rf {} + 2>/dev/null || true

if [ -s "$GPU_LOG" ]; then
    echo "peak_gpu_memory_used=$(awk -F', ' '{gsub(/ MiB/,"",$2); if ($2+0>m) m=$2+0} END{print m" MiB"}' "$GPU_LOG")"
fi
echo "task: domain=$DOMAIN fold=$FOLD rc=$RUN_RC"
exit "$RUN_RC"
