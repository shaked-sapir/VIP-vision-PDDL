#!/bin/bash
# One task = one manifest row of an existing run: the baselines named in
# $VIP_BL_BASELINES on the row's frozen observations, unchanged
# (benchmark/backfill_baseline.py --out-root), over the whole training-size sweep.
#
# Large corpora, one task per (domain, mask, noise, fold):
#   VIP_BL_BASELINES="pisam" VIP_BL_ARGS="--learn-timeout 3600" \
#   sbatch --array=0-749 scripts/cluster_large/baseline_out_root_fold.sbatch
#
# Small corpora, one task per (domain, mask, noise), all folds in the task (a
# manifest row whose fold field is "all"):
#   VIP_BL_BASELINES="pisam" VIP_BL_SOURCE_ROOT=benchmark/running_results_small_src \
#   VIP_BL_WORKERS=3 VIP_BL_ARGS="--learn-timeout 300" \
#   sbatch --array=0-149 scripts/cluster_large/baseline_out_root_fold.sbatch \
#       scripts/cluster_large/anchored_small_cell_manifest.csv
#
# Reads $VIP_BL_SOURCE_ROOT/<domain>/<run_name>__mask=..__noise=../ (default root
# benchmark/running_results; the source may be packed) and writes only under
# $VIP_BL_OUT (default benchmark/running_results_baselines_rerun).
# $VIP_BL_WORKERS is the cells run in parallel (default 1) and $VIP_BL_ARGS any
# further backfill_baseline flags (training knobs, timeouts, --force).
#SBATCH --partition main
#SBATCH --job-name vip-bl
#SBATCH --output logs/%x-%A_%a.out
#SBATCH --time 3-00:00:00
#SBATCH --cpus-per-task 3
#SBATCH --constraint cpu256
#SBATCH --gpus=0
#SBATCH --mem 12G

set -euo pipefail
shopt -s failglob

cd "${SLURM_SUBMIT_DIR}"

MANIFEST="${1:-scripts/cluster_large/fold_manifest.csv}"
OUT_ROOT="${VIP_BL_OUT:-benchmark/running_results_baselines_rerun}"
SOURCE_ROOT="${VIP_BL_SOURCE_ROOT:-benchmark/running_results}"
WORKERS="${VIP_BL_WORKERS:-1}"
if [ -z "${VIP_BL_BASELINES:-}" ]; then
    echo "ERROR: VIP_BL_BASELINES is not set" >&2
    exit 1
fi

LINE="$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "$MANIFEST" | tr -d '\r')"
if [ -z "$LINE" ]; then
    echo "ERROR: no manifest row for array index ${SLURM_ARRAY_TASK_ID} in ${MANIFEST}" >&2
    exit 1
fi
IFS=, read -r DOMAIN DATA_DIR P_MASK P_NOISE FOLD RUN_NAME <<< "$LINE"
EXPERIMENT="${RUN_NAME}__mask=${P_MASK}__noise=${P_NOISE}"
SOURCE_EXP="${SOURCE_ROOT}/${DOMAIN}/${EXPERIMENT}"

echo "SLURM_JOBID=${SLURM_JOB_ID:-?} node=${SLURM_JOB_NODELIST:-?}"
echo "task=${SLURM_ARRAY_TASK_ID}: domain=${DOMAIN} mask=${P_MASK} noise=${P_NOISE} fold=${FOLD}"
echo "source=${SOURCE_EXP} out_root=${OUT_ROOT} baselines=${VIP_BL_BASELINES} workers=${WORKERS} extra_args=${VIP_BL_ARGS:-}"
echo "cpu_model=$(awk -F': ' '/^model name/{print $2; exit}' /proc/cpuinfo)"

if [ ! -d "$SOURCE_EXP/testing" ]; then
    echo "ERROR: no source experiment at ${SOURCE_EXP}" >&2
    exit 1
fi
if [ ! -d "$DATA_DIR/gt_trajectories" ]; then
    echo "ERROR: no gt_trajectories/ under ${DATA_DIR}" >&2
    exit 1
fi

if [ "$FOLD" = "all" ]; then
    CELL_ARGS=()
else
    CELL_ARGS=(--cells "fold${FOLD}_numtrajs")
fi

module load anaconda
source activate "${VIP_ENV:-vip_venv11}"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

RUN_RC=0
python -m benchmark.backfill_baseline \
    --experiment-dir "$SOURCE_EXP" \
    --data-dir "$DATA_DIR" \
    --baselines $VIP_BL_BASELINES \
    --out-root "$OUT_ROOT" \
    --workers "$WORKERS" ${CELL_ARGS[@]+"${CELL_ARGS[@]}"} ${VIP_BL_ARGS:-} || RUN_RC=$?

# The ROSAME arms leave a copy of the fold's trajectories in each cell.
find "${OUT_ROOT}/${DOMAIN}/${EXPERIMENT}/testing" -maxdepth 2 -type d -name temp_rosame_workspace -exec rm -rf {} + 2>/dev/null || true

echo "cell: domain=$DOMAIN mask=$P_MASK noise=$P_NOISE fold=$FOLD rc=$RUN_RC"
exit "$RUN_RC"
