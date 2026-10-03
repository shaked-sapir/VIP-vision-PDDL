#!/bin/bash
# One task = one manifest row of an existing run: the anchored MILP loop
# (gt_anchoring init_only) with a per-solve time limit on the row's frozen
# observations, unchanged (benchmark/backfill_anchored_solve_cap.py), over the
# whole training-size sweep.
#
# Large corpora, one task per (domain, mask, noise, fold):
#   sbatch --array=0-449   scripts/cluster_large/anchored_cap_fold.sbatch
#   sbatch --array=450-749 scripts/cluster_large/anchored_cap_fold.sbatch
#
# Small corpora, one task per (domain, mask, noise), all folds in the task (a
# manifest row whose fold field is "all"):
#   VIP_ANCH_SOURCE_ROOT=benchmark/running_results_small_src \
#   VIP_LARGE_CONFIG=benchmark/run_config.yaml \
#   VIP_ANCH_ARGS="--learn-timeout 300" \
#   sbatch --array=0-149 scripts/cluster_large/anchored_cap_fold.sbatch \
#       scripts/cluster_large/anchored_small_cell_manifest.csv
#
# Reads $VIP_ANCH_SOURCE_ROOT/<domain>/<run_name>__mask=..__noise=../ (default
# root benchmark/running_results; the source may be packed) and writes only
# under $VIP_ANCH_OUT (default benchmark/running_results_anchored_solve60).
# $VIP_SOLVE_LIMIT is the per-solve cap in seconds (default 60), $VIP_ANCH_WORKERS
# the cells run in parallel (default 1) and $VIP_ANCH_ARGS extra driver flags.
#SBATCH --partition main
#SBATCH --job-name vip-anch-s60
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
CONFIG="${VIP_LARGE_CONFIG:-benchmark/run_config_large.yaml}"
OUT_ROOT="${VIP_ANCH_OUT:-benchmark/running_results_anchored_solve60}"
SOURCE_ROOT="${VIP_ANCH_SOURCE_ROOT:-benchmark/running_results}"
SOLVE_LIMIT="${VIP_SOLVE_LIMIT:-60}"
WORKERS="${VIP_ANCH_WORKERS:-1}"

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
echo "source=${SOURCE_EXP} out_root=${OUT_ROOT} config=${CONFIG} arm=pisam_milp_loop(init_only) solve_limit=${SOLVE_LIMIT} workers=${WORKERS} extra_args=${VIP_ANCH_ARGS:-}"
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
    FOLDS="$(ls "$SOURCE_EXP/testing" | sed -n 's/^fold\([0-9]*\)_numtrajs.*/\1/p' | sort -un)"
else
    CELL_ARGS=(--cells "fold${FOLD}_numtrajs")
    FOLDS="$FOLD"
fi

module load anaconda
source activate "${VIP_ENV:-vip_venv11}"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

# The exit code is held so a failed size still gets the finished ones packed.
RUN_RC=0
python -m benchmark.backfill_anchored_solve_cap \
    --experiment-dir "$SOURCE_EXP" \
    --data-dir "$DATA_DIR" \
    --milp-config "$CONFIG" \
    --out-root "$OUT_ROOT" \
    --solve-time-limit "$SOLVE_LIMIT" \
    --workers "$WORKERS" ${CELL_ARGS[@]+"${CELL_ARGS[@]}"} ${VIP_ANCH_ARGS:-} || RUN_RC=$?

source scripts/cluster_large/pack_fold.sh
for F in $FOLDS; do
    pack_nogt_instances "${OUT_ROOT}/${DOMAIN}/${EXPERIMENT}" "$F"
done

echo "cell: domain=$DOMAIN mask=$P_MASK noise=$P_NOISE fold=$FOLD rc=$RUN_RC"
exit "$RUN_RC"
