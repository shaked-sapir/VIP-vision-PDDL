#!/bin/bash
# One task = one (domain, mask, noise, fold) of an existing large-corpora run:
# the unanchored MILP loop on that fold's frozen observations with states 0 and 1
# degraded (benchmark/backfill_nogt_s0.py), over the fold's whole L sweep.
#
#   sbatch --array=0-449   scripts/cluster_large/nogt_s0_fold.sbatch
#   sbatch --array=450-749 scripts/cluster_large/nogt_s0_fold.sbatch
#   (fold_manifest.csv rows are grouped by domain, 150 each)
#
# Extra driver flags go in $VIP_NOGT_ARGS, e.g. to redo only the sizes whose row
# recorded an error:
#   VIP_NOGT_ARGS=--retry-errors sbatch --array=100,105 scripts/cluster_large/nogt_s0_fold.sbatch
#
# Reads $VIP_NOGT_SOURCE_ROOT/<domain>/<run_name>__mask=..__noise=../ (default
# root benchmark/running_results) and writes only under $VIP_NOGT_OUT (default
# benchmark/running_results_nogt_s0). The source run may be packed: frozen
# observations are read from the tarball.
#
# $VIP_NOGT_ALGORITHMS lists the arms to run in order (default pisam_milp_loop),
# and $VIP_LARGE_CONFIG the YAML their pisam_milp block is read from.
# $VIP_NOGT_LOOP_ARGS holds driver flags for the pisam_milp_loop arm only. The
# small-corpora run:
#   VIP_NOGT_SOURCE_ROOT=benchmark/running_results_small_src \
#   VIP_LARGE_CONFIG=benchmark/run_config.yaml \
#   VIP_NOGT_ALGORITHMS="pisam_milp_loop pisam_milp_single_round" \
#   VIP_NOGT_LOOP_ARGS="--solve-time-limit 60" \
#   sbatch --array=0-749 scripts/cluster_large/nogt_s0_fold.sbatch \
#       scripts/cluster_large/nogt_small_fold_manifest.csv
#
# Resources and the one-worker, whole-L-sweep-per-job shape match
# sweep_fold.sbatch, so the wall-clock learning budget is comparable with the
# rows of the source run.
#SBATCH --partition main
#SBATCH --job-name vip-nogt
#SBATCH --output logs/%x-%A_%a.out
#SBATCH --time 3-00:00:00
#SBATCH --cpus-per-task 6
#SBATCH --constraint cpu256
#SBATCH --gpus=0
#SBATCH --mem 24G

set -euo pipefail
shopt -s failglob

cd "${SLURM_SUBMIT_DIR}"

MANIFEST="${1:-scripts/cluster_large/fold_manifest.csv}"
CONFIG="${VIP_LARGE_CONFIG:-benchmark/run_config_large.yaml}"
OUT_ROOT="${VIP_NOGT_OUT:-benchmark/running_results_nogt_s0}"
SOURCE_ROOT="${VIP_NOGT_SOURCE_ROOT:-benchmark/running_results}"
ALGORITHMS="${VIP_NOGT_ALGORITHMS:-pisam_milp_loop}"

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
echo "source=${SOURCE_EXP} out_root=${OUT_ROOT} config=${CONFIG} algorithms=${ALGORITHMS} extra_args=${VIP_NOGT_ARGS:-} loop_args=${VIP_NOGT_LOOP_ARGS:-}"
echo "cpu_model=$(awk -F': ' '/^model name/{print $2; exit}' /proc/cpuinfo)"

if [ ! -d "$SOURCE_EXP/testing" ]; then
    echo "ERROR: no source experiment at ${SOURCE_EXP}" >&2
    exit 1
fi
if [ ! -d "$DATA_DIR/gt_trajectories" ]; then
    echo "ERROR: no gt_trajectories/ under ${DATA_DIR}" >&2
    exit 1
fi

CGROUP_REL="$(awk -F: '{print $3}' /proc/self/cgroup | head -1)"
CGROUP="/sys/fs/cgroup${CGROUP_REL%/step_*}"
MEMLOG="logs/mem-nogt-${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}.tsv"
printf 'iso_time\telapsed_s\tcgroup_current_mb\tcgroup_peak_mb\n' > "$MEMLOG"
MON_START=$(date +%s)
( while true; do
    cur=$(cat "$CGROUP/memory.current" 2>/dev/null || echo 0)
    pk=$(cat "$CGROUP/memory.peak" 2>/dev/null || echo 0)
    printf '%s\t%s\t%s\t%s\n' "$(date -Iseconds)" "$(( $(date +%s) - MON_START ))" \
        "$(( cur / 1048576 ))" "$(( pk / 1048576 ))" >> "$MEMLOG" || true
    sleep 60
  done ) &
MONITOR_PID=$!
trap 'kill $MONITOR_PID 2>/dev/null || true' EXIT

module load anaconda
source activate "${VIP_ENV:-vip_venv11}"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

# The exit code is held so a failed size still gets the finished ones packed.
RUN_RC=0
for ALGORITHM in $ALGORITHMS; do
    ARM_ARGS=""
    if [ "$ALGORITHM" = "pisam_milp_loop" ]; then
        ARM_ARGS="${VIP_NOGT_LOOP_ARGS:-}"
    fi
    python -m benchmark.backfill_nogt_s0 \
        --experiment-dir "$SOURCE_EXP" \
        --data-dir "$DATA_DIR" \
        --algorithm "$ALGORITHM" \
        --milp-config "$CONFIG" \
        --out-root "$OUT_ROOT" \
        --cells "fold${FOLD}_numtrajs" \
        --workers 1 ${VIP_NOGT_ARGS:-} $ARM_ARGS || RUN_RC=$?
done

source scripts/cluster_large/pack_fold.sh
pack_nogt_instances "${OUT_ROOT}/${DOMAIN}/${EXPERIMENT}" "$FOLD"

kill $MONITOR_PID 2>/dev/null || true
echo "==================== MEMORY SUMMARY ===================="
echo "cell: domain=$DOMAIN mask=$P_MASK noise=$P_NOISE fold=$FOLD"
echo "cgroup_peak_mb=$(( $(cat "$CGROUP/memory.peak" 2>/dev/null || echo 0) / 1048576 )) of ${SLURM_MEM_PER_NODE:-?}MB"
echo "========================================================"
exit "$RUN_RC"
