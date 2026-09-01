#!/bin/bash
# Fold-level sweep template: one task = one (domain, mask, noise, fold), running
# that fold's WHOLE L sweep and every arm.
#
#   python scripts/cluster_large/make_manifest.py --per-fold
#   sbatch --array=0-224%45 scripts/cluster_large/sweep_fold.sbatch fold_manifest.csv
#
# Why split by fold but NOT by L:
#
#   The learning budget is wall-clock, so a cell's L axis has to run on one node
#   and one CPU model to stay comparable -- splitting L across jobs would scatter
#   L=10 and L=2000 onto different hardware. Folds carry no such tie: cv_split is
#   deterministic per fold and folds never interact, so one fold per job keeps
#   every comparison intact.
#
# What it buys, from the v2 sweep: two cells hit the 24 h wall and lost 50 folds
# of work between them, because one slow arm blocked the whole cell (a single
# gt=none evaluation took 8.4 h). Per fold, that arm blocks one fold. Each job is
# also ~1/5 the work, so the wall is far less likely to be reached at all.
#
# S_test is generated per (fold, gt_rate) inside the cell's own testing/ dir, so
# the five jobs of one fold across different mask/noise points never share it --
# each computes its own, as the per-cell sweep already did.
#
# Resources, against the v2 measurements: a cell peaked at 7.2-14.6 GB with five
# fold workers; one fold is a fifth of that, so 24G is generous. max_workers is
# len(fold_list), so a single-fold job spawns one worker -- 6 logical CPUs give
# it 3 physical cores, with headroom over the 1 cpu-second/second it uses.
#SBATCH --partition main
#SBATCH --job-name vip-fold
#SBATCH --output logs/%x-%A_%a.out
#SBATCH --time 1-00:00:00
#SBATCH --cpus-per-task 6
#SBATCH --constraint cpu256
#SBATCH --gpus=0
#SBATCH --mem 24G

set -euo pipefail
shopt -s failglob

cd "${SLURM_SUBMIT_DIR}"

MANIFEST="${1:-scripts/cluster_large/fold_manifest.csv}"
CONFIG="${VIP_LARGE_CONFIG:-benchmark/run_config_large.yaml}"

LINE="$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "$MANIFEST" | tr -d '\r')"
if [ -z "$LINE" ]; then
    echo "ERROR: no manifest row for array index ${SLURM_ARRAY_TASK_ID} in ${MANIFEST}" >&2
    exit 1
fi
IFS=, read -r DOMAIN DATA_DIR P_MASK P_NOISE FOLD RUN_NAME <<< "$LINE"

echo "SLURM_JOBID=${SLURM_JOB_ID:-?} node=${SLURM_JOB_NODELIST:-?}"
echo "task=${SLURM_ARRAY_TASK_ID}: domain=${DOMAIN} mask=${P_MASK} noise=${P_NOISE} fold=${FOLD}"
echo "run_name=${RUN_NAME} config=${CONFIG}"
echo "cpu_model=$(awk -F': ' '/^model name/{print $2; exit}' /proc/cpuinfo)"

if [ ! -d "$DATA_DIR/gt_trajectories" ]; then
    echo "ERROR: no gt_trajectories/ under ${DATA_DIR}. Generate the corpus locally first." >&2
    exit 1
fi

# Same two-view monitoring as the per-cell sweep: the cgroup total is what the
# OOM killer acts on, the per-worker RSS is what shows one worker ballooning.
CGROUP_REL="$(awk -F: '{print $3}' /proc/self/cgroup | head -1)"
CGROUP="/sys/fs/cgroup${CGROUP_REL%/step_*}"
MEMLOG="logs/mem-fold-${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}.tsv"
printf 'iso_time\telapsed_s\tcgroup_current_mb\tcgroup_peak_mb\tworker_rss_mb_csv\n' > "$MEMLOG"
MON_START=$(date +%s)
( while true; do
    cur=$(cat "$CGROUP/memory.current" 2>/dev/null || echo 0)
    pk=$(cat "$CGROUP/memory.peak" 2>/dev/null || echo 0)
    rss=$(find "$CGROUP" -name cgroup.procs -exec cat {} + 2>/dev/null \
          | sort -un | while read -r pid; do
              [ -r "/proc/$pid/statm" ] || continue
              awk '{printf "%d\n", $2 * 4 / 1024}' "/proc/$pid/statm" 2>/dev/null
            done | sort -rn | head -8 | paste -sd, -) || rss=""
    printf '%s\t%s\t%s\t%s\t%s\n' "$(date -Iseconds)" "$(( $(date +%s) - MON_START ))" \
        "$(( cur / 1048576 ))" "$(( pk / 1048576 ))" "${rss:-}" >> "$MEMLOG" || true
    sleep 20
  done ) &
MONITOR_PID=$!
trap 'kill $MONITOR_PID 2>/dev/null || true' EXIT

module load anaconda
source activate "${VIP_ENV:-vip_venv11}"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

# n_folds stays whatever the config says, so cv_split produces the identical
# partition; --only-fold selects which fold to RUN, not how the data is split.
python -m benchmark.benchmark_runner \
    --config "$CONFIG" \
    --domains "$DOMAIN" \
    --only-mask "$P_MASK" \
    --only-noise "$P_NOISE" \
    --only-fold "$FOLD"

kill $MONITOR_PID 2>/dev/null || true
echo "==================== MEMORY SUMMARY ===================="
echo "cell: domain=$DOMAIN mask=$P_MASK noise=$P_NOISE fold=$FOLD"
echo "cgroup_peak_mb=$(( $(cat "$CGROUP/memory.peak" 2>/dev/null || echo 0) / 1048576 )) of ${SLURM_MEM_PER_NODE:-?}MB"
echo "========================================================"
