#!/bin/bash
# Repair template: one task = one (cell, fold) at ONE L, for cells that timed out
# with their smaller L tiers already complete.
#
#   sbatch --array=0-14 scripts/cluster_large/repair_fold.sbatch repair_manifest.csv
#
# Why this splits by fold AND by L, when the sweep splits by neither:
#
#   The sweep keeps all five L values inside one job on purpose -- the learning
#   budget is wall-clock, so a cell's L axis has to run on one node and one CPU
#   model to be comparable. That argument does not apply here: these cells' L=10
#   to L=500 tiers are already finished and their timings already recorded, so
#   this only produces the missing L=2000 rows. Splitting them out is what turns
#   a 24 h job that timed out twice into a ~6 h job per fold.
#
# Measured on the completed noise=0.2 cells: L=2000 costs 4.9-6.1 h of learning
# per fold plus ~0.5 h of evaluation. --time 2 days is ~7x that, because the
# gt=none arm's evaluation has a long tail (one took 8.4 h).
#SBATCH --partition main
#SBATCH --job-name vip-repair
#SBATCH --output logs/%x-%A_%a.out
#SBATCH --time 2-00:00:00
#SBATCH --cpus-per-task 6
#SBATCH --constraint cpu256
#SBATCH --gpus=0
#SBATCH --mem 24G

set -euo pipefail
shopt -s failglob

cd "${SLURM_SUBMIT_DIR}"

MANIFEST="${1:-scripts/cluster_large/repair_manifest.csv}"
CONFIG="${VIP_LARGE_CONFIG:-benchmark/run_config_large.yaml}"

LINE="$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "$MANIFEST" | tr -d '\r')"
if [ -z "$LINE" ]; then
    echo "ERROR: no manifest row for array index ${SLURM_ARRAY_TASK_ID}" >&2
    exit 1
fi
IFS=, read -r DOMAIN P_MASK P_NOISE FOLD NUM_TRAJS <<< "$LINE"

echo "SLURM_JOBID=${SLURM_JOB_ID:-?} node=${SLURM_JOB_NODELIST:-?}"
echo "repair: domain=${DOMAIN} mask=${P_MASK} noise=${P_NOISE} fold=${FOLD} L=${NUM_TRAJS}"
echo "cpu_model=$(awk -F': ' '/^model name/{print $2; exit}' /proc/cpuinfo)"

# Same monitoring as the sweep: cgroup total is what the OOM killer acts on,
# per-worker RSS is what shows one worker ballooning while others stay flat.
CGROUP_REL="$(awk -F: '{print $3}' /proc/self/cgroup | head -1)"
CGROUP="/sys/fs/cgroup${CGROUP_REL%/step_*}"
MEMLOG="logs/mem-repair-${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}.tsv"
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

# A scratch config pinned to this task's single L; the fold comes from the CLI.
# n_folds stays 5, so cv_split produces the identical split -- --only-fold
# selects which fold to RUN, it does not change how the data is partitioned.
PROBE_CFG="$(mktemp /tmp/repair_cfg_XXXX.yaml)"
python - "$CONFIG" "$PROBE_CFG" "$NUM_TRAJS" <<'PY'
import sys, yaml
src, dst, L = sys.argv[1], sys.argv[2], int(sys.argv[3])
c = yaml.safe_load(open(src))
c["shared"]["num_trajectories"] = [L]
c["shared"]["resume"] = True
yaml.safe_dump(c, open(dst, "w"), sort_keys=False)
PY

python -m benchmark.benchmark_runner \
    --config "$PROBE_CFG" \
    --domains "$DOMAIN" \
    --only-mask "$P_MASK" \
    --only-noise "$P_NOISE" \
    --only-fold "$FOLD"

kill $MONITOR_PID 2>/dev/null || true
echo "peak_memory_mb=$(( $(cat "$CGROUP/memory.peak" 2>/dev/null || echo 0) / 1048576 ))"
rm -f "$PROBE_CFG"
