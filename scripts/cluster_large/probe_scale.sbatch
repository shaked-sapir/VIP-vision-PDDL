#!/bin/bash
# Measure peak memory and wall-clock for ONE domain at ONE L -- all five folds,
# as the real sweep runs them -- so the sweep's --mem and --time can be set per
# domain instead of guessed.
#
#   sbatch --export=ALL,PROBE_DOMAIN=hanoi,PROBE_L=500 scripts/cluster_large/probe_scale.sbatch
#
# Deliberately over-allocated (200G): the point is to observe the peak, not to
# cap it. A probe that OOMs measures nothing.
#
# Reads memory.peak from the job's own cgroup (v2) every 15s and prints the
# high-water mark at the end, alongside the per-L directory progress.
#SBATCH --partition main
#SBATCH --job-name vip-probe
#SBATCH --output logs/%x-%j.out
#SBATCH --time 0-08:00:00
#SBATCH --cpus-per-task 12
#SBATCH --constraint cpu256
#SBATCH --gpus=0
#SBATCH --mem 200G

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

DOMAIN="${PROBE_DOMAIN:?set PROBE_DOMAIN}"
L="${PROBE_L:-500}"
MASK="${PROBE_MASK:-0.1}"
NOISE="${PROBE_NOISE:-0.1}"
CONFIG="${VIP_LARGE_CONFIG:-benchmark/run_config_large.yaml}"

echo "probe: domain=$DOMAIN L=$L mask=$MASK noise=$NOISE"
echo "node=$(hostname)  mem_limit=${SLURM_MEM_PER_NODE:-?}MB  cpus=${SLURM_CPUS_PER_TASK:-?}"
echo "cpu_model=$(awk -F': ' '/^model name/{print $2; exit}' /proc/cpuinfo)"

# Sample the job cgroup's high-water mark while the run proceeds.
CGROUP_REL="$(awk -F: '{print $3}' /proc/self/cgroup | head -1)"
CGROUP="/sys/fs/cgroup${CGROUP_REL%/step_*}"
( while true; do
    p=$(cat "$CGROUP/memory.peak" 2>/dev/null || echo 0)
    c=$(cat "$CGROUP/memory.current" 2>/dev/null || echo 0)
    # Per-worker RSS, this job's cgroup only: the total alone hides one worker
    # ballooning while its siblings stay flat, which is how the uncapped
    # rosame_milp_* arm was found. PIDs from the whole subtree -- the fold
    # workers live at step_batch/user/task_0, four levels down.
    rss=$(find "$CGROUP" -name cgroup.procs -exec cat {} + 2>/dev/null \
          | sort -un | while read -r pid; do
              [ -r "/proc/$pid/statm" ] || continue
              awk '{printf "%d\n", $2 * 4 / 1024}' "/proc/$pid/statm" 2>/dev/null
            done | sort -rn | head -8 | paste -sd, -)
    echo "  [mem] peak=$(( p / 1048576 ))MB current=$(( c / 1048576 ))MB workers=[${rss}] t=$(date +%H:%M:%S)"
    sleep 15
  done ) &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null || true' EXIT

module load anaconda
source activate "${VIP_ENV:-vip_venv11}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

# A scratch config: one fold, one L, one cell — everything else inherited.
PROBE_CFG="$(mktemp /tmp/probe_cfg_XXXX.yaml)"
python - "$CONFIG" "$PROBE_CFG" "$L" "$DOMAIN" <<'PY'
import sys, yaml
src, dst, L, domain = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
c = yaml.safe_load(open(src))
c["run_name"] = f"probe__{domain}__L{L}"
# ALL 5 folds: max_workers = len(fold_list), so folds=[0] would run one worker
# and measure a fifth of the real peak. The sweep runs five concurrently, and
# it is their combined footprint that the cgroup limit acts on.
c["shared"]["num_trajectories"] = [L]
c["shared"]["resume"] = False
yaml.safe_dump(c, open(dst, "w"), sort_keys=False)
PY

START=$(date +%s)
python -m benchmark.benchmark_runner --config "$PROBE_CFG" \
    --domains "$DOMAIN" --only-mask "$MASK" --only-noise "$NOISE" || echo "RUN FAILED (exit $?)"
ELAPSED=$(( $(date +%s) - START ))

kill $SAMPLER 2>/dev/null || true
PEAK=$(cat "$CGROUP/memory.peak" 2>/dev/null || echo 0)
echo "==================== PROBE RESULT ===================="
echo "domain=$DOMAIN L=$L folds=5 (as the sweep runs them)"
echo "peak_memory_MB=$(( PEAK / 1048576 ))"
echo "peak_memory_GB=$(awk -v v=$PEAK 'BEGIN{printf "%.2f", v/1073741824}')"
echo "wall_seconds=$ELAPSED  ($(( ELAPSED / 60 )) min)"
echo "======================================================"
rm -f "$PROBE_CFG"
