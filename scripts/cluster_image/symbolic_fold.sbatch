#!/bin/bash
### One task = one row (domain, fold) of a symbolic image manifest
### (scripts/cluster_image/make_image_manifests.py): that fold's whole
### training-size sweep and every arm of the row's run config, through
### benchmark/benchmark_runner.py --only-fold.
###
### Submit from the project root, one array per domain, after the setup pass
### (freeze_folds.sbatch) of that domain has ended:
###   sbatch --array=0-4 scripts/cluster_image/symbolic_fold.sbatch scripts/cluster_image/image_large_symbolic_manifest.csv
###   sbatch --time 0-08:00:00 --array=15-19 scripts/cluster_image/symbolic_fold.sbatch scripts/cluster_image/image_small_symbolic_manifest.csv
###
### Writes benchmark/running_results/<domain>/<experiment>/. Two tasks never
### share a cell folder as long as one manifest row is submitted once.
###   VIP_SYM_FROZEN_ROOT  the setup pass's tree (default benchmark/running_results_image_src).
###                        The task refuses to start before that tree holds this
###                        fold's cells, and takes the fold's test states from it.
###   VIP_SYM_NO_FREEZE=1  start without the setup pass (one fold job per domain
###                        at a time only: the pool's prepared files are shared)
###   VIP_ENV              conda env (default vip_venv11)
#SBATCH --partition main
#SBATCH --job-name vip-img-sym
#SBATCH --output logs/%x-%A_%a.out
#SBATCH --time 2-00:00:00
#SBATCH --cpus-per-task 6
#SBATCH --constraint cpu256
#SBATCH --gpus=0
#SBATCH --mem 24G

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

MANIFEST="${1:?usage: symbolic_fold.sbatch <symbolic manifest csv>}"
FROZEN_ROOT="${VIP_SYM_FROZEN_ROOT:-benchmark/running_results_image_src}"

TASK="${SLURM_ARRAY_TASK_ID:-0}"
LINE="$(sed -n "$((TASK + 2))p" "$MANIFEST" | tr -d '\r')"
if [ -z "$LINE" ]; then
    echo "ERROR: no manifest row for array index ${TASK} in ${MANIFEST}" >&2
    exit 1
fi
IFS=, read -r DOMAIN DATA_DIR EXPERIMENT FOLD CONFIG <<< "$LINE"
LIVE_EXP="benchmark/running_results/${DOMAIN}/${EXPERIMENT}"
FROZEN_EXP="${FROZEN_ROOT}/${DOMAIN}/${EXPERIMENT}"

echo "SLURM_JOBID=${SLURM_JOB_ID:-?} node=${SLURM_JOB_NODELIST:-?} commit=$(cat PINNED_COMMIT 2>/dev/null || echo unpinned)"
echo "task=${TASK}: domain=${DOMAIN} fold=${FOLD} experiment=${EXPERIMENT} config=${CONFIG}"
echo "cpu_model=$(awk -F': ' '/^model name/{print $2; exit}' /proc/cpuinfo)"

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: run config ${CONFIG} not found" >&2
    exit 1
fi
if [ ! -d "$DATA_DIR/training/image_trajectories" ] || [ ! -d "$DATA_DIR/gt_trajectories" ]; then
    echo "ERROR: ${DATA_DIR} lacks training/image_trajectories/ or gt_trajectories/" >&2
    exit 1
fi

if [ "${VIP_SYM_NO_FREEZE:-0}" != "1" ]; then
    if ! compgen -G "${FROZEN_EXP}/testing/fold${FOLD}_numtrajs*_gtrate*/fold_info.json" > /dev/null; then
        echo "ERROR: ${FROZEN_EXP} holds no frozen cell of fold ${FOLD}." >&2
        echo "       Run scripts/cluster_image/freeze_folds.sbatch ${CONFIG} ${DOMAIN} first." >&2
        exit 1
    fi
    mkdir -p "${LIVE_EXP}/testing"
    for SHARED in "${FROZEN_EXP}/testing/fold${FOLD}"_gtrate*_shared; do
        [ -d "$SHARED" ] || continue
        if [ ! -e "${LIVE_EXP}/testing/$(basename "$SHARED")" ]; then
            cp -r "$SHARED" "${LIVE_EXP}/testing/"
            echo "test states taken from ${SHARED}"
        fi
    done
fi

module load anaconda
source activate "${VIP_ENV:-vip_venv11}"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

RUN_RC=0
python -m benchmark.benchmark_runner \
    --config "$CONFIG" \
    --domains "$DOMAIN" \
    --only-fold "$FOLD" || RUN_RC=$?

# The ROSAME arms leave a copy of the fold's trajectories in each finished cell.
for INSTANCE in "${LIVE_EXP}/testing/fold${FOLD}"_numtrajs*_gtrate*; do
    [ -f "$INSTANCE/fold_result.json" ] || continue
    rm -rf "$INSTANCE/temp_rosame_workspace"
done

FINISHED="$( (compgen -G "${LIVE_EXP}/testing/fold${FOLD}_numtrajs*_gtrate*/fold_result.json" || true) | wc -l | tr -d ' ')"
CGROUP_REL="$(awk -F: '{print $3}' /proc/self/cgroup | head -1)"
PEAK="$(cat "/sys/fs/cgroup${CGROUP_REL%/step_*}/memory.peak" 2>/dev/null || echo 0)"
echo "task: domain=$DOMAIN fold=$FOLD finished_cells=$FINISHED cgroup_peak_mb=$(( PEAK / 1048576 )) rc=$RUN_RC"
exit "$RUN_RC"
