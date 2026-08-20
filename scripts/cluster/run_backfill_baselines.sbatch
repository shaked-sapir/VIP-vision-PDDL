#!/bin/bash
# ROSAME-family backfill SLURM array template: each task = one
# (experiment_dir, num_trajs) manifest row, i.e. the 5 folds of that cell group,
# which backfill_baseline.py runs in parallel in-process (--workers 5) - hence
# 12 logical CPUs per task, same shape as run_cell.sbatch.
#
# Fills in the ROSAME, ROSAME_MILP and ROSAME_MILP_TAG rows next to the CDPS row
# already present in each cell's fold_result.json. Learning runs off the frozen
# original_observations/, so no data is regenerated and the baselines see exactly
# the masked/noised observations CDPS saw.
#
# Usage (from the project root on the cluster):
#   python scripts/cluster/make_backfill_cdps-anchored_manifest.py \
#       --run-name <run> --out scripts/cluster/backfill_baselines_manifest.csv
#   mkdir -p logs
#   sbatch --array=0-<N-1>%60 scripts/cluster/run_backfill_baselines.sbatch \
#          [scripts/cluster/backfill_baselines_manifest.csv]
#
# The manifest generator is algorithm-agnostic (domain, experiment_dir,
# num_trajs), so it is shared with the cdps-anchored backfill via --out.
#
# Conda env: defaults to "vip_venv11"; override with  export VIP_ENV=my_env  before sbatch.
#
# --learn-timeout 600 matches the learning_timeout_seconds of the run being
# backfilled (see evaluation_results/run_params.json), so every algorithm in a
# given fold_result.json got the same budget. --planning-timeout 60 likewise
# mirrors the original evaluation.
#
# Hardware pinning (see the header of submit.sh for the measurements):
#   --constraint cpu256  pins to the AMD EPYC 7763 nodes. Not optional here: the
#     MILP variants hand CP-SAT a wall-clock time limit, so a row learned on a
#     slower node would not be comparable to the CDPS row beside it. Never use
#     `cpu128` -- it is the slow EPYC.
#   --cpus-per-task 12   SLURM counts *logical* CPUs and every node here has
#     ThreadsPerCore=2, so 12 => 6 physical cores for the 5 worker processes plus
#     the parent.
#   --time 0-03:00:00    worst case is 5 parallel cells x 3 baselines x (600s
#     learn cap + 2 test problems at a 60s planning cap). Historical means are far
#     lower (ROSAME 3s, MILP 20s, MILP_TAG 27s), but runtime does not scale
#     monotonically with num_trajs, so keep the headroom.
#SBATCH --partition main
#SBATCH --job-name vip-bf-rosame
#SBATCH --output logs/%x-%A_%a.out
#SBATCH --time 0-03:00:00
#SBATCH --cpus-per-task 12
#SBATCH --constraint cpu256
#SBATCH --gpus=0
#SBATCH --mem 24G

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"   # submit from the project root

MANIFEST="${1:-scripts/cluster/backfill_baselines_manifest.csv}"

# Row for this task: +1 to skip the CSV header, +1 because sed is 1-based.
# `tr -d '\r'` guards against a CRLF manifest leaking \r into the last field.
LINE="$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "$MANIFEST" | tr -d '\r')"
if [ -z "$LINE" ]; then
    echo "ERROR: no manifest row for array index ${SLURM_ARRAY_TASK_ID} in ${MANIFEST}" >&2
    exit 1
fi
IFS=, read -r DOMAIN EXPERIMENT_DIR NUM_TRAJS <<< "$LINE"

if [ ! -d "$EXPERIMENT_DIR/testing" ]; then
    echo "ERROR: no testing/ under '${EXPERIMENT_DIR}'" >&2
    exit 1
fi

echo "SLURM_JOBID=${SLURM_JOB_ID:-?} node=${SLURM_JOB_NODELIST:-?}"
echo "task=${SLURM_ARRAY_TASK_ID}: domain=${DOMAIN} numtrajs=${NUM_TRAJS}"
echo "experiment_dir=${EXPERIMENT_DIR}"

# A wall-clock solver budget makes results hardware-dependent, so the CPU model
# belongs in the log next to every result.
echo "cpu_model=$(awk -F': ' '/^model name/{print $2; exit}' /proc/cpuinfo)"
echo "cpus_allocated=${SLURM_CPUS_PER_TASK:-?} (logical; ThreadsPerCore=2 on this cluster)"

module load anaconda
source activate "${VIP_ENV:-vip_venv11}"

# One BLAS/OpenMP thread per worker process. backfill_common.worker_init sets
# these too, but exporting here also covers the parent and any child that does
# not go through the pool (torch in particular reads them at import time).
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# --cells matches the 5 fold dirs of this num_trajs value only
# (cells are named fold{f}_numtrajs{n}_gtrate{g}).
python -m benchmark.backfill_baseline \
    --experiment-dir "$EXPERIMENT_DIR" \
    --domain "$DOMAIN" \
    --baselines rosame_24 rosame_milp_24 rosame_milp_24_tag \
    --learn-timeout 600 \
    --planning-timeout 60 \
    --cells "_numtrajs${NUM_TRAJS}_gtrate" \
    --workers 5
