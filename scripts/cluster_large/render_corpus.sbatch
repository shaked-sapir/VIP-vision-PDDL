#!/bin/bash
# Render state_*.png frames into the symbolic large corpora: one array task =
# one (domain, chunk) row of render_manifest.csv, CPU only.
#
# Usage (from the project root on the cluster, corpora already rsynced under
# benchmark/data/<domain>/<corpus>/ with their gt_trajectories/):
#   mkdir -p logs
#   sbatch --array=0-29 scripts/cluster_large/render_corpus.sbatch [manifest.csv]
#
# Only the PDDLGym domains (blocksworld, hanoi, npuzzle) render here: they
# need pddlgym + matplotlib, which vip_venv11 has. depot and gripper render
# through planning-visualizer's Node canvas worker (node + the `canvas`
# package), which the cluster env does not have; run those on the laptop with
# scripts/cluster_large/render_local.sh.
#
# Resources, against the laptop measurement (three windows per domain,
# 2026-11-08): ~20 frames/s for blocksworld and hanoi, ~15 frames/s for
# npuzzle, one process. A 250-problem chunk is ~3,750 frames, under 5 min;
# --time is generous for queue-friendliness, not need. Memory: matplotlib
# figures are closed after every frame; 2 GB is plenty.
#
# The script is idempotent: a problem folder that already holds one frame per
# state is skipped, so a failed task is resubmitted as is.
#SBATCH --partition main
#SBATCH --job-name vip-render
#SBATCH --output logs/%x-%A_%a.out
#SBATCH --time 0-01:00:00
#SBATCH --cpus-per-task 2
#SBATCH --gpus=0
#SBATCH --mem 2G

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"   # submit from the project root

MANIFEST="${1:-scripts/cluster_large/render_manifest.csv}"

# Row for this task: +1 to skip the CSV header, +1 because sed is 1-based.
LINE="$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "$MANIFEST" | tr -d '\r')"
if [ -z "$LINE" ]; then
    echo "ERROR: no manifest row for array index ${SLURM_ARRAY_TASK_ID} in ${MANIFEST}" >&2
    exit 1
fi
IFS=, read -r DOMAIN CORPUS DOMAIN_FILE CHUNK NUM_CHUNKS <<< "$LINE"

echo "SLURM_JOBID=${SLURM_JOB_ID:-?} node=${SLURM_JOB_NODELIST:-?}"
echo "task=${SLURM_ARRAY_TASK_ID}: domain=${DOMAIN} corpus=${CORPUS} chunk=${CHUNK}/${NUM_CHUNKS}"

if [ ! -d "$CORPUS/gt_trajectories" ]; then
    echo "ERROR: no gt_trajectories/ under ${CORPUS}. rsync the corpus first." >&2
    exit 1
fi

module load anaconda
source activate "${VIP_ENV:-vip_venv11}"

export MPLBACKEND=Agg
export OMP_NUM_THREADS=1

python benchmark/render_corpus.py \
    --domain "$DOMAIN" \
    --corpus "$CORPUS" \
    --domain-file "$DOMAIN_FILE" \
    --chunk "$CHUNK" \
    --num-chunks "$NUM_CHUNKS"
