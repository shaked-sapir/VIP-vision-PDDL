#!/bin/bash
### One CPU task running probe_scored_example.py; pass VIP_VENDOR, VIP_VLM_JOB
### (serving job id, read for the node) and VIP_TARGETS ("domain:problem ...").
#SBATCH --partition main
#SBATCH --job-name vip-probe
#SBATCH --output logs/%x-%j.out
#SBATCH --time 0-03:00:00
#SBATCH --gpus=0
#SBATCH --mem 8G
#SBATCH --cpus-per-task 4
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
module load anaconda
source activate "${VIP_ENV:-vip_venv11}"
if [ -n "${VIP_VLM_JOB:-}" ]; then
  export VIP_LLM_BASE_URL="http://$(cat "logs/vlm_endpoint-${VIP_VLM_JOB}.txt")/v1"
fi
python -m scripts.cluster_image.probe_scored_example --vendor "${VIP_VENDOR:?}" ${VIP_TARGETS:?}
