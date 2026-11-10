#!/bin/bash
### The VLM pre-check (benchmark/vlm_precheck.py) as one array task per domain,
### run on CPU nodes: the model is remote (OpenAI, or a vLLM node from
### serve_vlm.sbatch). Submit from the image checkout root:
###   VIP_VENDOR=openai sbatch scripts/cluster_image/precheck.sbatch
###   VIP_VENDOR=vllm VIP_LLM_BASE_URL=http://<node>:8000/v1 sbatch scripts/cluster_image/precheck.sbatch
###   VIP_VENDOR=vllm VIP_VLM_JOB=<id> sbatch --dependency=after:<id> scripts/cluster_image/precheck.sbatch
### Knobs: VIP_LABEL (output dir prefix, default precheck), VIP_OWN_FRAME0=1 (ignore
### the shared example_dir), VIP_PROBLEMS (default 5), VIP_WORKERS (VLM calls in flight, default 1),
### VIP_DOMAINS (space-separated, default the five image domains; the array index picks one).
#SBATCH --partition main
#SBATCH --job-name vip-precheck
#SBATCH --output logs/%x-%A_%a.out
#SBATCH --time 0-04:00:00
#SBATCH --gpus=0
#SBATCH --mem 8G
#SBATCH --cpus-per-task 4
#SBATCH --array=0-4

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs
module load anaconda
source activate "${VIP_ENV:-vip_venv11}"

DOMAINS=(${VIP_DOMAINS:-blocksworld hanoi npuzzle depot gripper})
DOMAIN="${DOMAINS[$SLURM_ARRAY_TASK_ID]}"
VENDOR="${VIP_VENDOR:?set VIP_VENDOR=openai|vllm}"
case "${DOMAIN}" in
  depot|gripper) FRAMES="src/domains/${DOMAIN}/problems" ;;
  *) FRAMES="benchmark/data/${DOMAIN}/precheck_frames__seed7/training/trajectories" ;;
esac

# VIP_VLM_JOB=<serve_vlm job id>: read the serving node from that job's endpoint
# file and wait until its server answers, so the array can be queued with
# --dependency=after:<id> before the model has loaded.
if [ -n "${VIP_VLM_JOB:-}" ]; then
  ENDPOINT_FILE="logs/vlm_endpoint-${VIP_VLM_JOB}.txt"
  # The file is re-read on every poll: a preempted serving job is requeued
  # under the same id and may come back on another node.
  for _ in $(seq 1 240); do
    if [ -s "${ENDPOINT_FILE}" ]; then
      export VIP_LLM_BASE_URL="http://$(cat "${ENDPOINT_FILE}")/v1"
      curl -sf -m 10 "${VIP_LLM_BASE_URL}/models" >/dev/null && break
    fi
    sleep 30
  done
  curl -sf -m 10 "${VIP_LLM_BASE_URL:-http://unset}/models" >/dev/null || { echo "server of job ${VIP_VLM_JOB} never answered (${VIP_LLM_BASE_URL:-no endpoint file})" >&2; exit 1; }
fi

echo "node=$(hostname) domain=${DOMAIN} vendor=${VENDOR} base_url=${VIP_LLM_BASE_URL:-<config>} frames=${FRAMES}"
python -m benchmark.vlm_precheck --domain "${DOMAIN}" --vendor "${VENDOR}" \
  --frames-root "${FRAMES}" --problems "${VIP_PROBLEMS:-5}" \
  --inference-workers "${VIP_WORKERS:-1}" --label "${VIP_LABEL:-precheck}" \
  ${VIP_OWN_FRAME0:+--own-frame0}
