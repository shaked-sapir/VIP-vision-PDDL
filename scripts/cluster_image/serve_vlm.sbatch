#!/bin/bash
### An open VLM behind vLLM's OpenAI-compatible server, on one GPU node.
###
### Submit from the project root, picking the GPU on the command line:
###   sbatch --constraint rtx_6000 scripts/cluster_image/serve_vlm.sbatch
### Environment knobs (all optional):
###   VIP_VLM_ENV     conda env with vllm installed      (default vip_vllm)
###   VIP_VLM_MODEL   Hugging Face model id               (default Qwen/Qwen3.8-27B)
###   VIP_VLM_NAME    --served-model-name, the config.yaml vllm.*.model_name (default qwen3.8-27b)
###   VIP_VLM_PORT    port on the node                    (default 8000)
###   VIP_VLM_MAXSEQS concurrent sequences (default 32; a hybrid-attention model
###                   such as Qwen3.5/3.8 has one Mamba cache block per sequence)
###   VIP_VLM_EXTRA   extra `vllm serve` arguments, e.g. a reasoning parser
###   VIP_VLM_MAXLEN  context length in tokens            (default 12288: a 24 GB card has ~1.8 GB of KV cache left after the 8B weights)
### The node and port land in logs/vlm_endpoint-<jobid>.txt; reach them from a
### laptop with  ssh -N -L 8000:<node>:<port> bgu  and config.yaml's vllm block.
#SBATCH --partition main
#SBATCH --job-name vip-vlm
#SBATCH --output logs/%x-%j.out
#SBATCH --time 1-00:00:00
#SBATCH --gpus=1
#SBATCH --mem 48G
#SBATCH --cpus-per-task 6

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"   # submit from the project root
mkdir -p logs

# The account's pip.conf names an NVIDIA index the nodes cannot resolve; pip
# retries it for every package, so the config files are ignored here.
export PIP_CONFIG_FILE=/dev/null
module load anaconda
ENV="${VIP_VLM_ENV:-vip_vllm}"
if ! conda env list | awk '{print $1}' | grep -qx "${ENV}"; then
  echo "creating ${ENV} (first run)"
  conda create -y -n "${ENV}" python=3.12
fi
source activate "${ENV}"
if ! python -c "import vllm" 2>/dev/null; then
  echo "installing vllm into ${ENV}"
  pip install --upgrade pip
  pip install "vllm${VIP_VLLM_VERSION:+==$VIP_VLLM_VERSION}"
fi
python -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.version.cuda)"

MODEL="${VIP_VLM_MODEL:-Qwen/Qwen3.8-27B}"
NAME="${VIP_VLM_NAME:-qwen3.8-27b}"
PORT="${VIP_VLM_PORT:-8000}"
MAXLEN="${VIP_VLM_MAXLEN:-12288}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

echo "node=$(hostname) job=${SLURM_JOB_ID} model=${MODEL} served_as=${NAME} port=${PORT}"
nvidia-smi -L
nvidia-smi --query-gpu=driver_version,memory.total --format=csv,noheader
echo "$(hostname):${PORT}" > "logs/vlm_endpoint-${SLURM_JOB_ID}.txt"

exec vllm serve "${MODEL}" \
  --served-model-name "${NAME}" \
  --host 0.0.0.0 --port "${PORT}" \
  --dtype bfloat16 \
  --max-model-len "${MAXLEN}" \
  --gpu-memory-utilization 0.92 \
  --limit-mm-per-prompt '{"image": 4}' \
  --enable-prefix-caching \
  --max-num-seqs "${VIP_VLM_MAXSEQS:-32}" \
  ${VIP_VLM_EXTRA:-}
