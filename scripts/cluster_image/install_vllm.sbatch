#!/bin/bash
### One-time: a conda env with vLLM for serve_vlm.sbatch, built on a compute
### node rather than the login node. Submit from the project root:
###   sbatch scripts/cluster_image/install_vllm.sbatch
#SBATCH --partition main
#SBATCH --job-name vip-vllm-install
#SBATCH --output logs/%x-%j.out
#SBATCH --time 0-03:00:00
#SBATCH --gpus=0
#SBATCH --mem 16G
#SBATCH --cpus-per-task 4

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs
# The account's pip.conf names an NVIDIA index the nodes cannot resolve; pip
# retries it for every package, so the config files are ignored here.
export PIP_CONFIG_FILE=/dev/null
module load anaconda
ENV="${VIP_VLM_ENV:-vip_vllm}"
if ! conda env list | awk '{print $1}' | grep -qx "${ENV}"; then
  conda create -y -n "${ENV}" python=3.12
fi
source activate "${ENV}"
pip install --upgrade pip
pip install "vllm${VIP_VLLM_VERSION:+==$VIP_VLLM_VERSION}"
python -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.version.cuda)"
