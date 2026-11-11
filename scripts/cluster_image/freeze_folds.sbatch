#!/bin/bash
### The setup pass of an image run config (benchmark/freeze_image_folds.py), one
### job: prepares each domain's pool once and freezes every cell's inputs under
### $VIP_FREEZE_OUT. Run it before the symbolic and the pixel arrays of those
### domains, and never two of these jobs on one domain at a time.
###
### Submit from the project root:
###   sbatch scripts/cluster_image/freeze_folds.sbatch benchmark/run_config_large_image.yaml blocksworld
###   sbatch scripts/cluster_image/freeze_folds.sbatch benchmark/run_config_small_image.yaml
###
###   VIP_FREEZE_OUT   the frozen tree (default benchmark/running_results_image_src)
###   VIP_FREEZE_ARGS  further freeze_image_folds flags (default --test-states;
###                    set it to an empty string to skip the test states)
###   VIP_ENV          conda env (default vip_venv11)
#SBATCH --partition main
#SBATCH --job-name vip-img-freeze
#SBATCH --output logs/%x-%j.out
#SBATCH --time 0-08:00:00
#SBATCH --cpus-per-task 6
#SBATCH --gpus=0
#SBATCH --mem 16G

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

CONFIG="${1:?usage: freeze_folds.sbatch <image run config> [domain ...]}"
shift
OUT_ROOT="${VIP_FREEZE_OUT:-benchmark/running_results_image_src}"
EXTRA_ARGS="${VIP_FREEZE_ARGS---test-states}"

echo "SLURM_JOBID=${SLURM_JOB_ID:-?} node=${SLURM_JOB_NODELIST:-?} commit=$(cat PINNED_COMMIT 2>/dev/null || echo unpinned)"
echo "config=${CONFIG} domains=${*:-all} out_root=${OUT_ROOT} extra_args=${EXTRA_ARGS}"

module load anaconda
source activate "${VIP_ENV:-vip_venv11}"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

DOMAIN_ARGS=()
if [ "$#" -gt 0 ]; then
    DOMAIN_ARGS=(--domains "$@")
fi

python -m benchmark.freeze_image_folds \
    --config "$CONFIG" \
    --out-root "$OUT_ROOT" \
    ${DOMAIN_ARGS[@]+"${DOMAIN_ARGS[@]}"} ${EXTRA_ARGS}
