#!/bin/bash
# Diagnostic: is ROSAME_MILP_24's frozen agreement a budget problem or a signal
# problem?
#
# In the sweep the arm runs epochs=100 with pre_mip_epochs=50, so the MILP gets
# 50 epochs to move the network -- and across depot, gripper, hanoi and npuzzle
# the agreement is identical in all 51 rounds (one distinct value in three of
# them). Either 50 epochs is too few to pull the net off its warmup fixed point,
# or the pseudo-label CE term is too weak to move it at all.
#
# This gives the MILP 250 epochs instead of 50, changing nothing else. Agreement
# that starts moving means the budget; agreement still frozen after 250 rounds
# means the CE weight.
#
#   sbatch --export=ALL,PROBE_DOMAIN=depot scripts/cluster_large/epoch_probe.sbatch
#SBATCH --partition main
#SBATCH --job-name ep-probe
#SBATCH --output logs/%x-%A_%a.out
#SBATCH --time 0-08:00:00
#SBATCH --cpus-per-task 6
#SBATCH --constraint cpu256
#SBATCH --gpus=0
#SBATCH --mem 24G
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
DOM="${PROBE_DOMAIN:?set PROBE_DOMAIN}"
L="${PROBE_L:-100}"
EPOCHS="${PROBE_EPOCHS:-300}"
echo "ep-probe: domain=$DOM L=$L epochs=$EPOCHS pre_mip=50 mask=0.0 noise=0.2 fold=0"
module load anaconda; source activate "${VIP_ENV:-vip_venv11}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
CFG="$(mktemp /tmp/epcfg_XXXX.yaml)"
python - "$L" "$EPOCHS" "$CFG" <<'PY'
import sys, yaml
L, ep, dst = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
c = yaml.safe_load(open("benchmark/run_config_large.yaml"))
c["run_name"] = f"epprobe__ep{ep}__L{L}"
c["shared"]["num_trajectories"] = [L]
c["shared"]["folds"] = [0]
c["shared"]["resume"] = False
c["shared"]["epochs"] = ep
c["shared"]["pre_mip_epochs"] = 50
c["shared"]["algorithms"] = ["rosame_milp_24"]
yaml.safe_dump(c, open(dst, "w"), sort_keys=False)
PY
python -m benchmark.benchmark_runner --config "$CFG" \
    --domains "$DOM" --only-mask 0.0 --only-noise 0.2 --only-fold 0
rm -f "$CFG"
