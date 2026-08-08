#!/bin/bash
# Submit a manifest as a SLURM array, computing the array range from the file.
#
# Usage (from the project root on the cluster):
#   scripts/cluster/submit.sh                                   # manifest.csv, throttle 15
#   scripts/cluster/submit.sh scripts/cluster/manifest.csv 10   # explicit manifest + throttle
#   DRY_RUN=1 scripts/cluster/submit.sh                         # print the command, don't submit
#
# The sbatch template is picked automatically: a manifest whose header has a
# "fold" column uses run_fold.sbatch, otherwise run_cell.sbatch.

set -euo pipefail

MANIFEST="${1:-scripts/cluster/manifest.csv}"
THROTTLE="${2:-15}"

if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: manifest not found: $MANIFEST (run scripts/cluster/make_manifest.py first)" >&2
    exit 1
fi

N=$(( $(wc -l < "$MANIFEST") - 1 ))   # minus header
if [ "$N" -le 0 ]; then
    echo "ERROR: manifest has no task rows: $MANIFEST" >&2
    exit 1
fi

if head -1 "$MANIFEST" | grep -q ',fold$'; then
    TEMPLATE="scripts/cluster/run_fold.sbatch"
else
    TEMPLATE="scripts/cluster/run_cell.sbatch"
fi

mkdir -p logs

CMD=(sbatch --array="0-$((N - 1))%${THROTTLE}" "$TEMPLATE" "$MANIFEST")
echo "${CMD[*]}"
if [ "${DRY_RUN:-0}" != "1" ]; then
    "${CMD[@]}"
    echo
    echo "Monitor:    squeue --me"
    echo "Per-task logs under logs/ ; failed indices: sacct -j <array_job_id> | grep -Ev 'COMPLETED|RUNNING|PENDING'"
    echo "Re-run only failed indices (idempotent thanks to --resume):"
    echo "  sbatch --array=<i,j,k> $TEMPLATE $MANIFEST"
fi
