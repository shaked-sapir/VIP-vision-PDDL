#!/usr/bin/env bash
# Submit the large-corpora sweep: one array task per (domain, mask, noise) cell.
#
#   scripts/cluster_large/submit.sh                    # all rows, 15 at a time
#   scripts/cluster_large/submit.sh manifest.csv 8     # custom manifest / throttle
#   DRY_RUN=1 scripts/cluster_large/submit.sh          # print the sbatch, submit nothing
#
# Run from the project root, on the login node, after make_manifest.py.
set -euo pipefail

MANIFEST="${1:-scripts/cluster_large/manifest.csv}"
THROTTLE="${2:-15}"
TEMPLATE="scripts/cluster_large/run_cell.sbatch"

[ -f "$MANIFEST" ] || { echo "No manifest at $MANIFEST — run make_manifest.py first." >&2; exit 1; }
[ -f "$TEMPLATE" ] || { echo "No template at $TEMPLATE" >&2; exit 1; }

ROWS=$(($(wc -l < "$MANIFEST") - 1))   # minus the header
[ "$ROWS" -ge 1 ] || { echo "Manifest $MANIFEST has no rows." >&2; exit 1; }

mkdir -p logs

echo "manifest : $MANIFEST ($ROWS cells)"
echo "array    : 0-$((ROWS - 1))%${THROTTLE}"
echo "template : $TEMPLATE"
echo "config   : ${VIP_LARGE_CONFIG:-benchmark/run_config_large.yaml}"
echo "env      : ${VIP_ENV:-vip_venv11}"
echo

CMD=(sbatch --array="0-$((ROWS - 1))%${THROTTLE}" "$TEMPLATE" "$MANIFEST")
if [ -n "${DRY_RUN:-}" ]; then
    echo "DRY_RUN — would run:"; printf '  %q' "${CMD[@]}"; echo
    exit 0
fi
"${CMD[@]}"
