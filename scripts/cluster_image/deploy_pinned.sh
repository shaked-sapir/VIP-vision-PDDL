#!/bin/bash
# Build a pinned copy of one commit on the cluster for the pixel-arm runs.
#
#   scripts/cluster_image/deploy_pinned.sh [commit] [remote dir] [ssh host]
#
# The code is `git archive <commit>` (no working-tree edits, no benchmark/data),
# so a job reads exactly what the commit holds; PINNED_COMMIT in the remote dir
# records which. The manifest's data folders and a slim copy of each image
# experiment (run_params.json and, per cell, fold_info.json,
# domain_reference.pddl and original_observations/) are copied from this
# checkout. config.yaml is taken from $VIP_CONFIG_FROM on the cluster.
set -euo pipefail

COMMIT="$(git rev-parse "${1:-HEAD}")"
REMOTE_DIR="${2:-projects/VIP-pixel}"
HOST="${3:-bgu}"
CONFIG_FROM="${VIP_CONFIG_FROM:-projects/VIP-image/config.yaml}"
MANIFEST="scripts/cluster_image/pixel_fold_manifest.csv"
SRC_ROOT="benchmark/running_results_image_src"

ssh "$HOST" "mkdir -p '$REMOTE_DIR/logs'"
git archive "$COMMIT" -- . ':!benchmark/data' | ssh "$HOST" "tar -x -C '$REMOTE_DIR'"
ssh "$HOST" "echo '$COMMIT' > '$REMOTE_DIR/PINNED_COMMIT' && cp '$CONFIG_FROM' '$REMOTE_DIR/config.yaml'"

tail -n +2 "$MANIFEST" | cut -d, -f1-3 | sort -u | while IFS=, read -r DOMAIN DATA_DIR EXPERIMENT; do
    ssh -n "$HOST" "mkdir -p '$REMOTE_DIR/$DATA_DIR' '$REMOTE_DIR/$SRC_ROOT/$DOMAIN/$EXPERIMENT'"
    rsync -a "$DATA_DIR/" "$HOST:$REMOTE_DIR/$DATA_DIR/"
    rsync -a --prune-empty-dirs \
        --include '*/' \
        --include 'evaluation_results/run_params.json' \
        --include 'testing/*/fold_info.json' \
        --include 'testing/*/domain_reference.pddl' \
        --include 'testing/*/original_observations/***' \
        --exclude '*' \
        "benchmark/running_results/$DOMAIN/$EXPERIMENT/" \
        "$HOST:$REMOTE_DIR/$SRC_ROOT/$DOMAIN/$EXPERIMENT/"
    echo "copied $DOMAIN: $DATA_DIR + $EXPERIMENT"
done
echo "pinned $COMMIT at $HOST:$REMOTE_DIR"
