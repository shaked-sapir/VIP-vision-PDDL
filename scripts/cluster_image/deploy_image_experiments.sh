#!/bin/bash
# Build a pinned copy of one commit on the cluster for the image experiments,
# with the corpora their manifests name.
#
#   scripts/cluster_image/deploy_image_experiments.sh [commit] [remote dir] [ssh host] [manifest ...]
#
# Defaults: HEAD, projects/VIP-imgexp, bgu, and the two pixel manifests
# (image_small_pixel_manifest.csv, image_large_pixel_manifest.csv). A manifest
# is any csv whose second column is a data folder.
#
# The code is `git archive <commit>` (no working-tree edits, no benchmark/data);
# PINNED_COMMIT in the remote dir records which. config.yaml is taken from
# $VIP_CONFIG_FROM on the cluster. Of each data folder this copies the image
# pool (pddl, GT json, inferred .trajectory and .masking_info, and the frames of
# every problem that has inferred files), gt_trajectories/ and the top-level files. The symbolic pool, the normalized
# pools and the pool's prepared *_gtrate*_frame_axioms files stay behind; the
# setup pass (freeze_folds.sbatch) writes the prepared files on the cluster.
#
#   VIP_DEPLOY_DRY_RUN=1   print what rsync would copy, change nothing remote
set -euo pipefail

COMMIT="$(git rev-parse "${1:-HEAD}")"
REMOTE_DIR="${2:-projects/VIP-imgexp}"
HOST="${3:-bgu}"
if [ "$#" -gt 3 ]; then
    MANIFESTS=("${@:4}")
else
    MANIFESTS=(scripts/cluster_image/image_small_pixel_manifest.csv
               scripts/cluster_image/image_large_pixel_manifest.csv)
fi
CONFIG_FROM="${VIP_CONFIG_FROM:-projects/VIP-image/config.yaml}"
DRY_RUN="${VIP_DEPLOY_DRY_RUN:-0}"

REQUIRED=(benchmark/freeze_image_folds.py
          scripts/cluster_image/symbolic_fold.sbatch
          scripts/cluster_image/freeze_folds.sbatch
          scripts/cluster_image/pixel_fold.sbatch
          "${MANIFESTS[@]}")
for FILE in "${REQUIRED[@]}"; do
    if ! git cat-file -e "$COMMIT:$FILE" 2>/dev/null; then
        echo "ERROR: $FILE is not in commit $COMMIT; commit it before deploying" >&2
        exit 1
    fi
done

DATA_DIRS="$(for MANIFEST in "${MANIFESTS[@]}"; do tail -n +2 "$MANIFEST" | cut -d, -f2; done | sort -u)"
while IFS= read -r DATA_DIR; do
    if [ ! -d "$DATA_DIR/training/image_trajectories" ] || [ ! -d "$DATA_DIR/gt_trajectories" ]; then
        echo "ERROR: $DATA_DIR lacks training/image_trajectories/ or gt_trajectories/" >&2
        exit 1
    fi
done <<< "$DATA_DIRS"

RSYNC_FLAGS=(-a)
if [ "$DRY_RUN" = "1" ]; then
    RSYNC_FLAGS+=(--dry-run --stats)
else
    ssh "$HOST" "mkdir -p '$REMOTE_DIR/logs'"
    git archive "$COMMIT" -- . ':!benchmark/data' | ssh "$HOST" "tar -x -C '$REMOTE_DIR'"
    ssh "$HOST" "echo '$COMMIT' > '$REMOTE_DIR/PINNED_COMMIT' && cp '$CONFIG_FROM' '$REMOTE_DIR/config.yaml'"
fi

while IFS= read -r DATA_DIR; do
    if [ "$DRY_RUN" != "1" ]; then
        ssh -n "$HOST" "mkdir -p '$REMOTE_DIR/$DATA_DIR'"
    fi
    # Frames of a problem with no inferred files are read by no arm.
    NO_FRAMES="$(mktemp)"
    for PROBLEM_DIR in "$DATA_DIR"/training/image_trajectories/*/; do
        NAME="$(basename "$PROBLEM_DIR")"
        if [ ! -f "$PROBLEM_DIR/$NAME.masking_info" ]; then
            echo "/training/image_trajectories/$NAME/state_*.png" >> "$NO_FRAMES"
        fi
    done
    rsync "${RSYNC_FLAGS[@]}" \
        --exclude-from "$NO_FRAMES" \
        --exclude '/training/trajectories/' \
        --exclude '/training/trajectories_normalized/' \
        --exclude '/training/image_trajectories_normalized/' \
        --exclude '/testing/' \
        --exclude '*_gtrate*_frame_axioms.*' \
        --exclude '*_gtinit.*' \
        "$DATA_DIR/" "$HOST:$REMOTE_DIR/$DATA_DIR/"
    rm -f "$NO_FRAMES"
    echo "copied $DATA_DIR ($(find "$DATA_DIR/training/image_trajectories" -name '*.masking_info' -not -name '*_gtrate*' | wc -l | tr -d ' ') problems with inferred files)"
done <<< "$DATA_DIRS"
echo "pinned $COMMIT at $HOST:$REMOTE_DIR"
