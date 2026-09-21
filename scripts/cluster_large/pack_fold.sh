#!/bin/bash
# Sourced by the large-sweep job templates. Defines pack_fold_instances.
#
# The home quota on this cluster is a file-count problem: observations and
# ROSAME's workspace copies cost 5 files per trajectory, 13,300 per fold over the
# L sweep. Packing a fold's FINISHED instances leaves about 1,000.

# pack_fold_instances EXP_DIR FOLD
#   For every instance of FOLD under EXP_DIR/testing that has its
#   fold_result.json: remove ROSAME's workspace copies and the Lamanna trace
#   files, and turn original_observations/ into original_observations.tar.gz.
#   fold_result.json is the resume marker and stays, so a re-submitted job still
#   skips these instances; an instance without it is left whole for the re-run.
#   Unpack the tarball before a backfill_* pass or a dashboard --refresh-stats.
pack_fold_instances() {
    local exp_dir="$1" fold="$2" inst
    # Most cells have no *_workspace dir (the Lamanna arms are regime-gated), so
    # an unmatched glob is normal here, not the error failglob treats it as.
    local had_failglob=0
    shopt -q failglob && had_failglob=1
    shopt -u failglob
    shopt -s nullglob
    local instances=( "$exp_dir"/testing/fold"${fold}"_numtrajs*_gtrate* )
    if [ "${#instances[@]}" -eq 0 ]; then
        echo "WARNING: no fold ${fold} instances under ${exp_dir}/testing; nothing packed" >&2
    else
        for inst in "${instances[@]}"; do
            [ -f "$inst/fold_result.json" ] || continue
            rm -rf "$inst/temp_rosame_workspace" "$inst"/*_workspace/traces
            if [ -d "$inst/original_observations" ]; then
                tar czf "$inst/original_observations.tar.gz" -C "$inst" original_observations \
                    && rm -rf "$inst/original_observations"
            fi
        done
        echo "packed: $(find "${instances[@]}" -type f | wc -l) files left in fold ${fold}'s ${#instances[@]} instances"
    fi
    shopt -u nullglob
    [ "$had_failglob" -eq 1 ] && shopt -s failglob
    return 0
}
