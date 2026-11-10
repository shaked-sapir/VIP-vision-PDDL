#!/bin/bash
# Render the large corpora on the laptop, one domain after another.
#
#   scripts/cluster_large/render_local.sh                 # depot and gripper (need Node)
#   scripts/cluster_large/render_local.sh all             # all five domains
#   LIMIT=3 scripts/cluster_large/render_local.sh all     # three problems per domain
#
# Idempotent: finished problem folders are skipped, so rerun after a stop.
# depot and gripper go through planning-visualizer's Node canvas worker;
# VIP_PLANNING_VISUALIZER overrides the checkout (default: the sibling
# ../planning-visualizer, or rendering.planning_visualizer_dir in config.yaml).
set -euo pipefail
cd "$(dirname "$0")/../.."

export MPLBACKEND=Agg
PY="${PY:-venv11/bin/python}"
LIMIT_ARGS=()
if [ -n "${LIMIT:-}" ]; then LIMIT_ARGS=(--limit "$LIMIT"); fi

render() {
    "$PY" benchmark/render_corpus.py --domain "$1" --corpus "$2" --domain-file "$3" "${LIMIT_ARGS[@]}"
}

render depot   benchmark/data/depot/large_corpus_n2500   src/domains/depot/depot.pddl
render gripper benchmark/data/gripper/large_corpus_n2500 src/domains/gripper/gripper.pddl

if [ "${1:-}" = "all" ]; then
    render blocksworld benchmark/data/blocksworld/large_corpus_L500_n2500 src/domains/blocks/blocks.pddl
    render hanoi       benchmark/data/hanoi/large_corpus_n2500            src/domains/hanoi/hanoi.pddl
    render npuzzle     benchmark/data/npuzzle/large_corpus_n2500          src/domains/n_puzzle/n_puzzle.pddl
fi
