"""Write render_manifest.csv: one row per (domain, chunk) for render_corpus.sbatch.

    python scripts/cluster_large/make_render_manifest.py            # 10 chunks per domain
    python scripts/cluster_large/make_render_manifest.py --chunks 25

Only the PDDLGym domains are listed; depot and gripper render locally
(scripts/cluster_large/render_local.sh).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = Path(__file__).resolve().parent / "render_manifest.csv"

CLUSTER_CORPORA = {
    "blocksworld": ("benchmark/data/blocksworld/large_corpus_L500_n2500",
                    "src/domains/blocks/blocks.pddl"),
    "hanoi": ("benchmark/data/hanoi/large_corpus_n2500", "src/domains/hanoi/hanoi.pddl"),
    "npuzzle": ("benchmark/data/npuzzle/large_corpus_n2500",
                "src/domains/n_puzzle/n_puzzle.pddl"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--chunks", type=int, default=10, help="Array tasks per domain")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    rows = [
        (domain, corpus, domain_file, chunk, args.chunks)
        for domain, (corpus, domain_file) in CLUSTER_CORPORA.items()
        for chunk in range(args.chunks)
    ]
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["domain", "corpus", "domain_file", "chunk", "num_chunks"])
        writer.writerows(rows)
    print(f"{len(rows)} rows -> {args.out}  (sbatch --array=0-{len(rows) - 1})")


if __name__ == "__main__":
    main()
