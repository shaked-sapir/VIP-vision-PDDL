"""Domain assets for the vendored ROSAME ``Convertor`` and DL head.

Three files are resolved by domain name. ``Convertor.__init__`` reads two,
relative to the vendor tree root:

    vendor/planning_structs/specs/<domain>/domain.json   (the CP domain spec)
    vendor/pddl/<domain>/domain.pddl                     (the `gt_am` reference)

and ``ROSAMEMixin._build_around`` reads a third pair through ``get_domain_model``:

    <assets_root>/<domain>/domain_model.json             (the DL head's vocabulary)
    <assets_root>/<domain>/objects.json                  (its grounding universe)

Upstream ships all of these per-domain and hand-written. This module regenerates
them from our own ``src/domains/*.pddl``, so every vocabulary is derived from the
same file the rest of the pipeline parses rather than maintained in parallel.

The first two are static and checked in. The second pair is *not*: upstream's
``objects.json`` is a fixed universe (blocks ships five blocks), while ours is
the per-fold object union, so ``write_grounding_assets`` writes both into a
run-scoped directory that the caller passes as ``domain_assets_root``.

Directories are named by **benchmark domain key** (``blocksworld``, ``npuzzle``,
...), which is what the runners already pass around, so ``Convertor``'s
hardcoded ``"blocks" -> "blocksworld"`` alias stays a no-op and the vendored
file needs no edit.

Regenerate the checked-in assets after changing any domain PDDL::

    python -m src.milp.domain_assets            # all domains
    python -m src.milp.domain_assets blocksworld
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from pddl_plus_parser.lisp_parsers import DomainParser

import src.milp  # noqa: F401  (vendor sys.path bootstrap)
from src.milp.converter import build_ps_domain
from src.utils.config import load_config

from planning_structs.domain import Domain as PSDomain
from planning_structs.domain import Type as PSType
from planning_structs.util import (
    collect_types_hierarchy,
    dump_domain_to_json,
    load_domain_from_json,
)

_VENDOR_ROOT = Path(__file__).parent / "vendor"
SPECS_ROOT = _VENDOR_ROOT / "planning_structs" / "specs"
PDDL_ROOT = _VENDOR_ROOT / "pddl"

# The domains carried by the image experiments the ROSAME arms report on.
BENCH_DOMAINS: List[str] = ["blocksworld", "depot", "gripper", "hanoi", "npuzzle"]


def spec_path(domain_key: str) -> Path:
    """Where ``Convertor`` looks for the CP domain spec of ``domain_key``."""
    return SPECS_ROOT / domain_key / "domain.json"


def reference_pddl_path(domain_key: str) -> Path:
    """Where ``Convertor`` looks for the ``gt_am`` reference PDDL of ``domain_key``."""
    return PDDL_ROOT / domain_key / "domain.pddl"


def source_pddl_path(domain_key: str) -> Path:
    """The project's canonical domain PDDL, from ``config.yaml``."""
    config = load_config()
    domains = config["domains"]
    if domain_key not in domains:
        raise KeyError(
            f"'{domain_key}' is not a domain in config.yaml; known: {sorted(domains)}"
        )
    entry = domains[domain_key]
    if "domain_file" not in entry:
        raise KeyError(f"config.yaml domains.{domain_key} has no 'domain_file'")
    return Path(entry["domain_file"])


def build_domain(domain_key: str) -> PSDomain:
    """Parse the project's domain PDDL into a vendored ``Domain``."""
    pddl_domain = DomainParser(source_pddl_path(domain_key), partial_parsing=True).parse_domain()
    return build_ps_domain(pddl_domain)


def type_counts(types: Sequence[PSType]) -> Dict[str, int]:
    """The ``{type_name: count}`` map ROSAME stores instead of an ordered signature."""
    counts: Dict[str, int] = {}
    for one_type in types:
        counts[one_type.name] = counts.get(one_type.name, 0) + 1
    return counts


def rosame_argument_permutation(types: Sequence[PSType]) -> List[int]:
    """PDDL argument index for each ROSAME argument slot.

    ``Predicate.__init__`` sorts the count map's keys by type name, and
    ``Predicate.ground`` emits one ``itertools.permutations`` group per key in
    that order, so a PDDL signature that is not already grouped and name-sorted
    is grounded as a permutation of itself.

    A stable sort by type name, which keeps same-typed arguments in their PDDL
    order. That matters for a signature repeating a type at non-adjacent
    positions, such as hanoi's ``move_peg_disc(disc, peg, disc)``: ``ground``
    permutes rather than combines, so the two ``disc`` slots stay ordered and
    distinct and no argument identity is lost.
    """
    return sorted(range(len(types)), key=lambda i: types[i].name)


def rosame_argument_order(types: Sequence[PSType]) -> List[str]:
    """The argument type sequence ROSAME reconstructs from :func:`type_counts`."""
    return [types[i].name for i in rosame_argument_permutation(types)]


def build_rosame_domain_model_dict(domain_key: str) -> Dict[str, Any]:
    """The ``domain_model.json`` payload for ``Domain_Model.create_from_json``."""
    domain = build_domain(domain_key)
    return {
        "types": [
            {"name": t.name, "parent": (t.parent.name if t.parent else None)}
            for t in collect_types_hierarchy(domain.types_set)
        ],
        "predicates": [
            {"name": p.name, "params": type_counts(p.types)} for p in domain.predicates
        ],
        "action_schemas": [
            {"name": a.name, "params": type_counts(a.types)} for a in domain.action_schemas
        ],
    }


def write_grounding_assets(
    domain_key: str, objects: Mapping[str, Sequence[str]], root: Path
) -> Path:
    """Write ``domain_model.json`` + ``objects.json`` for one run into ``root``.

    ``root`` is what the caller passes to the network as ``domain_assets_root``;
    ``get_domain_model`` appends ``domain_key`` to it. Returns that subdirectory.
    """
    declared = {t.name for t in build_domain(domain_key).types_set}
    unknown = sorted(set(objects) - declared)
    if unknown:
        raise ValueError(
            f"objects for '{domain_key}' use types not declared in its domain PDDL: "
            f"{unknown}; known types are {sorted(declared)}"
        )
    if not any(objects.values()):
        raise ValueError(f"objects for '{domain_key}' are empty; nothing would ground")

    out_dir = Path(root) / domain_key
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "domain_model.json").write_text(
        json.dumps(build_rosame_domain_model_dict(domain_key), indent=4), encoding="utf-8"
    )
    (out_dir / "objects.json").write_text(
        json.dumps({t: list(names) for t, names in objects.items()}, indent=4),
        encoding="utf-8",
    )
    return out_dir


def generate(domain_key: str) -> Dict[str, Path]:
    """Write the spec JSON and the reference PDDL for one domain."""
    source = source_pddl_path(domain_key)
    if not source.exists():
        raise FileNotFoundError(f"domain PDDL for '{domain_key}' not found at {source}")

    spec_out = spec_path(domain_key)
    spec_out.parent.mkdir(parents=True, exist_ok=True)
    dump_domain_to_json(build_domain(domain_key), spec_out)

    pddl_out = reference_pddl_path(domain_key)
    pddl_out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, pddl_out)

    return {"spec": spec_out, "pddl": pddl_out}


def generate_all(domain_keys: List[str] | None = None) -> Dict[str, Dict[str, Path]]:
    """Write assets for every domain in ``domain_keys`` (default: `BENCH_DOMAINS`)."""
    return {key: generate(key) for key in (domain_keys or BENCH_DOMAINS)}


def main(argv: List[str]) -> int:
    keys = argv[1:] or BENCH_DOMAINS
    for key in keys:
        written = generate(key)
        domain = load_domain_from_json(written["spec"])
        print(
            f"{key:12s} {len(domain.predicates):3d} predicates "
            f"{len(domain.action_schemas):3d} schemas  -> {written['spec'].parent}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
