"""Audit the MILP arms against each other across a run's cells.

Produces the numbers quoted in docs/cdps-milp-denoiser-design.md 7.1a and 7.1b;
re-run it before citing them.

    PYTHONPATH=. python benchmark/evaluation/milp_arm_audit.py

Four things, per domain:

1. Row coverage -- which algorithm rows exist and on how many folds. A backfill
   that silently skipped folds shows up here as a count below the fold total.

2. The design 7.1 lower bound, cost(MILP) <= cost(best CDPS CFM). Neither cost
   is a top-level column: the MILP side is `algorithm_specific.milp_repair_cost`
   on the CDPS_MILP_SR row, and the CDPS side is not on the CDPS row at all --
   it has to come from the fold's conflict_free_solutions_log.json. Only
   eq16=off rows are checked; eq16 changes the objective, so T' differs and the
   bound says nothing there.

   The CDPS side is NOT the logged `cost`. Two corrections, both from P5.7:

   (a) `cost` counts raw patch-set members, but patches are applied by toggling
       (utils/pddl_state.flip_fluent_in_state matches {f, not f} and inverts),
       so a same-key polarity pair is two toggles = zero net edits charged as
       two. Use odd parity over normalized keys -- the rule now shipped as
       plan_denoising/patch_accounting. Logs written before that change have no
       `net_cost` field, so it is recomputed here from `fluent_patches`.

   (b) CDPS may also buy conflict-freeness with REQUIRE/FORBID model
       constraints, which cost 0.0 by CDPSConfig default. The MILP has no such
       move, so a CDPS model carrying Phi != {} is not a point the MILP could
       have reached and says nothing about its optimality. Only Phi == {}
       models are comparable; both figures are printed, because the restricted
       one turns out to be empty of anything but 0 <= 0.

3. The eq16 on/off A/B, and SR vs loop, on the metrics all arms report. Both are
   *paired* over the folds carrying both rows rather than a difference of
   independent means, since the arms share data, masking and fold split.

4. Loop round counts and stop reasons. A loop whose null result comes from
   timing out is a budget finding; one that reaches `fixpoint` has genuinely
   exhausted its subset space. Without this the two are indistinguishable.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

from src.plan_denoising.patch_accounting import net_patch_count_from_records

DEFAULT_ROOT = Path("benchmark/running_results")
DEFAULT_DOMAINS = ("blocksworld", "hanoi", "gripper", "npuzzle")
DEFAULT_CELLS = "simulation-final-run__*"

SR = "CDPS_MILP_SR"
SR_EQ16 = "CDPS_MILP_SR__eq16=0.4"
LOOP = "CDPS_MILP_LOOP"
METRICS = ("precision_overall", "recall_overall", "solving_ratio", "learning_time_seconds")


def load_json(path: Path):
    """Return parsed JSON, or None if the file is missing or unreadable."""
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def entry_net_cost(entry: dict) -> int:
    """Realized edits for one logged conflict-free model.

    Prefers the `net_cost` the writer now emits; recomputes it for logs written
    before that field existed, so the corpus on disk stays readable.
    """
    if isinstance(entry.get("net_fluent_patch_count"), int):
        return entry["net_fluent_patch_count"]
    return net_patch_count_from_records(entry.get("fluent_patches") or [])


def best_cfm_costs(fold_dir: Path) -> tuple[int | None, int | None]:
    """(cheapest CFM overall, cheapest CFM with no model constraints).

    The second is the only one the MILP could have reached; see the module
    docstring, note (b).
    """
    log = load_json(fold_dir / "conflict_free_solutions_log.json")
    entries = [e for e in (log or []) if isinstance(e, dict) and e.get("cost") is not None]
    if not entries:
        return None, None
    pure = [e for e in entries if e.get("model_constraint_count", 0) == 0]
    return (
        min(entry_net_cost(e) for e in entries),
        min((entry_net_cost(e) for e in pure), default=None),
    )


def row_by_alg(rows: list[dict]) -> dict[str, dict]:
    return {r.get("algorithm"): r for r in rows}


def audit_domain(domain: str, root: Path, cells: str) -> list[dict]:
    """Print the per-domain audit; return the domain's loop rows for the summary."""
    folds = []
    for cell in sorted(root.joinpath(domain).glob(cells)):
        for fr in sorted(cell.glob("testing/*/fold_result.json")):
            data = load_json(fr)
            if data is None:
                print(f"  !! unreadable {fr}")
                continue
            folds.append((fr.parent, row_by_alg(data)))

    print(f"\n=== {domain}: {len(folds)} folds")
    counts = Counter()
    for _, rows in folds:
        counts.update(rows.keys())
    for name, n in sorted(counts.items()):
        gap = "" if n == len(folds) else f"   <-- MISSING {len(folds) - n}"
        print(f"    {name:<28} {n}{gap}")

    # --- design 7.1 lower bound, eq16=off only
    #
    # A fold where CDPS produced no conflict-free model is not missing data: the
    # bound is vacuous there because there is no CDPS cost to be below. Counted
    # separately from a genuine no-MILP-cost fold, which would be a real gap.
    ok = bad = no_cfm = no_milp = 0
    p_ok = p_bad = p_none = p_nontrivial = 0
    worst = None
    for fold_dir, rows in folds:
        sr = rows.get(SR)
        if sr is None:
            continue
        milp = (sr.get("algorithm_specific") or {}).get("milp_repair_cost")
        cfm, pure = best_cfm_costs(fold_dir)
        if milp is None:
            no_milp += 1
            continue
        if cfm is None:
            no_cfm += 1
        elif milp <= cfm:
            ok += 1
        else:
            bad += 1
            if worst is None or milp - cfm > worst[0]:
                worst = (milp - cfm, fold_dir, milp, cfm)
        if pure is None:
            p_none += 1
        else:
            if milp <= pure:
                p_ok += 1
            else:
                p_bad += 1
            if milp > 0 or pure > 0:
                p_nontrivial += 1
    if ok or bad or no_cfm or no_milp:
        print(
            f"    net cost(MILP) <= net cost(any CFM):   ok={ok} violated={bad}"
            f"  [CDPS found no CFM: {no_cfm}]  [no MILP cost: {no_milp}]"
        )
        if worst:
            delta, fold_dir, milp, cfm = worst
            print(f"      worst: +{delta}  milp={milp} cfm={cfm}   {fold_dir.name}")
        # The comparable subset. `nontrivial` is the number that matters: a fold
        # where both sides are 0 passes the check without exercising it.
        print(
            f"    net cost(MILP) <= net cost(Phi={{}} CFM): ok={p_ok} violated={p_bad}"
            f"  [no pure-data CFM: {p_none}]  [of the {p_ok + p_bad} comparable,"
            f" non-trivial: {p_nontrivial}]"
        )

    # --- eq16 A/B and arm comparison, paired over folds carrying both
    for label, a, b in (("eq16 off -> on", SR, SR_EQ16), ("SR -> loop", SR, LOOP)):
        pairs = defaultdict(list)
        for _, rows in folds:
            ra, rb = rows.get(a), rows.get(b)
            if ra is None or rb is None:
                continue
            for m in METRICS:
                va, vb = ra.get(m), rb.get(m)
                if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                    pairs[m].append((va, vb))
        if not pairs:
            continue
        n = len(next(iter(pairs.values())))
        print(f"    {label}  (paired on {n} folds)")
        for m in METRICS:
            vals = pairs.get(m)
            if not vals:
                continue
            ma = statistics.mean(v for v, _ in vals)
            mb = statistics.mean(v for _, v in vals)
            print(f"      {m:<24} {ma:8.4f} -> {mb:8.4f}   delta={mb - ma:+.4f}")

    return [rows[LOOP] for _, rows in folds if LOOP in rows]


def report_loop_rounds(loop_rows: list[dict]) -> None:
    """Whether the loop's verdict was reached or merely timed out.

    Reported corpus-wide rather than per domain: the question is about the loop
    as a mechanism, and per-domain slices of a stop-reason histogram are noise.
    """
    rounds, stops = [], Counter()
    for row in loop_rows:
        spec = row.get("algorithm_specific") or {}
        count = spec.get("milp_loop_rounds")
        if isinstance(count, int):
            rounds.append(count)
        stops[str(spec.get("milp_loop_stop_reason"))] += 1
    if not rounds:
        return
    print(f"\n=== loop rounds over {len(rounds)} folds")
    print(
        f"    min={min(rounds)} median={statistics.median(rounds)}"
        f" mean={statistics.mean(rounds):.1f} max={max(rounds)}"
    )
    print(f"    stop reasons: {dict(stops.most_common())}")
    timed_out = sum(n for reason, n in stops.items() if "timeout" in reason.lower())
    verdict = "budget-limited" if timed_out else "exhaustive -- no timeouts"
    print(f"    -> the loop's result is {verdict}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                        help="results root (default: %(default)s)")
    parser.add_argument("--domains", nargs="+", default=list(DEFAULT_DOMAINS),
                        help="domains to audit (default: %(default)s)")
    parser.add_argument("--cells", default=DEFAULT_CELLS,
                        help="glob for experiment dirs under each domain "
                             "(default: %(default)s)")
    args = parser.parse_args()

    loop_rows: list[dict] = []
    for domain in args.domains:
        loop_rows.extend(audit_domain(domain, args.root, args.cells))
    report_loop_rounds(loop_rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
