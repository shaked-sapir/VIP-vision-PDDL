"""Build an HTML dashboard of CFM-quality plots across the mask x noise grid.

For every domain under the results root this produces one interactive page:
  - a tab per domain,
  - a metric selector and a per-metric / composite (6-panel) toggle,
  - a 3x3 grid of the mask x noise cells (rows = masking, cols = noising),
  - each subfigure subtitled with the real avg masked and flipped fluents/state,
  - click-to-enlarge lightbox.

Images are referenced by *relative path* to the PNGs already produced by the
plot-generation step (``build_grid_plots.py``), so the page always reflects the
latest plots but must be opened in place under the results root.

Fluent-count derivation (no ground-truth diff needed):
  masked/state    -> average number of predicates per line in ``.masking_info``
  candidates/state-> unmasked grounded predicates left after masking
  flipped/state   -> noise_ratio * candidates/state  (the percentage noiser
                     flips max(1, round(ratio * candidates)) of the unmasked
                     pool per state; the initial state t=0 is untouched GT)

Usage:
    python -m benchmark.evaluation.cfm.build_results_dashboard [--prefix sim_run]
        [--results-root benchmark/running_results] [--out results_dashboard.html]
        [--refresh-stats]
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Dict, List, Optional

from benchmark.evaluation.cfm.cfm_quality_analysis import (
    generate_cfm_quality_analysis,
    get_max_solution_index,
)

CELL_PATTERN = re.compile(r"mask=([0-9.]+)__noise=([0-9.]+)")


def parse_cell(dir_name: str) -> Optional[tuple[float, float]]:
    """Return (mask, noise) parsed from an experiment folder name, or None."""
    match = CELL_PATTERN.search(dir_name)
    if not match:
        return None
    return float(match.group(1)), float(match.group(2))


def find_grid_cells(domain_dir: Path, prefix: Optional[str]) -> List[Path]:
    """Return a domain's grid-cell experiment dirs, sorted by (mask, noise).

    Args:
        domain_dir: A domain folder under the results root.
        prefix: If given, only folders starting with this prefix are considered;
            otherwise any folder whose name encodes a mask/noise cell is used.
    """
    cells: List[Path] = []
    for child in domain_dir.iterdir():
        if not child.is_dir():
            continue
        if prefix is not None and not child.name.startswith(prefix):
            continue
        if parse_cell(child.name) is None:
            continue
        cells.append(child)
    return sorted(cells, key=lambda p: parse_cell(p.name))


def regenerate_raw_plots(cells: List[Path]) -> None:
    """Regenerate each cell's raw per-experiment plots (own x-limits).

    These live in ``evaluation_results/CFM_quality/`` and are meant for examining
    a single experiment on its own — no shared axis is applied.
    """
    for cell in cells:
        print(f"    raw plots: {cell.name}")
        generate_cfm_quality_analysis(cell)  # x_max=None -> own x-limits

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_RESULTS_ROOT = _PROJECT_ROOT / "benchmark" / "running_results"

# (metric_key, label) — key doubles as "<key>_trend.png".
METRICS: List[tuple[str, str]] = [
    ("pred_app_precision", "Predictive applicability precision"),
    ("pred_app_recall", "Predictive applicability recall"),
    ("pred_eff_precision", "Predicted effects precision"),
    ("pred_eff_recall", "Predicted effects recall"),
    ("solving_ratio", "Problem solving ratio"),
    ("fluent_patch_count", "Fluent patch count"),
]
COMPOSITE_PNG = "all_trends_summary.png"
_STATS_FILENAME = "_grid_fluent_stats.json"
# Shared-x image copies live here, separate from the raw per-experiment plots
# in CFM_quality/ (which keep their own x-limits).
SHARED_SUBDIR = "CFM_quality_shared"


def regenerate_shared_x_plots(cells: List[Path]) -> None:
    """Write shared-x-limit plot copies for a domain's grid cells.

    Computes the domain-wide max solution index and re-plots every cell with
    that value as the x-axis limit, into each cell's ``CFM_quality_shared/``
    dir. The raw ``CFM_quality/`` plots (own x-limits) are left untouched.
    """
    per_cell_max = {cell: (get_max_solution_index(cell) or 0) for cell in cells}
    domain_max = max(per_cell_max.values(), default=0)
    if domain_max == 0:
        return
    for cell in cells:
        if per_cell_max[cell] == 0:
            continue  # single CFM (no conflicts) — no trend to plot
        generate_cfm_quality_analysis(
            cell,
            x_max=domain_max,
            output_dir_override=cell / "evaluation_results" / SHARED_SUBDIR,
        )


# ── Fluent statistics ──────────────────────────────────────────────────────

def _masked_per_state_from_info(masking_info_path: Path) -> List[int]:
    """Return the masked-predicate count for each state (t>=1) of one problem.

    Reads the plain-text ``.masking_info`` file (no PDDL parsing needed). The
    first line is the initial state (always empty) and is skipped.
    """
    counts: List[int] = []
    with open(masking_info_path) as f:
        lines = [ln.strip() for ln in f]
    for line in lines[1:]:  # skip initial state
        counts.append(0 if not line else len(line.split(", ")))
    return counts


def compute_cell_fluent_stats(cell: Path, noise_ratio: float) -> Optional[Dict[str, float]]:
    """Compute avg masked / candidate / flipped fluents per state for one cell.

    ``masked/state`` is read directly from ``.masking_info``. ``candidates`` (the
    unmasked grounded fluents left after masking) requires PDDL parsing and is
    computed via ``load_masked_observation``; if parsing is unavailable the
    candidate/flipped figures are returned as ``None`` and only ``masked`` is set.

    Args:
        cell: A single mask/noise experiment directory.
        noise_ratio: The noise ratio parsed from the folder name.

    Returns:
        Dict with keys ``masked``, ``candidates``, ``flipped``, ``n_states``,
        or None when the cell has no masking data at all.
    """
    masked_counts: List[int] = []
    candidate_counts: List[int] = []

    for mf in sorted(cell.glob("testing/*/original_observations/*.masking_info")):
        masked_counts.extend(_masked_per_state_from_info(mf))

    if not masked_counts:
        return None

    # Candidate pool (unmasked grounded fluents) — needs PDDL parsing.
    try:
        candidate_counts = _candidate_counts_for_cell(cell)
    except Exception as exc:  # noqa: BLE001 - degrade gracefully, report once
        print(f"    [stats] candidate count unavailable for {cell.name}: {exc}")
        candidate_counts = []

    stats: Dict[str, float] = {
        "masked": round(statistics.mean(masked_counts), 2),
        "n_states": len(masked_counts),
        "candidates": None,
        "flipped": None,
    }
    if candidate_counts:
        avg_cand = statistics.mean(candidate_counts)
        flips = [
            max(1, round(noise_ratio * c)) if (c and noise_ratio) else 0
            for c in candidate_counts
        ]
        stats["candidates"] = round(avg_cand, 1)
        stats["flipped"] = round(statistics.mean(flips), 2)
    return stats


def _candidate_counts_for_cell(cell: Path) -> List[int]:
    """Unmasked grounded-predicate count per state (t>=1) across the cell.

    Uses each fold's ``domain_reference.pddl`` and the stored noisy
    observations. Imported lazily so the dashboard's dependency-free parts work
    even when the PDDL parser is not installed.
    """
    from pddl_plus_parser.lisp_parsers import DomainParser

    from src.utils.masking import load_masked_observation
    from src.utils.pddl_state import get_state_unmasked_predicates

    counts: List[int] = []
    for fold in sorted(cell.glob("testing/*")):
        domain_file = fold / "domain_reference.pddl"
        obs_dir = fold / "original_observations"
        if not domain_file.exists() or not obs_dir.is_dir():
            continue
        domain = DomainParser(domain_file, partial_parsing=True).parse_domain()
        for tf in sorted(obs_dir.glob("*.trajectory")):
            mf = tf.with_suffix(".masking_info")
            if not mf.exists():
                continue
            obs = load_masked_observation(tf, mf, domain)
            states = [obs.components[0].previous_state] + [
                c.next_state for c in obs.components
            ]
            for si, state in enumerate(states):
                if si == 0:
                    continue
                counts.append(len(get_state_unmasked_predicates(state)))
    return counts


def load_or_compute_domain_stats(
    domain_dir: Path, cells: List[Path], refresh: bool
) -> Dict[str, Dict]:
    """Return per-cell fluent stats for a domain, using a JSON cache."""
    cache_path = domain_dir / _STATS_FILENAME
    if cache_path.exists() and not refresh:
        cached = json.loads(cache_path.read_text())
        # Reuse the cache only if it is complete: every non-empty cell must have
        # a candidate/flipped count. A cache written where the PDDL parser was
        # unavailable (candidates=None) is treated as stale and recomputed.
        incomplete = any(
            v and v.get("candidates") is None for v in cached.values()
        )
        if not incomplete:
            return cached

    stats: Dict[str, Dict] = {}
    for cell in cells:
        mask, noise = parse_cell(cell.name)
        key = f"mask={mask}__noise={noise}"
        print(f"    computing fluent stats: {cell.name}")
        stats[key] = compute_cell_fluent_stats(cell, noise) or {}
    cache_path.write_text(json.dumps(stats, indent=2))
    return stats


# ── Data assembly ──────────────────────────────────────────────────────────

def build_dashboard_data(
    results_root: Path,
    prefix: Optional[str],
    domains: Optional[List[str]],
    refresh: bool,
    skip_plot_regen: bool,
    regen_raw: bool,
) -> Dict:
    """Scan the results tree and assemble the JSON payload for the page."""
    masks: set = set()
    noises: set = set()
    data: Dict[str, Dict] = {}

    domain_dirs = [
        d for d in sorted(results_root.iterdir())
        if d.is_dir() and (domains is None or d.name in domains)
    ]

    for domain_dir in domain_dirs:
        cells = find_grid_cells(domain_dir, prefix)
        if not cells:
            continue
        print(f"  {domain_dir.name}: {len(cells)} cells")
        if regen_raw:
            print("    regenerating raw per-experiment plots (own x-limits)...")
            regenerate_raw_plots(cells)
        if not skip_plot_regen:
            print("    generating shared-x plot copies...")
            regenerate_shared_x_plots(cells)
        cell_stats = load_or_compute_domain_stats(domain_dir, cells, refresh)

        domain_cells: Dict[str, Dict] = {}
        for cell in cells:
            mask, noise = parse_cell(cell.name)
            masks.add(mask)
            noises.add(noise)
            key = f"mask={mask}__noise={noise}"
            cfm_dir = cell / "evaluation_results" / SHARED_SUBDIR

            pngs = {
                mk: str((cfm_dir / f"{mk}_trend.png").relative_to(results_root))
                for mk, _ in METRICS
                if (cfm_dir / f"{mk}_trend.png").exists()
            }
            composite = cfm_dir / COMPOSITE_PNG
            composite_rel = (
                str(composite.relative_to(results_root)) if composite.exists() else None
            )

            if pngs:
                status = "ok"
            elif (get_max_solution_index(cell) or 0) == 0:
                status = "single"      # noise=0 -> one CFM, no trend
            else:
                status = "missing"

            domain_cells[key] = {
                "status": status,
                "pngs": pngs,
                "composite": composite_rel,
                "stats": cell_stats.get(key, {}),
            }
        data[domain_dir.name] = domain_cells

    return {
        "domains": list(data.keys()),
        "masks": sorted(masks),
        "noises": sorted(noises),
        "metrics": METRICS,
        "cells": data,
    }


# ── HTML rendering ─────────────────────────────────────────────────────────

def render_html(payload: Dict) -> str:
    """Render the self-contained dashboard HTML for the given payload."""
    return _HTML_TEMPLATE.replace("__DATA__", json.dumps(payload))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an HTML dashboard of CFM-quality grid plots."
    )
    parser.add_argument("--prefix", type=str, default=None,
                        help="Experiment-name prefix shared across domains (e.g. 'sim_run').")
    parser.add_argument("--domains", nargs="*", default=None,
                        help="Restrict to these domain folder names.")
    parser.add_argument("--results-root", type=Path, default=_DEFAULT_RESULTS_ROOT)
    parser.add_argument("--out", type=Path, default=None,
                        help="Output HTML path (default: <results-root>/results_dashboard.html).")
    parser.add_argument("--refresh-stats", action="store_true",
                        help="Recompute cached per-domain fluent statistics.")
    parser.add_argument("--skip-plot-regen", action="store_true",
                        help="Reuse existing shared-x plot copies instead of regenerating them.")
    parser.add_argument("--regen-raw", action="store_true",
                        help="Also regenerate each cell's raw per-experiment plots (own x-limits) "
                             "in CFM_quality/ before building the shared-x copies.")
    args = parser.parse_args()

    if not args.results_root.is_dir():
        raise FileNotFoundError(f"results root not found: {args.results_root}")

    payload = build_dashboard_data(
        args.results_root, args.prefix, args.domains,
        args.refresh_stats, args.skip_plot_regen, args.regen_raw,
    )
    if not payload["domains"]:
        print("No grid cells found. Nothing to build.")
        return

    out_path = args.out or (args.results_root / "results_dashboard.html")
    out_path.write_text(render_html(payload))
    print(f"\nDashboard written to: {out_path}")
    print(f"Domains: {', '.join(payload['domains'])}")


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>CFM quality — results grid</title>
<style>
  body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:0;padding:20px;color:#1a1a1a;background:#fff;}
  h1{font-size:20px;font-weight:500;margin:0 0 14px;}
  #tabs{display:flex;gap:4px;border-bottom:1px solid #e2e2e2;margin-bottom:14px;flex-wrap:wrap;}
  #tabs button{height:34px;border:none;border-bottom:2px solid transparent;background:none;color:#666;padding:0 12px;cursor:pointer;font-size:14px;}
  #tabs button.on{color:#111;font-weight:500;border-bottom-color:#185FA5;}
  .controls{display:flex;align-items:center;gap:14px;margin-bottom:16px;flex-wrap:wrap;font-size:13px;color:#555;}
  select{height:32px;font-size:13px;padding:0 6px;}
  .toggle{height:32px;border:1px solid #bbb;background:#fff;padding:0 12px;cursor:pointer;border-radius:6px;font-size:13px;}
  .toggle.on{background:#E6F1FB;border-color:#85B7EB;color:#0C447C;}
  .legend{margin-left:auto;display:flex;gap:12px;align-items:center;font-size:12px;}
  #grid{display:grid;gap:8px;align-items:stretch;}
  .colh{text-align:center;font-size:12px;color:#555;font-weight:500;padding-bottom:2px;}
  .rowh{display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;}
  .rowh b{font-size:12px;color:#555;font-weight:500;}
  .rowh span{font-size:11px;color:#999;margin-top:2px;}
  .tile{border:1px solid #e2e2e2;border-radius:8px;padding:6px;background:#fff;cursor:pointer;}
  .tile .sub{font-size:11px;color:#888;text-align:center;line-height:1.25;margin-bottom:4px;}
  .tile img{width:100%;height:auto;display:block;border-radius:4px;}
  .empty{border:1px dashed #cfcfcf;border-radius:8px;min-height:120px;display:flex;flex-direction:column;
         align-items:center;justify-content:center;background:#fafafa;color:#999;font-size:12px;gap:4px;text-align:center;padding:6px;}
  #lb{position:fixed;inset:0;background:rgba(0,0,0,.6);display:none;align-items:center;justify-content:center;z-index:50;}
  #lb .box{background:#fff;border-radius:10px;padding:12px;max-width:94vw;max-height:92vh;overflow:auto;}
  #lb .hd{display:flex;justify-content:space-between;align-items:center;gap:16px;margin-bottom:8px;font-size:13px;font-weight:500;}
  #lb img{max-width:88vw;max-height:80vh;display:block;}
  #lb button{border:1px solid #bbb;background:#fff;border-radius:6px;height:28px;width:28px;cursor:pointer;}
</style></head><body>
<h1>CFM quality — results grid</h1>
<div id="tabs"></div>
<div class="controls">
  <label>Metric <select id="metric"></select></label>
  <label>View
    <button class="toggle on" id="v-metric">Per-metric</button>
    <button class="toggle" id="v-comp">Composite</button>
  </label>
  <span class="legend"><span>rows = masking · columns = noising · subtitle = avg masked · flipped fluents/state</span></span>
</div>
<div id="grid"></div>
<div id="lb"><div class="box"><div class="hd"><span id="lb-title"></span><button id="lb-x">&times;</button></div><img id="lb-img" alt=""></div></div>
<script>
const DATA = __DATA__;
const state = {domain: DATA.domains[0], metric: DATA.metrics[0][0], view: "metric"};
const $ = id => document.getElementById(id);

function fmt(v){ return (v===null||v===undefined) ? "?" : v; }

function tileHTML(domain, mask, noise){
  const key = `mask=${mask}__noise=${noise}`;
  const cell = (DATA.cells[domain]||{})[key];
  if(!cell) return `<div class="empty">no run</div>`;
  const st = cell.stats||{};
  const sub = `≈${fmt(st.masked)} masked · ≈${fmt(st.flipped)} flipped`;
  if(cell.status === "single") return `<div class="empty">single model<br>(no conflicts)</div>`;
  const src = state.view === "comp" ? cell.composite : cell.pngs[state.metric];
  if(!src) return `<div class="empty">not generated</div>`;
  return `<div class="tile" onclick="openLB('${src}','${domain} · ${key}')">
      <div class="sub">${sub}</div><img loading="lazy" src="${src}" alt="${key}"></div>`;
}
function render(){
  $("tabs").innerHTML = DATA.domains.map(d =>
    `<button class="${d===state.domain?'on':''}" onclick="setDomain('${d}')">${d}</button>`).join("");
  const cols = DATA.noises, rows = DATA.masks;
  $("grid").style.gridTemplateColumns = `88px repeat(${cols.length}, minmax(0,1fr))`;
  let h = `<div></div>` + cols.map(n => `<div class="colh">noise = ${n}</div>`).join("");
  for(const m of rows){
    const st = ((DATA.cells[state.domain]||{})[`mask=${m}__noise=${cols[0]}`]||{}).stats||{};
    h += `<div class="rowh"><b>mask = ${m}</b><span>≈${fmt(st.masked)} masked/state</span></div>`;
    for(const n of cols) h += tileHTML(state.domain, m, n);
  }
  $("grid").innerHTML = h;
  $("v-metric").className = "toggle" + (state.view==="metric"?" on":"");
  $("v-comp").className = "toggle" + (state.view==="comp"?" on":"");
  $("metric").disabled = state.view==="comp";
}
function setDomain(d){ state.domain=d; render(); }
function openLB(src,title){ $("lb-img").src=src; $("lb-title").textContent=title; $("lb").style.display="flex"; }
$("metric").innerHTML = DATA.metrics.map(([k,l])=>`<option value="${k}">${l}</option>`).join("");
$("metric").onchange = e => { state.metric=e.target.value; render(); };
$("v-metric").onclick = ()=>{ state.view="metric"; render(); };
$("v-comp").onclick = ()=>{ state.view="comp"; render(); };
$("lb-x").onclick = ()=> $("lb").style.display="none";
$("lb").onclick = e => { if(e.target.id==="lb") $("lb").style.display="none"; };
document.addEventListener("keydown", e => { if(e.key==="Escape") $("lb").style.display="none"; });
render();
</script></body></html>
"""


if __name__ == "__main__":
    main()
