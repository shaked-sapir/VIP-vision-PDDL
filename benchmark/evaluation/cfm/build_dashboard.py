"""Build the VIP results dashboard — simulation + image — as one dark HTML page.

Driven entirely by ``dashboard_config.yaml`` (next to this file). The page has a
Simulation / Image toggle; within each, a metric nav switches every view.

Simulation section (per domain: a mask x noise grid-search):
  - a colour-coded summary heatmap (Best CFM / Last CFM toggle), columns grouped
    by p_mask with p_n sub-columns and white separators between mask blocks;
  - a data-corruption table (avg fluents/state): a leading "masked" column per
    p_mask block, then flipped values coloured per p_n;
  - stacked charts: one row per (p_mask, p_n) config, all domains across, using
    the shared-x plot copies so every cell shares an x-axis per domain.

Image section (single experiment per domain):
  - a summary heatmap row over domains (Best / Last), plus a per-domain chart row.

Numbers are derived, not guessed:
  best/last CFM  -> max/min and final of the padded-mean trend (per metric);
  masked/state   -> avg predicates per line in .masking_info;
  flipped/state  -> noise_ratio * unmasked-candidate pool (the percentage noiser).

Usage:
    python -m benchmark.evaluation.cfm.build_dashboard          # HTML only (fast)
    python -m benchmark.evaluation.cfm.build_dashboard --regen-plots   # + shared-x PNGs
        [--config PATH] [--refresh-stats] [--domains ...]

By default the shared-x trend PNGs are reused (only the HTML is rebuilt). Pass
--regen-plots to (re)generate them; that writes just the per-metric trend PNGs
the page uses, not the full plot suite.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import statistics
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import yaml

from benchmark.evaluation.cfm.cfm_quality_analysis import (
    compute_padded_trend,
    find_instance_dirs,
    get_max_solution_index,
    load_cfm_metrics,
    _apply_axis_scaling,
    _BOUNDED_METRICS,
)


def instance_counts(instance_dirs: List[Path]) -> List[int]:
    """[contributing, total]: instances with >= 1 CFM, and the total found."""
    contrib = sum(1 for d in instance_dirs if load_cfm_metrics(d))
    return [contrib, len(instance_dirs)]

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CONFIG = Path(__file__).resolve().parent / "dashboard_config.yaml"

SHARED_SUBDIR = "CFM_quality_shared"   # shared-x copies for the sim charts
RAW_SUBDIR = "CFM_quality"             # raw own-x plots (used by the image section)
CELL_PATTERN = re.compile(r"mask=([0-9.]+)__noise=([0-9.]+)")
_STATS_FILENAME = "_grid_fluent_stats.json"
INVERT_METRICS = {"fluent_patch_count"}  # lower is better


# ── Cell discovery ─────────────────────────────────────────────────────────

def parse_cell(dir_name: str) -> Optional[tuple[str, str]]:
    """Return (p_mask, p_noise) as strings parsed from a folder name, or None."""
    m = CELL_PATTERN.search(dir_name)
    return (m.group(1), m.group(2)) if m else None


def find_grid_cells(domain_dir: Path, prefix: Optional[str]) -> List[Path]:
    """Return a domain's grid-cell dirs, sorted by (p_mask, p_noise)."""
    cells = [
        c for c in domain_dir.iterdir()
        if c.is_dir()
        and (prefix is None or c.name.startswith(prefix))
        and parse_cell(c.name) is not None
    ]
    return sorted(cells, key=lambda p: tuple(float(x) for x in parse_cell(p.name)))


# ── Metric best/last ───────────────────────────────────────────────────────

def _metric_source(key: str) -> str:
    return "conflict_free_solutions_log" if key == "fluent_patch_count" else "all_solutions_metrics"


def metric_stats(instance_dirs: List[Path], key: str) -> Optional[Dict[str, float]]:
    """Best/last value (and their CFM index) of a metric's padded-mean trend."""
    sol, means, _stds, _n = compute_padded_trend(instance_dirs, key, source=_metric_source(key))
    if len(sol) == 0:
        return None
    means = [float(v) for v in means]
    sol = [int(v) for v in sol]
    if key in INVERT_METRICS:
        bi = min(range(len(means)), key=lambda i: means[i])
    else:
        bi = max(range(len(means)), key=lambda i: means[i])
    return {"best": round(means[bi], 3), "best_i": sol[bi],
            "last": round(means[-1], 3), "last_i": sol[-1]}


def all_metric_stats(instance_dirs: List[Path], metrics: List[dict]) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for m in metrics:
        s = metric_stats(instance_dirs, m["key"])
        if s is not None:
            out[m["key"]] = s
    return out


# ── Fluent statistics (masked / flipped per state) ─────────────────────────

def _masked_per_state(masking_info_path: Path) -> List[int]:
    lines = [ln.strip() for ln in masking_info_path.read_text().splitlines()]
    return [0 if not ln else len(ln.split(", ")) for ln in lines[1:]]  # skip initial state


def _candidate_counts(cell: Path) -> List[int]:
    """Unmasked-candidate count per state (t>=1): total grounded fluents minus
    the masked count.

    Deliberately avoids applying masking (``load_masked_observation``): we only
    parse and ground the noisy trajectory to get the total fluent universe, then
    subtract the masked count read straight from the ``.masking_info`` text. This
    is robust for any p_mask (the mask-application path fails on masked cells).
    """
    from pddl_plus_parser.lisp_parsers import DomainParser, TrajectoryParser

    from src.utils.pddl_state import (
        ground_observation_completely,
        get_state_grounded_predicates,
    )

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
            obs = TrajectoryParser(partial_domain=domain).parse_trajectory(tf)
            grounded = ground_observation_completely(domain, obs)
            states = [grounded.components[0].previous_state] + [
                c.next_state for c in grounded.components
            ]
            masked = _masked_per_state(mf)  # one entry per state t>=1
            for i in range(1, len(states)):
                total = len(get_state_grounded_predicates(states[i]))
                mc = masked[i - 1] if i - 1 < len(masked) else 0
                counts.append(max(0, total - mc))
    return counts


def cell_fluent_stats(cell: Path, noise_ratio: float) -> Dict[str, Optional[float]]:
    """Avg masked and flipped fluents per state for one cell."""
    masked = [c for mf in sorted(cell.glob("testing/*/original_observations/*.masking_info"))
              for c in _masked_per_state(mf)]
    out: Dict[str, Optional[float]] = {
        "masked": round(statistics.mean(masked), 1) if masked else None,
        "flipped": None,
    }
    try:
        cand = _candidate_counts(cell)
    except Exception as exc:  # noqa: BLE001 — degrade gracefully (e.g. no PDDL parser)
        print(f"    [stats] candidates unavailable for {cell.name}: {exc}")
        cand = []
    if cand:
        # Noise flips noise_ratio of the *unmasked* pool per state; report the
        # continuous average so the dependence on masking stays visible (rounding
        # each state to an integer collapses small differences at low noise).
        out["flipped"] = round(noise_ratio * statistics.mean(cand), 1)
    return out


def load_or_compute_stats(domain_dir: Path, cells: List[Path], refresh: bool) -> Dict[str, dict]:
    cache = domain_dir / _STATS_FILENAME
    if cache.exists() and not refresh:
        data = json.loads(cache.read_text())
        if not any(v and v.get("flipped") is None for v in data.values()):
            return data
    stats: Dict[str, dict] = {}
    for cell in cells:
        m, n = parse_cell(cell.name)
        print(f"    fluent stats: {cell.name}")
        stats[f"{m}_{n}"] = cell_fluent_stats(cell, float(n))
    cache.write_text(json.dumps(stats, indent=2))
    return stats


# ── Shared-x plot copies (simulation charts) ───────────────────────────────

def _plot_trend_thumb(solution_ids, means, stds, metric_key: str,
                      output_path: Path, x_max: Optional[int]) -> None:
    """A lean grid thumbnail: mean line + std band, tick numbers kept, but no
    title, axis labels, legend or footnote (the grid supplies those via the
    domain/config labels and the corner axis key).
    """
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    if len(solution_ids) == 1:
        # No conflicts: one CFM only. Show a bold point at its value + a note,
        # keeping the same axes as every other cell.
        val = float(means[0])
        ax.plot(solution_ids, means, "o", color="#1B6DB5", markersize=8, zorder=5)
        label = f"{val:.0f}" if metric_key == "fluent_patch_count" else f"{val:.2f}"
        ax.annotate(label, xy=(0, val), xytext=(8, -3), textcoords="offset points",
                    ha="left", va="top", fontsize=8, fontweight="bold", color="#1B6DB5")
        ax.text(0.5, 0.5, "single model\n(no conflicts)", transform=ax.transAxes,
                ha="center", va="center", color="#888888", fontsize=8)
    else:
        ax.plot(solution_ids, means, color="#1B6DB5", linewidth=1.4)
        lo, hi = means - stds, means + stds
        if metric_key in _BOUNDED_METRICS:
            lo, hi = np.clip(lo, 0.0, 1.0), np.clip(hi, 0.0, 1.0)
        ax.fill_between(solution_ids, lo, hi, color="#1B6DB5", alpha=0.2)
    _apply_axis_scaling(ax, metric_key)          # integer x ticks + fixed y scale
    if x_max and x_max > 0:
        margin = max(0.5, 0.02 * x_max)
        ax.set_xlim(-margin, x_max + margin)
    ax.tick_params(labelsize=7)
    fig.tight_layout(pad=0.3)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _generate_shared_trend_plots(cell: Path, x_max: int, metrics: List[dict]) -> None:
    """Write only the lean per-metric trend PNGs (shared x) the dashboard uses."""
    out_dir = cell / "evaluation_results" / SHARED_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    instance_dirs = find_instance_dirs(cell / "testing")
    for m in metrics:
        key = m["key"]
        sol, means, stds, _n = compute_padded_trend(instance_dirs, key, source=_metric_source(key))
        if len(sol) == 0:
            continue
        _plot_trend_thumb(sol, means, stds, key, out_dir / f"{key}_trend.png", x_max)


def regenerate_shared_x_plots(cells: List[Path], metrics: List[dict]) -> None:
    """Write shared-x trend PNGs (CFM_quality_shared/) for a domain's cells."""
    per_max = {c: (get_max_solution_index(c) or 0) for c in cells}
    domain_max = max(per_max.values(), default=0)
    if domain_max == 0:
        return
    # Includes single-CFM cells (noise=0): they get a one-point plot with the
    # same shared x-axis and a "single model" note (see _plot_trend_thumb).
    for cell in cells:
        _generate_shared_trend_plots(cell, domain_max, metrics)


# ── Data assembly ──────────────────────────────────────────────────────────

def _rel(png: Path, out_dir: Path) -> str:
    return os.path.relpath(png, out_dir).replace(os.sep, "/")


def img_src(png: Path, out_dir: Path, embed: bool) -> str:
    """A relative ``<img src>`` path, or a self-contained base64 data URI when
    ``embed`` is set (so the whole page can be shared as one file)."""
    if embed:
        return "data:image/png;base64," + base64.b64encode(png.read_bytes()).decode("ascii")
    return _rel(png, out_dir)


def build_sim_data(cfg: dict, root: Path, out_dir: Path, metrics: List[dict],
                   regen: bool, refresh: bool, embed: bool) -> dict:
    prefixes = cfg["simulation"].get("prefix", {})
    masks, noises = set(), set()
    cells_out: Dict[str, dict] = {}

    for domain in cfg["domains"]:
        domain_dir = root / domain
        if not domain_dir.is_dir():
            continue
        cells = find_grid_cells(domain_dir, prefixes.get(domain))
        if not cells:
            continue
        print(f"  [sim] {domain}: {len(cells)} cells")
        if regen:
            print("    regenerating shared-x trend plots...")
            regenerate_shared_x_plots(cells, metrics)
        fluent = load_or_compute_stats(domain_dir, cells, refresh)

        cells_out[domain] = {}
        for cell in cells:
            m, n = parse_cell(cell.name)
            masks.add(m); noises.add(n)
            key = f"{m}_{n}"
            instance_dirs = find_instance_dirs(cell / "testing")
            stats = all_metric_stats(instance_dirs, metrics)
            max_id = get_max_solution_index(cell)
            status = "missing" if max_id is None else ("single" if max_id == 0 else "ok")
            shared = cell / "evaluation_results" / SHARED_SUBDIR
            pngs = {mk["key"]: img_src(shared / f'{mk["key"]}_trend.png', out_dir, embed)
                    for mk in metrics if (shared / f'{mk["key"]}_trend.png').exists()}
            fl = fluent.get(key, {})
            cells_out[domain][key] = {
                "status": status, "stats": stats, "pngs": pngs,
                "masked": fl.get("masked"), "flipped": fl.get("flipped"),
                "n": instance_counts(instance_dirs),
            }

    return {
        "cells": cells_out,
        "masks": sorted(masks, key=float),
        "noises": sorted(noises, key=float),
    }


def regenerate_image_thumbs(exp: Path, metrics: List[dict]) -> None:
    """Write lean trend thumbnails for one image experiment (own x-axis)."""
    out_dir = exp / "evaluation_results" / SHARED_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    instance_dirs = find_instance_dirs(exp / "testing")
    for m in metrics:
        key = m["key"]
        sol, means, stds, _n = compute_padded_trend(instance_dirs, key, source=_metric_source(key))
        if len(sol) == 0:
            continue
        _plot_trend_thumb(sol, means, stds, key, out_dir / f"{key}_trend.png", None)  # own x


def build_image_data(cfg: dict, root: Path, out_dir: Path, metrics: List[dict],
                     regen: bool, embed: bool) -> dict:
    dirs = cfg.get("image", {}).get("experiment_dir", {})
    out: Dict[str, dict] = {}
    for domain in cfg["domains"]:
        rel = dirs.get(domain)
        if not rel:
            continue
        exp = (_PROJECT_ROOT / rel).resolve()
        testing = exp / "testing"
        if not testing.is_dir():
            print(f"  [img] {domain}: SKIP (no testing/ at {rel})")
            continue
        if regen:
            regenerate_image_thumbs(exp, metrics)
        instance_dirs = find_instance_dirs(testing)
        stats = all_metric_stats(instance_dirs, metrics)
        cfmq = exp / "evaluation_results" / SHARED_SUBDIR
        pngs = {mk["key"]: img_src(cfmq / f'{mk["key"]}_trend.png', out_dir, embed)
                for mk in metrics if (cfmq / f'{mk["key"]}_trend.png').exists()}
        out[domain] = {"stats": stats, "pngs": pngs, "n": instance_counts(instance_dirs)}
        print(f"  [img] {domain}: ok")
    return {"domains_data": out}


def build(config_path: Path, regen: bool, refresh: bool,
          domains: Optional[List[str]] = None, embed: bool = False) -> Path:
    cfg = yaml.safe_load(config_path.read_text())["results_dashboard"]
    if domains:
        cfg["domains"] = [d for d in cfg["domains"] if d in domains]
    root = (_PROJECT_ROOT / cfg["results_root"]).resolve()
    out_html = (_PROJECT_ROOT / cfg["output_html"]).resolve()
    out_dir = out_html.parent
    # Embedded export goes to a separate self-contained file so the live
    # relative-path page is left intact.
    if embed:
        out_html = out_html.with_name(out_html.stem + "_standalone" + out_html.suffix)
    metrics = cfg["metrics"]

    print("Assembling simulation data...")
    sim = build_sim_data(cfg, root, out_dir, metrics, regen, refresh, embed)
    print("Assembling image data...")
    img = build_image_data(cfg, root, out_dir, metrics, regen, embed)

    payload = {
        "domains": cfg["domains"],
        "metrics": [{"key": m["key"], "label": m["label"],
                     "invert": m["key"] in INVERT_METRICS} for m in metrics],
        "sim": sim,
        "image": img,
    }
    out_html.write_text(_HTML.replace("__DATA__", json.dumps(payload)))
    size_mb = out_html.stat().st_size / 1e6
    print(f"\nDashboard written to: {out_html}  ({size_mb:.1f} MB"
          f"{', self-contained' if embed else ', relative image paths'})")
    return out_html


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the VIP results dashboard.")
    ap.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    ap.add_argument("--regen-plots", action="store_true",
                    help="Regenerate the shared-x trend PNGs (CFM_quality_shared/). "
                         "Off by default — existing copies are reused.")
    ap.add_argument("--refresh-stats", action="store_true",
                    help="Recompute cached per-domain fluent statistics.")
    ap.add_argument("--domains", nargs="*", default=None,
                    help="Restrict to these domains (default: all in the config).")
    ap.add_argument("--embed", action="store_true",
                    help="Export a self-contained <name>_standalone.html with every chart "
                         "embedded as base64 (shareable as a single file). Larger file.")
    args = ap.parse_args()
    build(args.config, args.regen_plots, args.refresh_stats, args.domains, args.embed)


_HTML = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>VIP — results</title>
<style>
  body{background:#16181d;color:#e8e8e8;font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;margin:0;padding:18px 22px;}
  h1{font-size:18px;font-weight:500;margin:0;}
  button{font-family:inherit;}
  #metricnav button,.et,.seg{background:#1f232b;border:1px solid #333842;color:#cfd3da;padding:6px 11px;border-radius:6px;cursor:pointer;font-size:13px;}
  #metricnav button.on{background:#1d4ed8;border-color:#3b6fe0;color:#fff;}
  .et.on{background:#33384a;border-color:#4a5a86;color:#dce6ff;} .seg.on{background:#243049;border-color:#3b6fe0;color:#dce6ff;}
  .card{background:#1f232b;border:1px solid #333842;border-radius:10px;padding:12px;margin-bottom:14px;}
  .card h4{margin:0 0 10px;font-size:13px;font-weight:500;color:#c3c8d0;display:flex;align-items:center;gap:10px;}
  .twrap{overflow-x:auto;}
  table.heat{border-collapse:collapse;font-size:12px;white-space:nowrap;}
  table.heat th{color:#9aa0a8;font-weight:500;padding:4px 8px;text-align:center;font-size:11px;}
  table.heat th.mk{color:#8fa6c6;}
  table.heat td.dom{color:#c3c8d0;text-align:left;padding:4px 12px 4px 4px;font-size:12.5px;}
  table.heat td.hc{width:50px;text-align:center;color:#f2f4f7;padding:7px 0;}
  table.heat td.mkv{width:56px;text-align:center;color:#aab6c6;padding:7px 0;}
  table.heat td.na{width:50px;text-align:center;color:#5b616b;}
  .msep{border-left:3px solid #ffffff !important;}
  td.hl{box-shadow:inset 0 0 0 2px #ef4444;}
  .domhead,.colhead{text-align:center;font-size:12px;color:#c3c8d0;font-weight:500;align-self:center;}
  .cfglab{font-size:10.5px;color:#c8cdd6;border-left:2px solid #3b6fe0;padding-left:6px;line-height:1.3;align-self:center;}
  .cfglab span{display:block;color:#8b929c;font-size:10.5px;}
  #metricnav{position:sticky;top:0;background:#16181d;z-index:20;}
  .allscroll{max-height:74vh;overflow:auto;}
  .allhead{position:sticky;top:0;z-index:3;background:#1f232b;padding-bottom:6px;}
  .rowlab{font-size:12px;color:#c3c8d0;font-weight:500;border-left:2px solid #3b6fe0;padding-left:6px;align-self:center;}
  .keywrap{display:flex;align-items:center;justify-content:center;}
  .cell{border:1px solid #333842;border-radius:8px;padding:3px;background:#1b1e25;min-height:120px;}
  .cell img{width:100%;height:auto;display:block;cursor:zoom-in;background:#fff;border-radius:3px;}
  .cell .ph{font-size:10.5px;color:#6b7280;text-align:center;padding:20px 4px;}
  .ncap{font-size:10px;color:#8b929c;text-align:center;margin-top:3px;}
  .empty{border:1px dashed #3a4150;border-radius:8px;background:#1a1d24;color:#6b7280;font-size:11px;display:flex;align-items:center;justify-content:center;text-align:center;min-height:110px;padding:6px;}
  .rowsep{grid-column:1/-1;border-top:1px solid #333842;margin:9px 0;}
  .colsep{background:#333842;}
  .grow{display:flex;gap:12px;align-items:stretch;flex-wrap:wrap;margin-bottom:14px;}
  .grow>.card{margin:0;}
  .gcorr{flex:1 1 380px;min-width:0;}
  .gex{flex:0 0 320px;display:flex;flex-direction:column;}
  .gexfig{flex:1 1 auto;min-height:180px;}
  .gexp{flex:0 0 250px;}
  .gtext p{margin:5px 0;font-size:12px;color:#c3c8d0;line-height:1.5;}
  .gtext b{color:#dce6ff;font-weight:500;}
  .gtext .ex{color:#9aa0a8;} .gtext .ex b{color:#7aa2ff;}
  .dgrid{max-width:1000px;}
  .dgrid .cell img{max-height:210px;object-fit:contain;}
  .ctab{background:none;border:1px solid transparent;color:#8b929c;padding:4px 11px;cursor:pointer;font-size:12.5px;border-radius:6px;}
  .ctab.on{color:#fff;background:#243049;border-color:#3b6fe0;}
  .chdr{display:flex;align-items:center;gap:6px;border-bottom:1px solid #2b303a;padding-bottom:8px;margin-bottom:12px;flex-wrap:wrap;}
  .legkey{margin-left:auto;font-size:11px;color:#9aa0a8;display:flex;gap:12px;}
  .legkey i{display:inline-block;vertical-align:middle;}
  .cap{font-size:11px;color:#9aa0a8;text-align:center;margin-bottom:3px;}
  .note{font-size:11px;color:#7d828b;margin-top:8px;}
  #lb{position:fixed;inset:0;background:rgba(0,0,0,.88);display:none;align-items:center;justify-content:center;z-index:50;cursor:zoom-out;}
  #lb img{max-width:96vw;max-height:96vh;background:#fff;border-radius:4px;}
</style></head><body>
<div style="display:flex;align-items:center;gap:12px;margin-bottom:12px;">
  <h1>VIP — results</h1><span id="modelabel" style="font-size:12px;color:#9aa0a8;"></span>
  <div style="margin-left:auto;display:flex;gap:6px;background:#1f232b;border:1px solid #333842;border-radius:8px;padding:3px;">
    <button class="et on" data-et="sim">Simulation</button><button class="et" data-et="img">Image</button></div>
</div>
<div id="metricnav" style="display:flex;flex-wrap:wrap;gap:6px;border-bottom:1px solid #333842;padding-bottom:10px;margin-bottom:14px;"></div>
<div id="view"></div>
<div id="lb" onclick="this.style.display='none'"><div style="text-align:center;"><div id="lbcap" style="color:#cbd2dc;font-size:13px;margin-bottom:8px;"></div><img id="lbimg" src="" alt=""></div></div>
<script>
const DATA=__DATA__;
const MASKS=DATA.sim.masks, NOISES=DATA.sim.noises, DOMAINS=DATA.domains, METRICS=DATA.metrics;
const S={et:"sim",metric:METRICS[0].key,stat:"last",tab:"all"};
const $=id=>document.getElementById(id);
const meta=k=>METRICS.find(m=>m.key===k)||{};
const CONFIGS=[]; for(const m of MASKS)for(const n of NOISES)CONFIGS.push([m,n]);
function heat(v){const t=Math.max(0,Math.min(1,v));const lo=[124,52,52],mid=[132,110,52],hi=[46,110,64];const c=t<.5?lo.map((x,i)=>Math.round(x+(mid[i]-x)*t*2)):mid.map((x,i)=>Math.round(x+(hi[i]-x)*(t-.5)*2));return `rgb(${c})`;}
function amber(t){t=Math.max(0,Math.min(1,t));const lo=[38,41,48],hi=[205,124,38];return `rgb(${lo.map((x,i)=>Math.round(x+(hi[i]-x)*t))})`;}
function cellOf(d,m,n){return ((DATA.sim.cells[d]||{})[`${m}_${n}`])||null;}

function simHeat(){
  const ml=meta(S.metric).label;
  let h1=`<tr><th></th>`+MASKS.map((m,gi)=>`<th colspan="${NOISES.length}"${gi>0?' class="msep"':''}>p_mask = ${m}</th>`).join("")+`</tr>`;
  let h2=`<tr><th></th>`+MASKS.map((m,gi)=>NOISES.map((n,ni)=>`<th${(gi>0&&ni===0)?' class="msep"':''}>p_n = ${n}</th>`).join("")).join("")+`</tr>`;
  let body="";
  for(const d of DOMAINS){body+=`<tr><td class="dom">${d}</td>`;for(const[m,n]of CONFIGS){const sep=(n===NOISES[0]&&m!==MASKS[0])?' msep':'';const c=cellOf(d,m,n),s=c&&c.stats[S.metric];if(!s){body+=`<td class="na${sep}">–</td>`;continue;}body+=`<td class="hc${sep}" style="background:${heat(s[S.stat])}">${s[S.stat].toFixed(2)}</td>`;}body+=`</tr>`;}
  return `<div class="card"><h4>${ml} — summary heatmap<span style="margin-left:auto;display:flex;gap:6px;"><button class="seg ${S.stat==='best'?'on':''}" onclick="setStat('best')">Best CFM</button><button class="seg ${S.stat==='last'?'on':''}" onclick="setStat('last')">Last CFM</button></span></h4><div class="twrap"><table class="heat"><thead>${h1}${h2}</thead><tbody>${body}</tbody></table></div><div class="note">fixed colour scale · 0.0 red · 0.5 amber · 1.0 green (same for all metrics; fluent patch count is a raw count, so its colours are only indicative) · “–” = no data</div></div>`;
}
function corruption(){
  const fmax=Math.max(1,...DOMAINS.flatMap(d=>CONFIGS.map(([m,n])=>{const c=cellOf(d,m,n);return c&&c.flipped!=null?c.flipped:0;})));
  let h1=`<tr><th></th>`+MASKS.map((m,gi)=>`<th colspan="${NOISES.length+1}"${gi>0?' class="msep"':''}>p_mask = ${m}</th>`).join("")+`</tr>`;
  let h2=`<tr><th></th>`+MASKS.map((m,gi)=>`<th class="mk${gi>0?' msep':''}">masked</th>`+NOISES.map(n=>`<th>p_n = ${n}</th>`).join("")).join("")+`</tr>`;
  let body="";
  for(const d of DOMAINS){body+=`<tr><td class="dom">${d}</td>`;
    for(let gi=0;gi<MASKS.length;gi++){const m=MASKS[gi];const cm=cellOf(d,m,NOISES[0])||cellOf(d,m,NOISES[1]);const mv=cm&&cm.masked!=null?cm.masked:"–";
      body+=`<td class="mkv${gi>0?' msep':''}" data-mk="${d}|${m}">${mv}</td>`;
      for(const n of NOISES){const c=cellOf(d,m,n);const fv=c&&c.flipped!=null?c.flipped:null;if(fv==null){body+=`<td class="na" data-fc="${d}|${m}|${n}">–</td>`;continue;}body+=`<td class="hc" data-fc="${d}|${m}|${n}" style="background:${amber(fv/fmax)}">${fv}</td>`;}}
    body+=`</tr>`;}
  return `<div class="card gcorr"><h4>Data corruption — avg fluents / state <span style="color:#7d828b;font-weight:400;font-size:11px;">(masked = leading column per p_mask block; flipped coloured per p_n)</span></h4><div class="twrap"><table class="heat"><thead>${h1}${h2}</thead><tbody>${body}</tbody></table></div></div>`;
}
function guideSVG(){const w=300,h=210,lp=32,bp=26,tp=10,rp=12,MAX=24;
  const sd=s=>{let x=Math.sin(s)*1e4;return x-Math.floor(x);};
  const X=k=>lp+(w-lp-rp)*k/MAX,Y=v=>h-bp-(h-tp-bp)*v;
  const insts=[{stop:24,s:1},{stop:12,s:2},{stop:6,s:3}];
  const series=inst=>{const a=[];let last=0;for(let k=0;k<=MAX;k++){if(k<=inst.stop)last=Math.max(.05,Math.min(1,0.45+0.5*Math.pow(k/MAX,0.5)+(sd(inst.s*9+k)-.5)*.05));a.push(last);}return a;};
  const Sr=insts.map(series),mn=[];for(let k=0;k<=MAX;k++)mn.push(Sr.reduce((t,a)=>t+a[k],0)/Sr.length);
  const path=a=>{let p="";a.forEach((v,k)=>p+=(k?"L":"M")+X(k)+","+Y(v));return p;};
  const YT=[0,.5,1],XT=[0,5,10,15,20],ex={x:X(5),y:Y(mn[5])};
  let inst="";Sr.forEach((a,i)=>{inst+=`<path d="${path(a)}" fill="none" stroke="#6f9fe0" stroke-width="1" opacity="0.42"/><circle cx="${X(insts[i].stop)}" cy="${Y(a[insts[i].stop])}" r="2.3" fill="#6f9fe0" opacity="0.6"/>`;});
  return `<svg viewBox="0 0 ${w} ${h}" style="width:100%;height:100%;display:block;background:#12151b;border:1px solid #2b303a;border-radius:6px;">
    <line x1="${lp}" y1="${tp}" x2="${lp}" y2="${h-bp}" stroke="#8a93a3"/><line x1="${lp}" y1="${h-bp}" x2="${w-rp}" y2="${h-bp}" stroke="#8a93a3"/>
    ${YT.map(t=>`<text x="${lp-4}" y="${Y(t)+3}" font-size="8.5" text-anchor="end" fill="#9aa0a8">${t.toFixed(1)}</text>`).join("")}
    ${XT.map(v=>`<text x="${X(v)}" y="${h-bp+11}" font-size="8.5" text-anchor="middle" fill="#9aa0a8">${v}</text>`).join("")}
    <text x="9" y="${(h-bp)/2}" font-size="9" fill="#aeb6c2" transform="rotate(-90 9 ${(h-bp)/2})" text-anchor="middle">metric value</text>
    <text x="${lp+(w-lp-rp)/2}" y="${h-4}" font-size="9" fill="#aeb6c2" text-anchor="middle">CFM solution index</text>
    ${inst}<path d="${path(mn)}" fill="none" stroke="#1B6DB5" stroke-width="2.3"/>
    <line x1="${lp}" y1="${ex.y}" x2="${ex.x}" y2="${ex.y}" stroke="#7aa2ff" stroke-dasharray="2 2" stroke-width=".9"/><line x1="${ex.x}" y1="${h-bp}" x2="${ex.x}" y2="${ex.y}" stroke="#7aa2ff" stroke-dasharray="2 2" stroke-width=".9"/>
    <circle cx="${ex.x}" cy="${ex.y}" r="5.5" fill="none" stroke="#7aa2ff" stroke-width="1.7"/><circle cx="${ex.x}" cy="${ex.y}" r="2.2" fill="#7aa2ff"/><text x="${ex.x+8}" y="${ex.y-7}" font-size="9" fill="#9db6ff">(5, 0.80)</text></svg>`;
}
function exampleCard(){return `<div class="card gex"><h4>Example plot</h4><div class="gexfig">${guideSVG()}</div></div>`;}
function explainCard(){return `<div class="card gexp"><h4>How to read the plots</h4><div class="gtext">`+
    `<p><b>x — CFM solution index.</b> The k-th conflict-free model the search produced (0 = base model).</p>`+
    `<p><b>y — metric value.</b> Mean over the test instances; the shaded band is ±1 std.</p>`+
    `<p><b>“padded” mean.</b> Each instance is forward-filled — once it stops finding new models its last value is carried forward, so every instance contributes at every index (the mean flattens on the right).</p>`+
    `<p class="ex"><b>Example:</b> index 5 → 0.80 = the mean over instances of each one's value at its 6th CFM (or its last value, if it found fewer).</p>`+
  `</div></div>`;}
function guideRow(){return `<div class="grow">${corruption()}${exampleCard()}${explainCard()}</div>`;}
function ncap(c){return (c&&c.n)?`<div class="ncap">instances: ${c.n[0]}(${c.n[1]})</div>`:"";}
function chartCell(c,cap,hk){
  const hov=hk?` onmouseenter="hlCorr('${hk}',true)" onmouseleave="hlCorr('${hk}',false)"`:"";
  if(c&&c.pngs[S.metric]) return `<div${hov}><div class="cell"><img loading="lazy" src="${c.pngs[S.metric]}" onclick="zoom('${c.pngs[S.metric]}','${cap}')"></div>${ncap(c)}</div>`;
  if(c&&c.status==="single") return `<div${hov}><div class="empty">single model<br>(no conflicts)</div>${ncap(c)}</div>`;
  return `<div><div class="empty">no data</div>${ncap(c)}</div>`;
}
function hlCorr(key,on){const p=key.split("|");
  document.querySelectorAll(`[data-mk="${p[0]}|${p[1]}"],[data-fc="${p[0]}|${p[1]}|${p[2]}"]`).forEach(e=>e.classList.toggle("hl",on));
}
function allView(){
  const cols=`84px 1fr`+` 1px 1fr`.repeat(DOMAINS.length-1);
  let head=`<div class="allhead" style="display:grid;grid-template-columns:${cols};gap:7px;"><div></div>`+DOMAINS.map((d,i)=>(i?`<div class="colsep"></div>`:"")+`<div class="domhead">${d}</div>`).join("")+`</div>`;
  let body=`<div style="display:grid;grid-template-columns:${cols};gap:7px;">`;
  CONFIGS.forEach(([m,n],ri)=>{
    body+=`<div class="cfglab">p_mask = ${m}<span>p_n = ${n}</span></div>`+DOMAINS.map((d,i)=>(i?`<div class="colsep"></div>`:"")+chartCell(cellOf(d,m,n),`${d} · p_mask=${m} p_n=${n} · ${meta(S.metric).label}`)).join("");
    if(ri<CONFIGS.length-1)body+=`<div class="rowsep"></div>`;
  });
  body+="</div>";
  return `<div class="allscroll">${head}${body}</div>`;
}
function domainView(d){
  let h=`<div class="dgrid" style="display:grid;grid-template-columns:70px repeat(${MASKS.length},1fr);gap:7px;">`;
  h+=`<div></div>`+MASKS.map(m=>`<div class="colhead">p_mask = ${m}</div>`).join("");
  for(const n of NOISES){h+=`<div class="rowlab">p_n = ${n}</div>`;
    for(const m of MASKS)h+=chartCell(cellOf(d,m,n),`${d} · p_mask=${m} p_n=${n} · ${meta(S.metric).label}`,`${d}|${m}|${n}`);}
  return h+"</div>";
}
function chartsBody(){return S.tab==="all"?allView():domainView(S.tab);}
function simCharts(){
  const tabs=["all",...DOMAINS];
  const tabbar=tabs.map(t=>`<button class="ctab ${t===S.tab?'on':''}" data-tab="${t}" onclick="setTab('${t}')">${t}</button>`).join("");
  const legend=`<span class="legkey"><span><i style="width:14px;height:2px;background:#1B6DB5;"></i> mean</span><span><i style="width:14px;height:9px;background:#B5D4F4;border-radius:2px;"></i> ±1 std</span></span>`;
  return `<div class="card"><div class="chdr">${tabbar}${legend}</div><div id="chgrid">${chartsBody()}</div></div>`;
}
function imgView(){
  const dd=DATA.image.domains_data||{},ml=meta(S.metric).label;
  const have=DOMAINS.filter(d=>dd[d]);
  if(!have.length)return `<div class="card"><h4>Image</h4><div class="note">No image experiments found. Fill in image.experiment_dir paths in dashboard_config.yaml.</div></div>`;
  const head=`<tr><th></th>`+have.map(d=>`<th>${d}</th>`).join("")+`</tr>`;
  const row=(label,key)=>`<tr><td class="dom">${label}</td>`+have.map(d=>{const s=dd[d].stats[S.metric];return s?`<td class="hc" style="min-width:80px;background:${heat(s[key])}">${s[key].toFixed(2)}</td>`:`<td class="na">–</td>`;}).join("")+`</tr>`;
  const table=`<div class="card"><h4>${ml} — image experiments</h4><div class="twrap"><table class="heat"><thead>${head}</thead><tbody>${row("best CFM","best")}${row("last CFM","last")}</tbody></table></div></div>`;
  const guide=`<div class="grow">${exampleCard()}${explainCard()}</div>`;
  let charts=`<div class="card"><h4>${ml} — per domain (single config)<span class="legkey"><span><i style="width:14px;height:2px;background:#1B6DB5;"></i> mean</span><span><i style="width:14px;height:9px;background:#B5D4F4;border-radius:2px;"></i> ±1 std</span></span></h4>`;
  charts+=`<div style="display:grid;grid-template-columns:repeat(${have.length},1fr);gap:7px;margin-bottom:6px;">`+have.map(d=>`<div class="domhead">${d}</div>`).join("")+`</div>`;
  charts+=`<div style="display:grid;grid-template-columns:repeat(${have.length},1fr);gap:7px;">`+have.map(d=>{const p=dd[d].pngs[S.metric],cap=`${d} · ${ml}`,nc=dd[d].n?`<div class="ncap">instances: ${dd[d].n[0]}(${dd[d].n[1]})</div>`:"";return `<div>${p?`<div class="cell"><img loading="lazy" src="${p}" onclick="zoom('${p}','${cap}')"></div>`:`<div class="empty">no plot</div>`}${nc}</div>`;}).join("")+`</div></div>`;
  return table+guide+charts;
}
function render(){
  $("modelabel").textContent=S.et==="sim"?"mask × noise grid per domain":"single config per domain";
  $("metricnav").innerHTML=METRICS.map(m=>`<button class="${m.key===S.metric?'on':''}" onclick="setMetric('${m.key}')">${m.label}</button>`).join("");
  document.querySelectorAll(".et[data-et]").forEach(b=>b.classList.toggle("on",b.dataset.et===S.et));
  $("view").innerHTML=S.et==="sim"?simHeat()+guideRow()+simCharts():imgView();
}
function setMetric(k){S.metric=k;render();}
function setStat(s){S.stat=s;render();}
function renderCharts(){
  const g=$("chgrid");
  if(!g){render();return;}
  g.innerHTML=chartsBody();
  document.querySelectorAll(".ctab[data-tab]").forEach(b=>b.classList.toggle("on",b.dataset.tab===S.tab));
}
function setTab(t){S.tab=t;renderCharts();}
function zoom(src,cap){$("lbimg").src=src;$("lbcap").textContent=cap||"";$("lb").style.display="flex";}
document.querySelectorAll(".et[data-et]").forEach(b=>b.onclick=()=>{S.et=b.dataset.et;render();});
document.addEventListener("keydown",e=>{if(e.key==="Escape")$("lb").style.display="none";});
render();
</script></body></html>
"""


if __name__ == "__main__":
    main()
