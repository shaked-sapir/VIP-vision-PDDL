#!/usr/bin/env python3
"""Generate an interactive HTML viewer from a conflict search trace JSON.

Usage:
    python benchmark/diagnosis/visualize_trace.py search_trace.json [-o tree.html]

Reads a search_trace.json (produced by trace_spurious_effects.py --retrace)
and outputs a self-contained HTML file with:
  - Collapsible tree view of the search
  - "Expand to CFM #N" to show the full path to a conflict-free solution
  - Filter by action + predicate to highlight relevant nodes
  - Click any node card for full details (conflicts, patches, constraints)
"""

import argparse
import json
import sys
from pathlib import Path

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Conflict Search Trace — {title}</title>
<style>
  :root {
    --bg: #1e1e2e;
    --surface: #282838;
    --border: #444;
    --text: #cdd6f4;
    --dim: #6c7086;
    --green: #a6e3a1;
    --orange: #fab387;
    --red: #f38ba8;
    --blue: #89b4fa;
    --yellow: #f9e2af;
    --mauve: #cba6f7;
    --indent: 16px;
    --max-indent: 480px;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
    font-size: 13px;
    background: var(--bg);
    color: var(--text);
    padding: 16px;
  }
  h1 { font-size: 18px; margin-bottom: 4px; color: var(--blue); }
  .meta { color: var(--dim); margin-bottom: 16px; font-size: 12px; }

  /* Controls */
  .controls {
    display: flex; gap: 12px; align-items: center;
    margin-bottom: 8px; flex-wrap: wrap;
    position: sticky; top: 0; z-index: 10;
    background: var(--bg); padding: 8px 0;
    border-bottom: 1px solid var(--border);
  }
  .controls label { color: var(--dim); font-size: 12px; }
  .controls select, .controls input, .controls button {
    background: var(--surface); color: var(--text); border: 1px solid var(--border);
    padding: 4px 8px; border-radius: 4px; font-family: inherit; font-size: 12px;
  }
  .controls button { cursor: pointer; }
  .controls button:hover { border-color: var(--blue); }

  /* Flat tree — each node is a div, indentation via margin-left */
  #tree { padding: 4px 0; transition: margin-right 0.2s; }
  body.sidebar-open #tree { margin-right: 195px; }

  .node-row {
    margin: 3px 0;
    display: flex;
    align-items: flex-start;
  }

  .node-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 6px 10px;
    cursor: pointer;
    display: inline-block;
    transition: border-color 0.15s, opacity 0.15s;
    flex-shrink: 0;
  }
  .node-card:hover { border-color: var(--blue); }
  .node-card.cf { border-left: 3px solid var(--green); }
  .node-card.conflict { border-left: 3px solid var(--orange); }
  .node-card.root { border-left: 3px solid var(--blue); }
  .node-card.highlight { border-color: var(--yellow); box-shadow: 0 0 6px rgba(249,226,175,0.3); }
  .node-card.relevant {
    border: 3px solid #f5c211 !important;
    background: #2e2a10 !important;
    box-shadow: 0 0 14px rgba(245,194,17,0.6) !important;
    outline: 2px solid #f5c211 !important;
    outline-offset: 2px;
  }
  .node-card.dimmed { opacity: 0.25; }
  .node-card.on-path { border-color: var(--green); background: #1e2e1e; }
  .node-card.cfm-target {
    border: 2px solid var(--green);
    background: #1e3e1e;
    box-shadow: 0 0 12px rgba(166,227,161,0.4);
  }

  /* Sidebar for relevant-node navigation */
  #relevant-sidebar {
    display: none;
    position: fixed; top: 60px; right: 12px; z-index: 100;
    width: 180px; max-height: calc(100vh - 80px);
    background: var(--surface); border: 1px solid #f5c211;
    border-radius: 8px; overflow: hidden;
    font-size: 12px;
    box-shadow: 0 0 12px rgba(245,194,17,0.3);
    flex-direction: column;
  }
  #relevant-sidebar.visible { display: flex; }
  #relevant-sidebar .sidebar-header {
    padding: 8px 10px; background: #2e2a10; color: #f5c211;
    font-weight: bold; border-bottom: 1px solid #f5c211;
    display: flex; justify-content: space-between; align-items: center;
  }
  #relevant-sidebar .sidebar-header .close-btn {
    cursor: pointer; color: var(--dim); font-size: 14px; border: none; background: none;
  }
  #relevant-sidebar .sidebar-header .close-btn:hover { color: var(--text); }
  #relevant-sidebar .sidebar-list {
    overflow-y: auto; flex: 1; padding: 4px 0;
  }
  #relevant-sidebar .sidebar-item {
    padding: 5px 10px; cursor: pointer; color: var(--text);
    display: flex; gap: 6px; align-items: center;
    border-left: 3px solid transparent;
  }
  #relevant-sidebar .sidebar-item:hover { background: #2e2a10; }
  #relevant-sidebar .sidebar-item.active {
    background: #2e2a10; border-left-color: #f5c211; color: #f5c211;
  }
  #relevant-sidebar .sidebar-item .item-idx {
    color: #f5c211; font-weight: bold; min-width: 32px;
  }
  #relevant-sidebar .sidebar-item .item-info {
    color: var(--dim); font-size: 11px; overflow: hidden;
    text-overflow: ellipsis; white-space: nowrap;
  }
  #relevant-sidebar .sidebar-count {
    padding: 4px 10px; color: var(--dim); font-size: 11px;
    border-top: 1px solid var(--border); text-align: center;
  }

  .node-header { display: flex; gap: 8px; align-items: center; flex-wrap: nowrap; }
  .node-idx { color: var(--blue); font-weight: bold; }
  .badge {
    font-size: 10px; padding: 1px 5px; border-radius: 3px;
    font-weight: bold; text-transform: uppercase; white-space: nowrap;
  }
  .badge.cf { background: #1e3e1e; color: var(--green); }
  .badge.fluent { background: #3e2e1e; color: var(--orange); }
  .badge.model { background: #2e1e3e; color: var(--mauve); }
  .badge.frame { background: #3e3e1e; color: var(--yellow); }
  .node-stats { color: var(--dim); font-size: 11px; white-space: nowrap; }

  .toggle-btn {
    display: inline-block; width: 16px; text-align: center;
    color: var(--dim); cursor: pointer; user-select: none; margin-right: 2px;
    flex-shrink: 0;
  }

  .node-detail {
    display: none;
    margin-top: 8px; padding-top: 8px;
    border-top: 1px solid var(--border);
    font-size: 11px; color: var(--dim);
    max-height: 400px; overflow-y: auto;
  }
  .node-detail.open { display: block; }
  .node-detail h4 { color: var(--text); margin: 6px 0 2px; font-size: 12px; }
  .node-detail ul { padding-left: 16px; }
  .node-detail li { margin: 1px 0; list-style: disc; }
  .conflict-item { color: var(--orange); }
  .patch-item { color: var(--yellow); }
  .constraint-item { color: var(--mauve); }
  .children-info { color: var(--blue); }

  .depth-guide {
    color: var(--border); font-size: 11px; user-select: none;
    margin-right: 4px; flex-shrink: 0;
  }

  #status-bar {
    color: var(--dim); font-size: 11px; margin-bottom: 8px;
    position: sticky; top: 44px; z-index: 9;
    background: var(--bg); padding: 2px 0;
  }
</style>
</head>
<body>

<h1>Conflict Search Trace</h1>
<div class="meta" id="meta"></div>

<div class="controls">
  <div>
    <label>Expand to CFM:</label>
    <select id="cfm-select"><option value="">— choose —</option></select>
  </div>
  <div>
    <label>Filter action:</label>
    <input id="filter-action" type="text" placeholder="e.g. put_down" size="14">
  </div>
  <div>
    <label>Filter predicate:</label>
    <input id="filter-pred" type="text" placeholder="e.g. handempty" size="14">
  </div>
  <div>
    <button id="btn-apply">Apply Filter</button>
    <button id="btn-clear">Clear</button>
    <button id="btn-collapse-all">Collapse All</button>
  </div>
</div>
<div id="status-bar"></div>
<div id="relevant-sidebar">
  <div class="sidebar-header">
    <span>Relevant Nodes (<span id="sidebar-count">0</span>)</span>
    <button class="close-btn" id="sidebar-close">&times;</button>
  </div>
  <div class="sidebar-list" id="sidebar-list"></div>
</div>

<div id="tree"></div>

<script>
const DATA = __TRACE_DATA__;
const MAX_FILTER_RENDER = 500;  // safety cap for filter results

// ── Lookup structures (data only, no DOM) ──
const nodeMap = {};
DATA.nodes.forEach(n => { nodeMap[n.index] = n; });

const childrenOf = {};
DATA.nodes.forEach(n => {
  if (n.parent_index != null) {
    if (!childrenOf[n.parent_index]) childrenOf[n.parent_index] = [];
    childrenOf[n.parent_index].push(n.index);
  }
});

const roots = DATA.nodes.filter(n => n.parent_index == null).map(n => n.index);

// ── State ──
const expandedSet = new Set();   // which nodes are expanded
const renderedRows = {};         // nodeIdx → DOM row div
const treeEl = document.getElementById('tree');

// ── Populate meta ──
document.getElementById('meta').innerHTML =
  `Fold: <b>${DATA.fold_dir}</b> · ` +
  (DATA.target_action ? `Target: <b>${DATA.target_action}</b> / <b>${DATA.target_predicate}</b> · ` : '') +
  `Nodes: <b>${DATA.outcome.nodes_expanded}</b> · ` +
  `CFMs: <b>${DATA.outcome.conflict_free_count}</b> · ` +
  `Best cost: <b>${DATA.outcome.best_cost ?? 'none'}</b> · ` +
  `Mode: <b>${DATA.search_params.search_mode}</b>`;

// ── Populate CFM dropdown ──
const cfms = DATA.nodes.filter(n => n.cfm_index != null);
cfms.sort((a, b) => a.cfm_index - b.cfm_index);
const sel = document.getElementById('cfm-select');
cfms.forEach(n => {
  const opt = document.createElement('option');
  opt.value = n.index;
  opt.textContent = `CFM #${n.cfm_index} (node ${n.index}, cost ${n.cost.toFixed(2)}, ` +
    `mc=${n.model_constraints.length}, fp=${n.fluent_patches.length})`;
  sel.appendChild(opt);
});

// ── Helpers ──
function esc(s) {
  const d = document.createElement('span');
  d.textContent = s;
  return d.innerHTML;
}

function getAncestors(nodeIdx) {
  const path = [];
  let cur = nodeIdx;
  while (cur != null) {
    path.push(cur);
    cur = nodeMap[cur]?.parent_index;
  }
  return path;
}

function setStatus(msg) {
  document.getElementById('status-bar').textContent = msg;
}

function indent(depth) {
  return Math.min(depth * 16, 480);
}

// ── Create a single node row (flat div, not nested) ──
function createRow(nodeIdx) {
  if (renderedRows[nodeIdx]) return renderedRows[nodeIdx];

  const n = nodeMap[nodeIdx];
  const hasChildren = (childrenOf[n.index] || []).length > 0;
  const isExpanded = expandedSet.has(n.index);

  const row = document.createElement('div');
  row.className = 'node-row';
  row.id = `node-${n.index}`;
  row.dataset.idx = n.index;
  row.style.marginLeft = indent(n.depth) + 'px';

  const card = document.createElement('div');
  card.className = 'node-card'
    + (n.is_conflict_free ? ' cf' : ' conflict')
    + (n.parent_index == null ? ' root' : '');

  let badges = '';
  if (n.is_conflict_free) badges += `<span class="badge cf">CF #${n.cfm_index}</span>`;
  if (n.branch_type === 'fluent_fix') badges += '<span class="badge fluent">fluent</span>';
  else if (n.branch_type === 'model_fix') badges += '<span class="badge model">model</span>';
  else if (n.branch_type === 'frame_axiom_prev_fix') badges += '<span class="badge frame">frame</span>';

  const toggleChar = !hasChildren ? '·' : (isExpanded ? '▼' : '▶');

  card.innerHTML = `
    <div class="node-header">
      <span class="toggle-btn">${toggleChar}</span>
      <span class="node-idx">#${n.index}</span>
      ${badges}
      <span class="node-stats">
        d=${n.depth} cost=${n.cost.toFixed(2)}
        ${n.is_conflict_free ? '' : n.conflicts.length + 'c'}
        mc=${n.model_constraints.length} fp=${n.fluent_patches.length}
      </span>
    </div>
    <div class="node-detail">${buildDetail(n)}</div>
  `;

  // Toggle expand/collapse
  card.querySelector('.toggle-btn').addEventListener('click', (e) => {
    e.stopPropagation();
    if (!hasChildren) return;
    if (expandedSet.has(n.index)) {
      collapseNode(n.index);
    } else {
      expandSingle(n.index);
    }
  });

  // Toggle detail on card body click
  card.addEventListener('click', (e) => {
    if (e.target.closest('.toggle-btn')) return;
    card.querySelector('.node-detail').classList.toggle('open');
  });

  row.appendChild(card);
  renderedRows[nodeIdx] = row;
  return row;
}

// ── Expand one node: insert its children rows right after it ──
function expandSingle(nodeIdx) {
  expandedSet.add(nodeIdx);
  const row = renderedRows[nodeIdx];
  if (!row) return;

  // Update toggle icon
  const btn = row.querySelector('.toggle-btn');
  if (btn) btn.textContent = '▼';

  const children = childrenOf[nodeIdx] || [];
  if (children.length === 0) return;

  // Insert children right after this row
  let insertAfter = row;
  children.forEach(ci => {
    const childRow = createRow(ci);
    insertAfter.after(childRow);
    insertAfter = childRow;
    // If this child was already expanded (from a previous session), recursively show its children
    if (expandedSet.has(ci)) {
      insertAfter = insertDescendants(ci, insertAfter);
    }
  });
}

// ── Insert all visible descendants (for already-expanded subtrees) ──
function insertDescendants(nodeIdx, afterEl) {
  const children = childrenOf[nodeIdx] || [];
  let cursor = afterEl;
  children.forEach(ci => {
    const childRow = createRow(ci);
    cursor.after(childRow);
    cursor = childRow;
    if (expandedSet.has(ci)) {
      cursor = insertDescendants(ci, cursor);
    }
  });
  return cursor;
}

// ── Collapse one node: remove all descendant rows from DOM ──
function collapseNode(nodeIdx) {
  expandedSet.delete(nodeIdx);
  const row = renderedRows[nodeIdx];
  if (!row) return;

  const btn = row.querySelector('.toggle-btn');
  if (btn) btn.textContent = '▶';

  removeDescendants(nodeIdx);
}

function removeDescendants(nodeIdx) {
  const children = childrenOf[nodeIdx] || [];
  children.forEach(ci => {
    removeDescendants(ci);
    const childRow = renderedRows[ci];
    if (childRow && childRow.parentNode) {
      childRow.parentNode.removeChild(childRow);
    }
    delete renderedRows[ci];
  });
}

function buildDetail(n) {
  let html = '';

  if (n.model_constraints.length) {
    const show = n.model_constraints.slice(0, 50);
    html += `<h4>Model Constraints (${n.model_constraints.length})</h4><ul>`;
    show.forEach(c => html += `<li class="constraint-item">${esc(c)}</li>`);
    if (n.model_constraints.length > 50) html += `<li>… and ${n.model_constraints.length - 50} more</li>`;
    html += '</ul>';
  }

  if (n.fluent_patches.length) {
    const show = n.fluent_patches.slice(0, 50);
    html += `<h4>Fluent Patches (${n.fluent_patches.length})</h4><ul>`;
    show.forEach(p => html += `<li class="patch-item">${esc(p)}</li>`);
    if (n.fluent_patches.length > 50) html += `<li>… and ${n.fluent_patches.length - 50} more</li>`;
    html += '</ul>';
  }

  if (n.conflicts.length) {
    html += `<h4>Conflicts (${n.conflicts.length})</h4><ul>`;
    n.conflicts.forEach(c =>
      html += `<li class="conflict-item">${esc(c.type)}: ${esc(c.fluent)} vs ${esc(c.predicate)} in ${esc(c.action)} [obs=${c.obs}, comp=${c.comp}]</li>`
    );
    html += '</ul>';
  }

  if (n.chosen_group) {
    html += `<h4>Chosen Group (${n.chosen_group.length})</h4><ul>`;
    n.chosen_group.forEach(c =>
      html += `<li class="conflict-item">${esc(c.type)}: ${esc(c.fluent)} in ${esc(c.action)} [obs=${c.obs}, comp=${c.comp}]</li>`
    );
    html += '</ul>';
  }

  const ch = n.children;
  if (ch.fluent_fix || ch.model_fix) {
    html += '<h4>Children</h4><ul>';
    if (ch.fluent_fix) html += `<li class="children-info">A (fluent): cost=${ch.fluent_fix.cost?.toFixed(2) ?? '?'} — ${esc(ch.fluent_fix.desc)}</li>`;
    if (ch.model_fix) html += `<li class="children-info">B (model): cost=${ch.model_fix.cost?.toFixed(2) ?? '?'} — ${esc(ch.model_fix.desc)}</li>`;
    html += '</ul>';
  }

  return html || '<em>No details</em>';
}

// ── Initial render: only root nodes ──
roots.forEach(r => treeEl.appendChild(createRow(r)));
setStatus(`Loaded ${DATA.nodes.length} nodes. Click ▶ to expand, or select a CFM.`);

// ── Collapse everything ──
function collapseAll() {
  // Remove all non-root rows
  Object.keys(renderedRows).forEach(idx => {
    const id = parseInt(idx);
    if (!roots.includes(id)) {
      const row = renderedRows[id];
      if (row?.parentNode) row.parentNode.removeChild(row);
      delete renderedRows[id];
    }
  });
  expandedSet.clear();
  // Reset root toggle icons
  roots.forEach(r => {
    const row = renderedRows[r];
    if (!row) return;
    const btn = row.querySelector('.toggle-btn');
    const hasChildren = (childrenOf[r] || []).length > 0;
    if (btn) btn.textContent = hasChildren ? '▶' : '·';
  });
  clearClasses();
}

function clearClasses() {
  Object.values(renderedRows).forEach(row => {
    const card = row.querySelector('.node-card');
    if (card) card.classList.remove('highlight', 'dimmed', 'on-path', 'cfm-target', 'relevant');
    const det = row.querySelector('.node-detail');
    if (det) det.classList.remove('open');
  });
}

// ── Expand the path from root to a given node ──
function expandPathTo(nodeIdx) {
  const ancestors = getAncestors(nodeIdx).reverse();  // root first
  for (const idx of ancestors) {
    if (idx === nodeIdx) break;  // don't expand the target itself
    if (!expandedSet.has(idx)) {
      expandSingle(idx);
    }
  }
}

// ── Filter logic ──

// Does the node's chosen_group contain the action+predicate?
// This is the "relevant" check — the node branched on this action/predicate pair.
function chosenGroupMatches(node, action, pred) {
  if (!node.chosen_group || node.chosen_group.length === 0) return false;
  const a = action.toLowerCase();
  const p = pred.toLowerCase();
  return node.chosen_group.some(c => {
    const matchA = !action || c.action.toLowerCase() === a;
    const matchP = !pred || (c.predicate + ' ' + c.fluent).toLowerCase().includes(p);
    return matchA && matchP;
  });
}

// Does the node mention the action/predicate anywhere (conflicts, patches, constraints)?
function matchesFilter(node, action, pred) {
  const texts = [
    ...node.conflicts.map(c => c.action + ' ' + c.predicate + ' ' + c.fluent),
    ...(node.chosen_group || []).map(c => c.action + ' ' + c.predicate + ' ' + c.fluent),
    ...node.model_constraints,
    ...node.fluent_patches,
  ].join(' ').toLowerCase();
  if (action && !texts.includes(action.toLowerCase())) return false;
  if (pred && !texts.includes(pred.toLowerCase())) return false;
  return true;
}

// ── Relevant-node sidebar state ──
let relevantNodeIds = [];
let currentMatchIdx = -1;

function buildSidebar(nodeIds) {
  const sidebar = document.getElementById('relevant-sidebar');
  const list = document.getElementById('sidebar-list');
  const countEl = document.getElementById('sidebar-count');
  list.innerHTML = '';
  relevantNodeIds = nodeIds;
  currentMatchIdx = -1;

  if (nodeIds.length === 0) {
    sidebar.classList.remove('visible');
    document.body.classList.remove('sidebar-open');
    return;
  }

  countEl.textContent = nodeIds.length;
  const MAX_SIDEBAR = 200;
  const showIds = nodeIds.slice(0, MAX_SIDEBAR);
  showIds.forEach((idx, i) => {
    const n = nodeMap[idx];
    const item = document.createElement('div');
    item.className = 'sidebar-item';
    item.dataset.nodeIdx = idx;
    item.dataset.listIdx = i;

    // Brief description from chosen_group
    let info = '';
    if (n.chosen_group && n.chosen_group.length > 0) {
      const cg = n.chosen_group[0];
      info = cg.type + ': ' + cg.fluent;
    }

    item.innerHTML = '<span class="item-idx">#' + idx + '</span>'
      + '<span class="item-info">' + (info || 'd=' + n.depth) + '</span>';

    item.addEventListener('click', () => navigateToRelevant(i));
    list.appendChild(item);
  });

  if (nodeIds.length > MAX_SIDEBAR) {
    const more = document.createElement('div');
    more.className = 'sidebar-item';
    more.style.color = 'var(--dim)';
    more.style.fontStyle = 'italic';
    more.textContent = '… and ' + (nodeIds.length - MAX_SIDEBAR) + ' more';
    list.appendChild(more);
  }

  sidebar.classList.add('visible');
  document.body.classList.add('sidebar-open');
}

function hideSidebar() {
  document.getElementById('relevant-sidebar').classList.remove('visible');
  document.body.classList.remove('sidebar-open');
  relevantNodeIds = [];
  currentMatchIdx = -1;
}

function navigateToRelevant(listIdx) {
  if (relevantNodeIds.length === 0) return;
  currentMatchIdx = listIdx;

  // Update active state in sidebar
  document.querySelectorAll('#sidebar-list .sidebar-item').forEach((el, i) => {
    el.classList.toggle('active', i === listIdx);
  });

  const nodeIdx = relevantNodeIds[listIdx];
  expandPathTo(nodeIdx);
  const row = renderedRows[nodeIdx];
  if (row) {
    // Mark as relevant + open detail
    const card = row.querySelector('.node-card');
    if (card) card.classList.add('relevant');
    const det = row.querySelector('.node-detail');
    if (det) det.classList.add('open');
    row.scrollIntoView({ behavior: 'smooth', block: 'center' });
    // Flash
    if (card) {
      card.style.transition = 'none';
      card.style.boxShadow = '0 0 24px rgba(245,194,17,0.9)';
      setTimeout(() => {
        card.style.transition = 'box-shadow 0.6s';
        card.style.boxShadow = '0 0 14px rgba(245,194,17,0.6)';
      }, 250);
    }
  }
}

function applyFilter() {
  const cfmVal = sel.value;
  const action = document.getElementById('filter-action').value.trim();
  const pred = document.getElementById('filter-pred').value.trim();

  collapseAll();
  hideSidebar();

  if (!cfmVal && !action && !pred) {
    setStatus('Filters cleared.');
    return;
  }

  // ── CFM path ──
  let pathSet = new Set();
  if (cfmVal !== '') {
    const targetIdx = parseInt(cfmVal);
    expandPathTo(targetIdx);
    pathSet = new Set(getAncestors(targetIdx));
    pathSet.forEach(idx => {
      const row = renderedRows[idx];
      if (!row) return;
      const card = row.querySelector('.node-card');
      if (!card) return;
      if (idx === targetIdx) {
        card.classList.add('cfm-target');
        card.querySelector('.node-detail')?.classList.add('open');
      } else {
        card.classList.add('on-path');
      }
    });
  }

  // ── Action/predicate filter ──
  if (action || pred) {
    // 1. Find "relevant" nodes — chosen_group matches action+predicate (decision points)
    const relevant = new Set();
    DATA.nodes.forEach(n => {
      if (chosenGroupMatches(n, action, pred)) relevant.add(n.index);
    });

    // 2. Find "mentioned" nodes — action+predicate appears anywhere
    const mentioned = new Set();
    DATA.nodes.forEach(n => {
      if (matchesFilter(n, action, pred)) mentioned.add(n.index);
    });

    // Build sidebar — always, regardless of count.
    // Sidebar is just a list of indices, very lightweight.
    const sortedRelevant = [...relevant].sort((a, b) => a - b);
    buildSidebar(sortedRelevant);

    // Only bulk-expand if count is manageable
    const toRender = new Set();
    relevant.forEach(idx => getAncestors(idx).forEach(a => toRender.add(a)));

    if (toRender.size <= MAX_FILTER_RENDER) {
      // Expand and style all
      relevant.forEach(idx => expandPathTo(idx));
      Object.keys(renderedRows).forEach(k => {
        const id = parseInt(k);
        const card = renderedRows[id].querySelector('.node-card');
        if (!card) return;
        if (relevant.has(id)) {
          card.classList.add('relevant');
        } else if (!toRender.has(id) && !pathSet.has(id)) {
          card.classList.add('dimmed');
        }
      });
      setStatus(`Filter: ${relevant.size} relevant (chosen group), ${mentioned.size} mentions. All expanded.`);
      navigateToRelevant(0);
    } else {
      // Too many to bulk-expand — show sidebar only, navigate on click
      setStatus(`Filter: ${relevant.size} relevant (chosen group). Too many to expand all — use sidebar to navigate one at a time.`);
      if (sortedRelevant.length > 0) navigateToRelevant(0);
    }
  } else if (cfmVal) {
    setStatus(`Path to CFM #${nodeMap[parseInt(cfmVal)].cfm_index}: ${pathSet.size} nodes.`);
  }

  // Scroll to CFM target if selected (and no filter auto-navigated)
  if (cfmVal && !(action || pred)) {
    const target = renderedRows[parseInt(cfmVal)];
    if (target) setTimeout(() => target.scrollIntoView({ behavior: 'smooth', block: 'center' }), 50);
  }
}

// ── Event listeners ──
document.getElementById('btn-apply').addEventListener('click', applyFilter);
document.getElementById('btn-clear').addEventListener('click', () => {
  sel.value = '';
  document.getElementById('filter-action').value = '';
  document.getElementById('filter-pred').value = '';
  collapseAll();
  hideSidebar();
  setStatus('Filters cleared.');
});
document.getElementById('btn-collapse-all').addEventListener('click', () => {
  collapseAll();
  setStatus('Collapsed.');
});
document.getElementById('filter-action').addEventListener('keydown', e => { if (e.key === 'Enter') applyFilter(); });
document.getElementById('filter-pred').addEventListener('keydown', e => { if (e.key === 'Enter') applyFilter(); });
sel.addEventListener('change', applyFilter);
document.getElementById('sidebar-close').addEventListener('click', hideSidebar);
</script>
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate an interactive HTML viewer from a conflict search trace.",
    )
    parser.add_argument(
        "trace_json", type=Path,
        help="Path to search_trace.json (produced by trace_spurious_effects.py --retrace).",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output HTML path. Defaults to <trace_json>.html.",
    )
    args = parser.parse_args()

    trace_path: Path = args.trace_json.resolve()
    if not trace_path.exists():
        print(f"Error: {trace_path} not found.", file=sys.stderr)
        sys.exit(1)

    with open(trace_path) as f:
        trace_data = json.load(f)

    action = trace_data.get('target_action')
    predicate = trace_data.get('target_predicate')
    title = f"{action}/{predicate}" if action else "search trace"
    html = HTML_TEMPLATE.replace("{title}", title)
    html = html.replace("__TRACE_DATA__", json.dumps(trace_data))

    output_path = args.output or trace_path.with_suffix(".html")
    output_path.write_text(html, encoding="utf-8")
    print(f"Viewer saved to: {output_path}")


if __name__ == "__main__":
    main()
