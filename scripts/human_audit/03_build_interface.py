#!/usr/bin/env python3
"""Build the HTML annotation interface for the error-attribution human audit.

Reads human_audit/bundle/records.json and writes human_audit/index.html with
the records embedded (no fetch needed). Serve the human_audit/ directory so
that PDF / script links open in new tabs:

    cd human_audit && python3 -m http.server 8765
    open http://localhost:8765/

Annotations are autosaved to localStorage and exported via the toolbar
(JSON for merging/backup, CSV for analysis).
"""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BUNDLE = REPO / "human_audit" / "bundle"
OUT = REPO / "human_audit" / "index.html"

DIVERGENCE_TYPES = {
    "S1": "Wrong model specification — incorrect FE, clustering level, or SE type",
    "S2": "Wrong estimator / inference — wrong estimator (OLS vs IV) or missing inference step",
    "S3": "Data source substitution — proxy used / required dataset absent from package",
    "S4": "Wrong sample restriction — filter missing, wrong condition, or wrong subset",
    "S5": "Wrong variable construction — outcome/predictor coded differently than reference",
    "S6": "Missing analysis component — required step entirely omitted",
    "S8": "Wrong merge / transform logic — wrong join type, key, duplicate handling, or reshape",
    "S9": "Wrong sequencing — steps in wrong order, changing results",
    "S0": "Other — does not fit any category above",
}

SEVERITIES = {
    "minor": "unlikely to materially affect point estimates or conclusions",
    "medium": "could shift estimates noticeably; sign/significance probably stable",
    "critical": "likely changes sign, significance, or core conclusion of a main result",
}

ROOT_CAUSES = {
    "Data not in package": "Required raw data absent from the replication package — the agent could not compute this result.",
    "Paper-code mismatch": "The paper explicitly contradicts what the original code does (the agent followed the paper).",
    "Paper underspecified": "The original code does something the paper is silent about — the agent had to guess.",
    "Summary gap (contradicts)": "The methodology summary explicitly contradicts the paper.",
    "Summary gap (omission)": "The methodology summary dropped information the paper provided.",
    "Agent contradicted summary": "The agent explicitly deviated from the summary's instructions.",
    "Agent missed summary info": "The summary specified it, but the agent did not implement it.",
    "Insufficient specification": "One or more consistency checks returned 'unclear' — no definitive attribution.",
    "Unexplained": "No contradictions or omissions identified by any consistency check.",
}

CHECK_LABELS = {
    "paper_vs_original_code": "Check 1 — paper vs. original code (does the paper support the original behavior?)",
    "paper_vs_summary": "Check 2 — paper vs. summary (does the summary faithfully represent the paper?)",
    "summary_vs_agent": "Check 3 — summary vs. agent (does the agent code implement the summary?)",
}


def build_html(data: dict) -> str:
    payload = json.dumps({
        "records": data["records"],
        "meta": data.get("sample_meta", {}),
        "defs": {
            "types": DIVERGENCE_TYPES,
            "severities": SEVERITIES,
            "rootCauses": ROOT_CAUSES,
            "checks": CHECK_LABELS,
        },
    }, ensure_ascii=False).replace("</", "<\\/")

    return HTML_TEMPLATE.replace("__DATA__", payload)


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Error-Attribution Human Audit</title>
<style>
  :root {
    --bg: #f6f7f9; --panel: #ffffff; --border: #dcdfe4; --text: #1c2127;
    --muted: #5c6470; --accent: #2563eb; --accent-soft: #eff4ff;
    --ok: #15803d; --warn: #b45309; --bad: #b91c1c;
    --chip-bg: #eef1f5; --code-bg: #f3f4f6;
  }
  * { box-sizing: border-box; }
  body { margin: 0; font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
         color: var(--text); background: var(--bg); }
  .app { display: grid; grid-template-columns: 300px 1fr 0; grid-template-rows: 52px 1fr; height: 100vh; }
  .app.viewer-open { grid-template-columns: 300px minmax(0, 1fr) minmax(400px, 46%); }

  /* ---------- top bar ---------- */
  .topbar { grid-column: 1 / -1; display: flex; align-items: center; gap: 12px; padding: 0 16px;
            background: var(--panel); border-bottom: 1px solid var(--border); }
  .topbar h1 { font-size: 15px; margin: 0; white-space: nowrap; }
  .topbar .spacer { flex: 1; }
  .topbar input[type=text] { padding: 5px 8px; border: 1px solid var(--border); border-radius: 6px; width: 160px; }
  .topbar button { padding: 5px 10px; border: 1px solid var(--border); border-radius: 6px;
                   background: var(--panel); cursor: pointer; font-size: 13px; }
  .topbar button:hover { background: var(--accent-soft); }
  .progress { font-size: 13px; color: var(--muted); white-space: nowrap; }

  /* ---------- sidebar ---------- */
  .sidebar { overflow-y: auto; background: var(--panel); border-right: 1px solid var(--border); }
  .filters { display: flex; gap: 6px; padding: 10px 12px; position: sticky; top: 0;
             background: var(--panel); border-bottom: 1px solid var(--border); z-index: 2; }
  .filters button { flex: 1; padding: 4px 0; font-size: 12px; border: 1px solid var(--border);
                    border-radius: 6px; background: var(--panel); cursor: pointer; }
  .filters button.active { background: var(--accent); color: #fff; border-color: var(--accent); }
  .item { padding: 8px 12px; border-bottom: 1px solid #eceef1; cursor: pointer; }
  .item:hover { background: var(--accent-soft); }
  .item.selected { background: var(--accent-soft); box-shadow: inset 3px 0 0 var(--accent); }
  .item .row1 { display: flex; align-items: center; gap: 6px; font-size: 12.5px; font-weight: 600; }
  .item .row2 { font-size: 11.5px; color: var(--muted); margin-top: 1px;
                overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .dot { width: 8px; height: 8px; border-radius: 50%; background: #cbd2da; flex: none; }
  .dot.done { background: var(--ok); }
  .dot.partial { background: var(--warn); }

  /* ---------- main panel ---------- */
  .main { overflow-y: auto; padding: 20px 26px 60px; }
  .card { background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
          padding: 16px 20px; margin-bottom: 14px; max-width: 1000px; }
  .head-line { display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap; }
  .head-line h2 { font-size: 17px; margin: 0; }
  .head-sub { color: var(--muted); font-size: 12.5px; margin-top: 3px; }
  .chips { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 10px; }
  .chip { padding: 2px 10px; border-radius: 999px; background: var(--chip-bg); font-size: 12px; font-weight: 600; }
  .chip.sev-critical { background: #fee2e2; color: var(--bad); }
  .chip.sev-medium { background: #fef3c7; color: var(--warn); }
  .chip.sev-minor { background: #dcfce7; color: var(--ok); }
  .chip.type { background: #e0e7ff; color: #3730a3; }
  .chip.cause { background: #d1fae5; color: #065f46; }
  .desc { font-size: 15px; margin: 12px 0 2px; }

  .kv { display: grid; grid-template-columns: 190px 1fr; gap: 4px 14px; font-size: 13px; margin-top: 8px; }
  .kv .k { color: var(--muted); }
  .cells-list { margin: 4px 0 0; padding-left: 18px; font-size: 13px; }
  .cells-list li { margin: 1px 0; }

  details { border: 1px solid var(--border); border-radius: 8px; padding: 0; margin-top: 10px; }
  details summary { cursor: pointer; padding: 9px 14px; font-weight: 600; font-size: 13.5px;
                    list-style: none; display: flex; align-items: center; gap: 8px; }
  details summary::before { content: "▸"; transition: transform .15s; font-size: 11px; }
  details[open] summary::before { transform: rotate(90deg); }
  details .body { padding: 4px 16px 14px; border-top: 1px solid var(--border); }
  pre { background: var(--code-bg); border: 1px solid var(--border); border-radius: 6px;
        padding: 10px 12px; overflow-x: auto; font: 12px/1.45 "SF Mono", ui-monospace, Menlo, Consolas, monospace;
        white-space: pre-wrap; word-break: break-word; }
  .loc { font-size: 12px; color: var(--muted); margin: 8px 0 4px; }
  .verdict { display: inline-block; padding: 1px 8px; border-radius: 999px; font-size: 12px; font-weight: 700; }
  .verdict.consistent { background: #dcfce7; color: var(--ok); }
  .verdict.contradicts { background: #fee2e2; color: var(--bad); }
  .verdict.omission { background: #fef3c7; color: var(--warn); }
  .verdict.unclear, .verdict.missing { background: var(--chip-bg); color: var(--muted); }
  .check-block { margin: 10px 0; }
  .check-block .cb-title { font-size: 13px; font-weight: 600; }
  .check-block .cb-note { font-size: 13px; color: var(--muted); margin-top: 2px; }

  .links { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 4px; }
  .links a { display: inline-block; padding: 6px 12px; border: 1px solid var(--border); border-radius: 7px;
             background: var(--panel); color: var(--accent); text-decoration: none; font-size: 13px; font-weight: 600; }
  .links a:hover { background: var(--accent-soft); }
  .links a.active-doc { background: var(--accent); color: #fff; border-color: var(--accent); }
  .links .missing-note { font-size: 12px; color: var(--muted); align-self: center; }

  /* ---------- inline document viewer ---------- */
  .viewer { display: none; flex-direction: column; background: var(--panel);
            border-left: 1px solid var(--border); min-width: 0; overflow: hidden; }
  .app.viewer-open .viewer { display: flex; }
  .viewer-head { display: flex; align-items: center; gap: 10px; padding: 8px 12px;
                 border-bottom: 1px solid var(--border); flex: none; }
  .viewer-head .vtitle { flex: 1; font-size: 13px; font-weight: 600; overflow: hidden;
                         text-overflow: ellipsis; white-space: nowrap; }
  .viewer-head a { color: var(--accent); text-decoration: none; font-size: 12.5px; white-space: nowrap; }
  .viewer-head button { border: 1px solid var(--border); background: var(--panel); border-radius: 6px;
                        padding: 3px 9px; cursor: pointer; font-size: 13px; }
  .viewer-body { flex: 1; overflow: auto; min-height: 0; }
  .viewer-body iframe { width: 100%; height: 100%; border: 0; display: block; }
  .code-view { font: 12px/1.5 "SF Mono", ui-monospace, Menlo, Consolas, monospace; padding: 6px 0 30px; }
  .code-line { display: flex; }
  .code-line .ln { flex: none; width: 52px; padding-right: 12px; text-align: right; color: #9aa2ad;
                   user-select: none; }
  .code-line .lc { white-space: pre-wrap; word-break: break-word; flex: 1; padding-right: 12px; }
  .code-line.hl { background: #fef3c7; }
  .viewer-msg { padding: 16px; color: var(--muted); font-size: 13px; }
  .viewer-body pre.plain { border: 0; border-radius: 0; margin: 0; }
  .tbl-wrap { padding: 12px; overflow-x: auto; }
  .tbl-wrap h3 { font-size: 13.5px; margin: 0 0 8px; }
  table.tbl { border-collapse: collapse; font-size: 12.5px; }
  table.tbl th, table.tbl td { border: 1px solid var(--border); padding: 4px 9px; text-align: right;
                               white-space: nowrap; }
  table.tbl th { background: var(--chip-bg); font-weight: 600; }
  table.tbl td:first-child, table.tbl th:first-child { text-align: left; }
  table.tbl td.hl { background: #fef3c7; font-weight: 600; }
  .tbl-notes { font-size: 12px; color: var(--muted); margin-top: 10px; max-width: 620px; }

  /* ---------- annotation ---------- */
  .anno { border-left: 4px solid var(--accent); }
  .q { margin: 14px 0; }
  .q .q-label { font-weight: 600; font-size: 13.5px; margin-bottom: 2px; }
  .q .q-context { font-size: 12.5px; color: var(--muted); margin-bottom: 6px; }
  .q .q-context b { color: var(--text); }
  .opts { display: flex; gap: 8px; }
  .opts label { display: flex; align-items: center; gap: 5px; padding: 5px 12px; border: 1px solid var(--border);
                border-radius: 7px; cursor: pointer; font-size: 13px; background: var(--panel); }
  .opts label:has(input:checked) { border-color: var(--accent); background: var(--accent-soft); font-weight: 600; }
  .opts input { margin: 0; }
  .q.disabled { opacity: .45; pointer-events: none; }
  textarea { width: 100%; min-height: 64px; border: 1px solid var(--border); border-radius: 7px;
             padding: 8px 10px; font: inherit; resize: vertical; }
  .nav-row { display: flex; gap: 10px; margin-top: 14px; max-width: 1000px; }
  .nav-row button { padding: 8px 18px; border: 1px solid var(--border); border-radius: 8px;
                    background: var(--panel); cursor: pointer; font-size: 14px; }
  .nav-row .primary { background: var(--accent); color: #fff; border-color: var(--accent); font-weight: 600; }
  .nav-row .spacer { flex: 1; }
  .saved-note { font-size: 12px; color: var(--ok); align-self: center; }
  kbd { background: var(--chip-bg); border-radius: 4px; padding: 0 5px; font-size: 11px; }
</style>
</head>
<body>
<div class="app">
  <div class="topbar">
    <h1>Error-Attribution Human Audit</h1>
    <span class="progress" id="progress"></span>
    <span class="spacer"></span>
    <input type="text" id="annotator" placeholder="Annotator name">
    <button id="exportJson">Export JSON</button>
    <button id="exportCsv">Export CSV</button>
    <button id="importJson">Import JSON</button>
    <input type="file" id="importFile" accept=".json" style="display:none">
  </div>
  <div class="sidebar">
    <div class="filters">
      <button data-f="all" class="active">All</button>
      <button data-f="todo">To do</button>
      <button data-f="done">Done</button>
    </div>
    <div id="list"></div>
  </div>
  <div class="main" id="main"></div>
  <div class="viewer">
    <div class="viewer-head">
      <span class="vtitle" id="vtitle"></span>
      <a id="vnewtab" href="#" target="_blank">open in new tab ↗</a>
      <button onclick="closeViewer()" title="Close (Esc)">✕</button>
    </div>
    <div class="viewer-body" id="vbody"></div>
  </div>
</div>

<script>
const DATA = __DATA__;
const RECORDS = DATA.records;
const DEFS = DATA.defs;
const LS_KEY = "je_audit_v1";

let state = load();
let currentIdx = 0;
let filter = "all";

function load() {
  try { return JSON.parse(localStorage.getItem(LS_KEY)) || { annotator: "", annotations: {} }; }
  catch { return { annotator: "", annotations: {} }; }
}
function save() { localStorage.setItem(LS_KEY, JSON.stringify(state)); }

function annoFor(id) {
  if (!state.annotations[id]) state.annotations[id] = { real: "", source: "", type: "", severity: "", notes: "", ts: "" };
  return state.annotations[id];
}
function annoStatus(id) {
  const a = state.annotations[id];
  if (!a) return "none";
  const qs = a.real === "no" ? [a.real] : [a.real, a.source, a.type, a.severity];
  const done = qs.filter(Boolean).length;
  if (done === 0) return "none";
  return done === qs.length ? "done" : "partial";
}

function esc(s) {
  return String(s ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

/* ---------------- sidebar ---------------- */
function renderList() {
  const list = document.getElementById("list");
  list.innerHTML = "";
  RECORDS.forEach((r, i) => {
    const st = annoStatus(r.audit_id);
    if (filter === "todo" && st === "done") return;
    if (filter === "done" && st !== "done") return;
    const div = document.createElement("div");
    div.className = "item" + (i === currentIdx ? " selected" : "");
    div.innerHTML = `
      <div class="row1"><span class="dot ${st}"></span>${esc(r.audit_id)} · ${esc(r.record.output)}</div>
      <div class="row2">${esc(r.paper_slug)} · ${esc(r.approach)} · ${esc(r.root_cause)}</div>`;
    div.onclick = () => { if (i !== currentIdx) closeViewer(); currentIdx = i; render(); };
    list.appendChild(div);
  });
  const done = RECORDS.filter(r => annoStatus(r.audit_id) === "done").length;
  document.getElementById("progress").textContent = `${done} / ${RECORDS.length} annotated`;
}

/* ---------------- main panel ---------------- */
function checkBlock(rec, key) {
  const verdict = rec[key] || "—";
  const note = rec[key + "_note"] || "";
  return `<div class="check-block">
    <div class="cb-title">${esc(DEFS.checks[key])} &nbsp;<span class="verdict ${esc(verdict)}">${esc(verdict)}</span></div>
    ${note ? `<div class="cb-note">${esc(note)}</div>` : ""}
  </div>`;
}

function docLink(path, label, lineSpec) {
  const name = path.endsWith("/") ? label : `${label}: ${path.split("/").pop()}`;
  return `<a href="bundle/${encodeURI(path)}" target="_blank" title="${esc(path)}"
    data-doc="${esc(path)}" data-line="${esc(lineSpec || "")}"
    onclick="return linkClick(event, this)">${esc(name)}</a>`;
}
function fileLinks(paths, label, lineSpec) {
  return (paths || []).map(p => docLink(p, label, lineSpec)).join("");
}
function linkClick(e, el) {
  if (e.metaKey || e.ctrlKey || e.shiftKey) return true;  // modifier click -> real new tab
  openDoc(el.dataset.doc, el.dataset.line);
  document.querySelectorAll(".links a").forEach(a => a.classList.toggle("active-doc", a === el));
  return false;
}

/* ---------------- inline document viewer ---------------- */
function parseLineSpec(spec) {
  const m = String(spec || "").match(/(\d+)(?:\s*[-–:]\s*(\d+))?/);
  if (!m) return [null, null];
  const a = parseInt(m[1], 10);
  return [a, m[2] ? parseInt(m[2], 10) : a];
}

async function openDoc(path, lineSpec) {
  document.querySelector(".app").classList.add("viewer-open");
  document.getElementById("vtitle").textContent = path;
  document.getElementById("vnewtab").href = "bundle/" + encodeURI(path);
  const body = document.getElementById("vbody");
  const url = "bundle/" + encodeURI(path);

  if (path.toLowerCase().endsWith(".pdf") || path.endsWith("/")) {
    body.innerHTML = `<iframe src="${url}"></iframe>`;
    return;
  }
  body.innerHTML = `<div class="viewer-msg">loading…</div>`;
  try {
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(resp.status);
    const text = await resp.text();
    if (text.length > 1_500_000) {  // very large file: plain fast render, no line numbers
      const pre = document.createElement("pre");
      pre.className = "plain";
      pre.textContent = text;
      body.innerHTML = "";
      body.appendChild(pre);
      return;
    }
    const [a, b] = parseLineSpec(lineSpec);
    body.innerHTML = `<div class="code-view">` + text.split("\n").map((l, i) => {
      const n = i + 1;
      const hl = a !== null && n >= a && n <= b ? " hl" : "";
      return `<div class="code-line${hl}" id="L${n}"><span class="ln">${n}</span><span class="lc">${esc(l)}</span></div>`;
    }).join("") + `</div>`;
    if (a !== null) document.getElementById("L" + a)?.scrollIntoView({ block: "center" });
  } catch (err) {
    body.innerHTML = `<div class="viewer-msg">Could not load file (${esc(err.message)}) —
      <a href="${url}" target="_blank">try opening in a new tab</a>.</div>`;
  }
}

function closeViewer() {
  document.querySelector(".app").classList.remove("viewer-open");
  document.getElementById("vbody").innerHTML = "";
  document.querySelectorAll(".links a").forEach(a => a.classList.remove("active-doc"));
}

/* ---------------- table viewer ---------------- */
function normId(s) { return String(s || "").toLowerCase().replace(/[^a-z0-9]/g, ""); }

function tableClick(e, el, kind) {
  if (e.metaKey || e.ctrlKey || e.shiftKey) return true;
  openTable(kind);
  document.querySelectorAll(".links a").forEach(a => a.classList.toggle("active-doc", a === el));
  return false;
}

async function openTable(kind) {
  const r = RECORDS[currentIdx];
  const path = kind === "original" ? r.assets.original_results : r.assets.replicated_table;
  const title = kind === "original"
    ? `Original ${r.record.output} (extracted from PDF)`
    : `Replicated ${r.record.output} (agent output)`;
  document.querySelector(".app").classList.add("viewer-open");
  document.getElementById("vtitle").textContent = title;
  document.getElementById("vnewtab").href = "bundle/" + encodeURI(path);
  const body = document.getElementById("vbody");
  body.innerHTML = `<div class="viewer-msg">loading…</div>`;
  try {
    const resp = await fetch("bundle/" + encodeURI(path));
    if (!resp.ok) throw new Error(resp.status);
    const data = await resp.json();
    let tbl = data;
    if (kind === "original") {
      const target = normId(r.record.output);
      tbl = (data.tables || []).find(t => {
        const tid = normId(t.table_id);
        return tid && (tid === target || tid.includes(target) || target.includes(tid));
      });
      if (!tbl) {
        body.innerHTML = `<div class="viewer-msg">No extracted table matching
          "${esc(r.record.output)}" in original_results.json
          (available: ${esc((data.tables || []).map(t => t.table_id).join(", "))}).</div>`;
        return;
      }
    }
    body.innerHTML = renderTable(tbl, title, []);
  } catch (err) {
    body.innerHTML = `<div class="viewer-msg">Could not load table (${esc(err.message)}).</div>`;
  }
}

function renderTable(tbl, title, affected) {
  const cols = tbl.column_labels || [];
  const rows = tbl.row_labels || [];
  const grid = {};
  (tbl.cells || []).forEach(c => { grid[`${c.row_index},${c.col_index}`] = c; });
  const affSet = new Set(affected.map(a =>
    `${String(a.row_label || "").trim().toLowerCase()}|${String(a.column_label || "").trim().toLowerCase()}`));
  const isAff = (rl, cl) =>
    affSet.has(`${String(rl || "").trim().toLowerCase()}|${String(cl || "").trim().toLowerCase()}`);

  let html = `<div class="tbl-wrap"><h3>${esc(tbl.table_id || title)}</h3><table class="tbl"><thead><tr><th></th>`;
  cols.forEach(c => { html += `<th>${esc(c)}</th>`; });
  html += `</tr></thead><tbody>`;
  rows.forEach((rl, ri) => {
    html += `<tr><td>${esc(rl)}</td>`;
    cols.forEach((cl, ci) => {
      const c = grid[`${ri},${ci}`];
      html += `<td class="${isAff(rl, cl) ? "hl" : ""}">${esc(c ? c.raw_text : "")}</td>`;
    });
    html += `</tr>`;
  });
  html += `</tbody></table>`;
  if (affected.length) html += `<div class="tbl-notes">Highlighted cells = affected cells listed by the trace agent (matched by labels; duplicate labels highlight all occurrences).</div>`;
  if (tbl.significance_convention) html += `<div class="tbl-notes">${esc(tbl.significance_convention)}</div>`;
  if (tbl.notes) html += `<div class="tbl-notes">${esc(tbl.notes)}</div>`;
  html += `</div>`;
  return html;
}

function optGroup(id, field, options, current, disabled) {
  return `<div class="opts">` + options.map(([val, lab]) => `
    <label><input type="radio" name="${id}-${field}" value="${val}"
      ${current === val ? "checked" : ""} ${disabled ? "disabled" : ""}
      onchange="setAnno('${id}','${field}','${val}')"> ${lab}</label>`).join("") + `</div>`;
}

function render() {
  const r = RECORDS[currentIdx];
  const rec = r.record;
  const a = annoFor(r.audit_id);
  const notReal = a.real === "no";

  const cells = (rec.affected_cells || []).map(c =>
    `<li>${esc(c.item_id)} — row: <b>${esc(c.row_label || "—")}</b>, column: <b>${esc(c.column_label || "—")}</b></li>`).join("");
  const also = (rec.also_explains || []).map(x =>
    typeof x === "string" ? esc(x) : `${esc(x.item_id)} (${esc(x.sections)})`).join("; ");

  const origLoc = rec.original_location || {};
  const agentLoc = rec.agent_location || {};

  document.getElementById("main").innerHTML = `
  <div class="card">
    <div class="head-line">
      <h2>${esc(r.audit_id)} — ${esc(rec.output)}</h2>
    </div>
    <div class="head-sub">${esc(r.paper_slug)} &nbsp;·&nbsp; ${esc(r.approach)} &nbsp;·&nbsp; divergence #${esc(r.div_id)}</div>
    <div class="chips">
      <span class="chip cause" title="${esc(DEFS.rootCauses[r.root_cause] || "")}">Source: ${esc(r.root_cause)}</span>
      <span class="chip type" title="${esc(DEFS.types[rec.divergence_type] || "")}">Type: ${esc(rec.divergence_type)}</span>
      <span class="chip sev-${esc(rec.severity)}" title="${esc(DEFS.severities[rec.severity] || "")}">Severity: ${esc(rec.severity)}</span>
      <span class="chip">Data: ${esc(rec.data_available || "—")}</span>
    </div>
    <p class="desc">${esc(rec.description)}</p>
    <div class="kv">
      <span class="k">Affected output</span><span>${esc(rec.output)}</span>
      ${cells ? `<span class="k">Affected cells (${(rec.affected_cells || []).length})</span><ul class="cells-list">${cells}</ul>` : ""}
      ${(rec.explains_sections || []).length ? `<span class="k">Explains sections</span><span>${esc((rec.explains_sections || []).join("; "))}</span>` : ""}
      ${also ? `<span class="k">Also explains</span><span>${also}</span>` : ""}
    </div>

    <details>
      <summary>Evidence snippets (original vs. agent)</summary>
      <div class="body">
        <div class="loc"><b>Original behavior</b> — ${esc(origLoc.file || "—")}${origLoc.line ? ` : ${esc(origLoc.line)}` : ""}</div>
        <div>${esc(rec.original_behavior || "")}</div>
        ${rec.original_proof ? `<pre>${esc(rec.original_proof)}</pre>` : ""}
        <div class="loc"><b>Agent behavior</b> — ${esc(agentLoc.file || "—")}${agentLoc.line ? ` : ${esc(agentLoc.line)}` : ""}</div>
        <div>${esc(rec.agent_behavior || "")}</div>
        ${rec.agent_proof ? `<pre>${esc(rec.agent_proof)}</pre>` : ""}
        ${rec.data_available_note ? `<div class="loc"><b>Data availability</b> (${esc(rec.data_available)})</div><div>${esc(rec.data_available_note)}</div>` : ""}
      </div>
    </details>

    <details>
      <summary>Attribution check verdicts (how the error source was derived)</summary>
      <div class="body">
        ${checkBlock(rec, "paper_vs_original_code")}
        ${checkBlock(rec, "paper_vs_summary")}
        ${checkBlock(rec, "summary_vs_agent")}
        <div class="cb-note" style="margin-top:8px">The error-source label is derived mechanically from these verdicts
        (cascade: data missing → paper vs. code → paper vs. summary → summary vs. agent → unexplained).</div>
      </div>
    </details>
  </div>

  <div class="card">
    <div class="q-label" style="font-weight:600; margin-bottom:8px">Open inputs
      <span style="font-weight:400; color:var(--muted); font-size:12px">(inline — ⌘/ctrl-click for new tab)</span></div>
    <div class="links">
      ${r.assets.original_results ? `<a href="bundle/${encodeURI(r.assets.original_results)}" target="_blank"
          onclick="return tableClick(event, this, 'original')">📊 Original table</a>` : ""}
      ${r.assets.replicated_table ? `<a href="bundle/${encodeURI(r.assets.replicated_table)}" target="_blank"
          onclick="return tableClick(event, this, 'replicated')">📊 Replicated table</a>`
        : `<span class="missing-note">no replicated table file (agent did not produce ${esc(rec.output)})</span>`}
      ${docLink(r.assets.paper_pdf, "📄 Paper PDF")}
      ${docLink(r.assets.summary, "📋 Methodology summary")}
      ${fileLinks(r.assets.original_files, "🧾 Original", origLoc.line)}
      ${docLink(r.assets.original_code_dir, "🗂 Original code dir")}
      ${fileLinks(r.assets.agent_files, "🤖 Agent", agentLoc.line)}
      ${docLink(r.assets.agent_code_dir, "🗂 Agent code dir")}
      ${!(r.assets.original_files || []).length ? `<span class="missing-note">original script link unavailable — location: "${esc(origLoc.file || "—")}"</span>` : ""}
    </div>
  </div>

  <div class="card anno">
    <div class="q-label" style="font-size:15px">Validation</div>
    <div class="q">
      <div class="q-label">1. Is this an actual, meaningful divergence / error?</div>
      <div class="q-context">Does the described difference between original and agent implementation really exist, based on the evidence?</div>
      ${optGroup(r.audit_id, "real", [["yes", "Yes"], ["no", "No"], ["unsure", "Unsure"]], a.real)}
    </div>
    <div class="q ${notReal ? "disabled" : ""}">
      <div class="q-label">2. Error source classification correct?</div>
      <div class="q-context">Assigned: <b>${esc(r.root_cause)}</b> — ${esc(DEFS.rootCauses[r.root_cause] || "")}</div>
      ${optGroup(r.audit_id, "source", [["correct", "Correct"], ["incorrect", "Incorrect"], ["unsure", "Unsure"]], a.source, notReal)}
    </div>
    <div class="q ${notReal ? "disabled" : ""}">
      <div class="q-label">3. Divergence type correct?</div>
      <div class="q-context">Assigned: <b>${esc(rec.divergence_type)}</b> — ${esc(DEFS.types[rec.divergence_type] || "")}</div>
      ${optGroup(r.audit_id, "type", [["correct", "Correct"], ["incorrect", "Incorrect"], ["unsure", "Unsure"]], a.type, notReal)}
    </div>
    <div class="q ${notReal ? "disabled" : ""}">
      <div class="q-label">4. Severity correct?</div>
      <div class="q-context">Assigned: <b>${esc(rec.severity)}</b> — ${esc(DEFS.severities[rec.severity] || "")}</div>
      ${optGroup(r.audit_id, "severity", [["correct", "Correct"], ["incorrect", "Incorrect"], ["unsure", "Unsure"]], a.severity, notReal)}
    </div>
    <div class="q">
      <div class="q-label">Notes (optional — e.g. what the correct label would be)</div>
      <textarea id="notes" onchange="setAnno('${r.audit_id}','notes',this.value)">${esc(a.notes)}</textarea>
    </div>
  </div>

  <div class="nav-row">
    <button onclick="nav(-1)">← Previous</button>
    <button class="primary" onclick="nav(1)">Next →</button>
    <span class="saved-note" id="savedNote"></span>
    <span class="spacer"></span>
    <span class="q-context">navigate: <kbd>←</kbd> <kbd>→</kbd></span>
  </div>`;

  renderList();
}

function setAnno(id, field, val) {
  const a = annoFor(id);
  a[field] = val;
  a.ts = new Date().toISOString();
  save();
  if (field === "real") render();
  else renderList();
  const n = document.getElementById("savedNote");
  if (n) { n.textContent = "saved"; setTimeout(() => { if (n) n.textContent = ""; }, 1200); }
}

function nav(d) {
  const next = Math.min(RECORDS.length - 1, Math.max(0, currentIdx + d));
  if (next !== currentIdx) closeViewer();
  currentIdx = next;
  render();
  document.getElementById("main").scrollTop = 0;
}

/* ---------------- toolbar ---------------- */
document.getElementById("annotator").value = state.annotator || "";
document.getElementById("annotator").onchange = e => { state.annotator = e.target.value; save(); };

document.getElementById("exportJson").onclick = () => {
  download(`audit_annotations_${(state.annotator || "anon").replace(/\W+/g, "_")}.json`,
           JSON.stringify(state, null, 2), "application/json");
};
document.getElementById("exportCsv").onclick = () => {
  const cols = ["audit_id", "paper_slug", "agent_label", "approach", "div_id", "output",
                "root_cause", "divergence_type", "severity",
                "annotator", "q_real", "q_source", "q_type", "q_severity", "notes", "ts"];
  const rows = [cols.join(",")];
  RECORDS.forEach(r => {
    const a = state.annotations[r.audit_id] || {};
    const vals = [r.audit_id, r.paper_slug, r.agent_label, r.approach, r.div_id, r.record.output,
                  r.root_cause, r.record.divergence_type, r.record.severity,
                  state.annotator, a.real || "", a.source || "", a.type || "", a.severity || "", a.notes || "", a.ts || ""];
    rows.push(vals.map(v => `"${String(v ?? "").replace(/"/g, '""')}"`).join(","));
  });
  download(`audit_annotations_${(state.annotator || "anon").replace(/\W+/g, "_")}.csv`,
           rows.join("\n"), "text/csv");
};
document.getElementById("importJson").onclick = () => document.getElementById("importFile").click();
document.getElementById("importFile").onchange = e => {
  const f = e.target.files[0];
  if (!f) return;
  f.text().then(t => {
    const imported = JSON.parse(t);
    Object.assign(state.annotations, imported.annotations || {});
    if (imported.annotator && !state.annotator) state.annotator = imported.annotator;
    save(); render();
    alert(`Imported ${Object.keys(imported.annotations || {}).length} annotations (merged).`);
  });
};

document.querySelectorAll(".filters button").forEach(b => {
  b.onclick = () => {
    document.querySelectorAll(".filters button").forEach(x => x.classList.remove("active"));
    b.classList.add("active");
    filter = b.dataset.f;
    renderList();
  };
});

document.addEventListener("keydown", e => {
  if (e.target.tagName === "TEXTAREA" || e.target.tagName === "INPUT") return;
  if (e.key === "ArrowLeft") nav(-1);
  if (e.key === "ArrowRight") nav(1);
  if (e.key === "Escape") closeViewer();
});

function download(name, content, mime) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([content], { type: mime }));
  a.download = name;
  a.click();
}

render();
</script>
</body>
</html>
"""


def main():
    data = json.loads((BUNDLE / "records.json").read_text())
    OUT.write_text(build_html(data))
    print(f"Wrote {OUT} ({OUT.stat().st_size/1024:.0f} KB, {len(data['records'])} records)")
    print("Serve with:  cd human_audit && python3 -m http.server 8765")


if __name__ == "__main__":
    main()
