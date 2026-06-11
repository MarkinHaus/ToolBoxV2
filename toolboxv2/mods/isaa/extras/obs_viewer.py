"""
obs_viewer.py - ISAA Agent Observability Viewer V2

Generates a standalone HTML file with embedded JSON data.
TBJS Terminal style — mono, dark, phosphor, zero decoration.

Usage:
    from toolboxv2.mods.isaa.extras.obs_viewer import generate_viewer
    generate_viewer(agent=my_agent, output="obs_report.html")
    generate_viewer(obs_dir="/path/to/Agents/MyAgent/obs", output="obs_report.html")
    generate_viewer(runs=[run1.to_dict()], output="obs_report.html")
"""

from __future__ import annotations

import json
import os
from typing import Any


def generate_viewer(
    agent: Any = None,
    obs_dir: str | None = None,
    runs: list[dict] | None = None,
    agent_name: str = "unknown",
    output: str = "obs_viewer.html",
) -> str:
    run_data = []
    interrupted_data = []

    if agent is not None:
        obs = getattr(agent, 'obs', None)
        if obs is None:
            raise ValueError("Agent has no .obs (ObservabilityLayer)")
        agent_name = obs.agent_name
        obs_dir = obs.obs_dir

    if obs_dir is not None:
        agent_name = agent_name or os.path.basename(os.path.dirname(obs_dir))
        index_path = os.path.join(obs_dir, "_index.json")
        if os.path.exists(index_path):
            with open(index_path, "r", encoding="utf-8") as f:
                index = json.load(f)
            for run_id in index.get("runs", []):
                run_path = os.path.join(obs_dir, f"run_{run_id}.json")
                if os.path.exists(run_path):
                    with open(run_path, "r", encoding="utf-8") as f:
                        run_data.append(json.load(f))

        if os.path.isdir(obs_dir):
            for fname in os.listdir(obs_dir):
                if fname.startswith("live_") and fname.endswith(".jsonl"):
                    rid = fname.replace("live_", "").replace(".jsonl", "")
                    if any(r.get("run_id") == rid for r in run_data):
                        continue
                    live_path = os.path.join(obs_dir, fname)
                    steps = []
                    with open(live_path, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                try:
                                    steps.append(json.loads(line))
                                except json.JSONDecodeError:
                                    continue
                    if steps:
                        interrupted_data.append({
                            "run_id": rid, "steps": steps, "status": "interrupted",
                        })

    elif runs is not None:
        run_data = runs

    payload = {
        "agent_name": agent_name,
        "runs": run_data,
        "interrupted": interrupted_data,
        "generated_at": __import__("time").time(),
    }

    html = _TEMPLATE.replace("__OBS_DATA_PLACEHOLDER__", json.dumps(payload, ensure_ascii=False))
    with open(output, "w", encoding="utf-8") as f:
        f.write(html)
    return os.path.abspath(output)


_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ISAA OBS</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&display=swap');
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --primary:oklch(55% 0.18 230);--success:oklch(65% 0.2 145);
  --warning:oklch(75% 0.18 85);--error:oklch(55% 0.22 25);
  --info:oklch(60% 0.15 230);
  --bg:#000;--bg1:#0a0a0f;--bg2:#030306;
  --bd:rgba(255,255,255,0.12);--sel:color-mix(in oklch,var(--primary) 30%,transparent);
  --fg:rgba(255,255,255,0.92);--fg1:rgba(255,255,255,0.5);
  --fg2:rgba(255,255,255,0.3);--acc:var(--primary);
  --mono:'IBM Plex Mono',ui-monospace,'SF Mono',Consolas,monospace;
}
html{background:var(--bg);color:var(--fg)}
body{font-family:var(--mono);font-size:12px;line-height:1.4;min-height:100vh;padding-bottom:32px}

/* NAV + TABS */
.nav{padding:8px 16px;border-bottom:1px solid var(--bd);display:flex;align-items:center;gap:16px;background:var(--bg);position:sticky;top:0;z-index:100}
.nav-title{color:var(--primary);font-size:13px;font-weight:600;letter-spacing:1px}
.nav-sep{color:var(--fg2)}
.tabs{display:flex;border-bottom:1px solid var(--bd);background:var(--bg);position:sticky;top:29px;z-index:99}
.tab{padding:8px 16px;color:var(--fg1);background:0;border:none;border-bottom:2px solid transparent;margin-bottom:-1px;cursor:pointer;font-family:var(--mono);font-size:11px;transition:all 100ms linear}
.tab:hover{color:var(--fg)}.tab.active{color:var(--primary);border-bottom-color:var(--primary)}

.view{display:none;padding:16px}.view.active{display:block}

/* STATS */
.stats-row{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:8px;margin-bottom:16px}
.stat{background:var(--bg1);border:1px solid var(--bd);padding:10px 14px}
.stat-l{font-size:10px;color:var(--fg2);text-transform:uppercase;letter-spacing:1px;margin-bottom:4px}
.stat-v{font-size:18px;font-weight:600}.stat-v.acc{color:var(--primary)}
.stat-s{font-size:10px;color:var(--fg1);margin-top:2px}

/* RUN CARDS */
.run-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:8px}
.rc{background:var(--bg1);border:1px solid var(--bd);padding:12px 14px;cursor:pointer;transition:border-color 100ms linear}
.rc:hover{border-color:var(--primary)}
.rc.int{border-left:3px solid var(--warning)}
.rc-h{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px}
.rc-id{color:var(--primary);font-weight:500;font-size:12px}
.rc-st{font-size:10px}.rc-st.ok{color:var(--success)}.rc-st.fail{color:var(--error)}.rc-st.int{color:var(--warning)}
.rc-q{color:var(--fg1);font-size:11px;margin-bottom:8px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.rc-m{display:flex;flex-wrap:wrap;gap:12px;font-size:10px;color:var(--fg2)}
.rc-m b{color:var(--fg1);font-weight:500}

/* SECTION LABEL */
.sec-label{font-size:10px;color:var(--fg2);text-transform:uppercase;letter-spacing:1px;margin-bottom:10px}

/* REPLAY */
.replay-back{color:var(--primary);cursor:pointer;background:0;border:0;font-family:var(--mono);font-size:11px;padding:4px 0;margin-bottom:8px}
.replay-back:hover{text-decoration:underline}
.kv{display:inline-block;margin-right:12px;font-size:11px}
.kv .k{color:var(--fg2)}.kv .v{color:var(--fg)}

/* STEP TIMELINE */
.step-tl{position:relative;padding-left:20px}
.step-tl::before{content:'';position:absolute;left:6px;top:0;bottom:0;width:1px;background:var(--bd)}
.step{position:relative;margin-bottom:12px;padding:10px 14px;background:var(--bg1);border:1px solid var(--bd)}
.step::before{content:'';position:absolute;left:-17px;top:14px;width:7px;height:7px;background:var(--primary)}
.step.err::before{background:var(--error)}
.step.warn::before{background:var(--warning)}
.step-h{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px}
.step-id{color:var(--primary);font-weight:500;font-size:12px}
.step-dur{font-size:10px;color:var(--fg1)}
.step-sec{margin-top:8px;padding-top:6px;border-top:1px solid var(--bd)}
.step-sec-l{font-size:10px;color:var(--fg2);text-transform:uppercase;letter-spacing:1px;margin-bottom:4px}

/* TIMING BARS in steps */
.t-bar-row{display:flex;align-items:center;gap:6px;margin:3px 0;font-size:10px}
.t-bar-label{width:70px;text-align:right;color:var(--fg2);flex-shrink:0}
.t-bar-track{flex:1;height:12px;background:var(--bg2);border:1px solid rgba(255,255,255,0.05);position:relative;overflow:hidden}
.t-bar-seg{position:absolute;height:100%;min-width:1px}
.t-bar-seg.llm{background:var(--primary);opacity:.85}
.t-bar-seg.tool{background:var(--success);opacity:.85}
.t-bar-seg.pre{background:var(--warning);opacity:.6}
.t-bar-seg.post{background:var(--error);opacity:.4}
.t-bar-val{width:70px;font-size:10px;color:var(--fg2);flex-shrink:0}

.tool-row,.vfs-row{font-size:11px;padding:3px 0;color:var(--fg1);display:grid;gap:8px}
.tool-row{grid-template-columns:160px 80px 60px 1fr}
.vfs-row{grid-template-columns:1fr 80px 60px 60px}
.tool-row.terr{color:var(--error)}
.file-path{color:var(--primary)}.file-d{color:var(--success)}.file-d.neg{color:var(--error)}

/* TABLES */
.tbl{width:100%;border-collapse:collapse;font-size:11px}
.tbl th{text-align:left;padding:4px 10px;font-size:10px;color:var(--fg2);text-transform:uppercase;letter-spacing:1px;border-bottom:1px solid var(--bd);font-weight:500}
.tbl td{padding:4px 10px;border-bottom:1px solid rgba(255,255,255,0.05);color:var(--fg1)}
.tbl tr:hover td{background:var(--sel)}
.tbl .n{text-align:right;font-variant-numeric:tabular-nums}

/* BARS */
.bar-chart{margin-bottom:12px}
.bar-r{display:flex;align-items:center;gap:8px;margin-bottom:4px;font-size:11px}
.bar-l{width:100px;text-align:right;color:var(--fg1);flex-shrink:0}
.bar-t{flex:1;height:14px;background:var(--bg2);border:1px solid var(--bd);position:relative;overflow:hidden}
.bar-f{height:100%;transition:width 300ms linear}
.bar-f.llm{background:var(--primary)}.bar-f.tool{background:var(--success)}.bar-f.other{background:var(--fg2)}
.bar-f.pre{background:var(--warning)}.bar-f.post{background:var(--error);opacity:.6}
.bar-v{width:100px;text-align:right;color:var(--fg2);font-size:10px;flex-shrink:0}

.legend{display:flex;gap:16px;font-size:10px;color:var(--fg2);margin-bottom:8px}
.legend-i{display:flex;align-items:center;gap:4px}
.legend-s{width:10px;height:10px;display:inline-block}

/* WATERFALL */
.wf-row{display:flex;align-items:center;height:20px;margin-bottom:2px;font-size:10px}
.wf-label{width:70px;text-align:right;color:var(--fg1);padding-right:8px;flex-shrink:0}
.wf-track{flex:1;height:14px;position:relative;background:var(--bg2);border:1px solid rgba(255,255,255,0.05)}
.wf-seg{position:absolute;height:100%;min-width:1px}
.wf-seg.llm{background:var(--primary);opacity:.85}
.wf-seg.tool{background:var(--success);opacity:.85}
.wf-seg.pre{background:var(--warning);opacity:.5}
.wf-seg.post{background:var(--error);opacity:.35}

/* CALC */
.calc-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:12px}
.calc-card{background:var(--bg1);border:1px solid var(--bd);padding:12px 14px}
.calc-card-t{font-size:10px;color:var(--fg2);text-transform:uppercase;letter-spacing:1px;margin-bottom:8px}
.slider-r{display:flex;align-items:center;gap:8px;margin:8px 0}
.slider-r label{font-size:11px;color:var(--fg1);width:120px}
.slider-r input[type=range]{flex:1;accent-color:var(--primary);height:4px}
.slider-r .val{width:60px;text-align:right;font-size:11px;color:var(--primary)}
.calc-res{margin-top:10px;padding:8px 10px;background:var(--bg2);border:1px solid var(--bd);font-size:11px;color:var(--fg1)}
.calc-res .hi{color:var(--success);font-weight:600}

.time-sec{margin-bottom:24px}
.time-sec-t{font-size:13px;color:var(--fg1);margin-bottom:10px;padding-bottom:6px;border-bottom:1px solid var(--bd)}
.time-sec-t::before{content:"## ";color:var(--fg2)}

/* WARNING BOX */
.warn-box{background:var(--bg1);border:1px solid var(--warning);border-left:3px solid var(--warning);padding:8px 12px;margin-bottom:12px;font-size:11px;color:var(--fg1)}
.warn-box .wt{color:var(--warning);font-size:10px;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px}

/* COLLAPSE CONTROLS */
.collapse-bar{display:flex;gap:6px;margin-bottom:12px}
.collapse-btn{background:var(--bg1);border:1px solid var(--bd);color:var(--fg1);font-family:var(--mono);font-size:10px;padding:4px 10px;cursor:pointer;text-transform:uppercase;letter-spacing:1px;transition:border-color 100ms linear}
.collapse-btn:hover{border-color:var(--primary);color:var(--fg)}
.step.minimized>.step-body{display:none}
.step.minimized{padding:6px 14px;margin-bottom:4px}
.step-hdr{cursor:pointer;user-select:none}
.step-hdr .chevron{color:var(--fg2);font-size:10px;margin-right:6px;display:inline-block;transition:transform 80ms linear}
.step.minimized .step-hdr .chevron{transform:rotate(-90deg)}

/* LLM DETAIL PANEL */
.llm-toggle{cursor:pointer;transition:background 100ms linear;margin:-2px -4px;padding:2px 4px}
.llm-toggle:hover{background:rgba(255,255,255,0.04)}
.llm-detail{display:none;margin-top:8px;background:var(--bg2);border:1px solid var(--bd);max-height:500px;overflow:auto}
.llm-detail.open{display:block}
.llm-detail-tab-bar{display:flex;border-bottom:1px solid var(--bd);background:var(--bg1)}
.llm-detail-tab{padding:4px 12px;font-size:10px;color:var(--fg2);cursor:pointer;border:none;background:0;font-family:var(--mono);border-bottom:2px solid transparent}
.llm-detail-tab.active{color:var(--primary);border-bottom-color:var(--primary)}
.llm-detail-content{padding:10px 12px;font-size:11px;white-space:pre-wrap;word-break:break-word;color:var(--fg1);line-height:1.5}
.llm-detail-content .msg{margin-bottom:8px;padding:6px 8px;border-left:2px solid var(--bd)}
.llm-detail-content .msg.role-system{border-left-color:var(--warning)}
.llm-detail-content .msg.role-user{border-left-color:var(--primary)}
.llm-detail-content .msg.role-assistant{border-left-color:var(--success)}
.llm-detail-content .msg.role-tool{border-left-color:var(--info)}
.llm-detail-content .msg-role{font-size:10px;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px;font-weight:600}
.llm-detail-content .msg-role.system{color:var(--warning)}
.llm-detail-content .msg-role.user{color:var(--primary)}
.llm-detail-content .msg-role.assistant{color:var(--success)}
.llm-detail-content .msg-role.tool{color:var(--info)}

/* TOOL DETAIL PANEL */
.tool-row-wrap{border-bottom:1px solid rgba(255,255,255,0.03)}
.tool-row{cursor:pointer}.tool-row:hover{background:rgba(255,255,255,0.03)}
.tool-detail{display:none;padding:6px 12px 8px;font-size:11px;background:var(--bg2);border:1px solid var(--bd);margin:2px 0 6px}
.tool-detail.open{display:block}
.tool-detail pre{white-space:pre-wrap;word-break:break-word;color:var(--fg1);line-height:1.4;max-height:300px;overflow:auto;margin:0}
.tool-detail-l{font-size:10px;color:var(--fg2);text-transform:uppercase;letter-spacing:1px;margin:4px 0 2px}

::-webkit-scrollbar{width:6px;height:6px}::-webkit-scrollbar-track{background:var(--bg)}::-webkit-scrollbar-thumb{background:rgba(255,255,255,0.15)}

@media(max-width:640px){.stats-row{grid-template-columns:repeat(2,1fr)}.run-grid{grid-template-columns:1fr}.calc-grid{grid-template-columns:1fr}}
/* Sub-agent nested rendering */
.subagent-wrap{margin:4px 0 6px;border-left:2px solid var(--primary);padding-left:8px}
.subagent-row{font-size:11px;padding:4px 6px;cursor:pointer;display:flex;gap:8px;align-items:center;background:rgba(120,120,255,0.05);border-radius:3px}
.subagent-row:hover{background:rgba(120,120,255,0.10)}
.sa-badge{color:var(--primary);font-weight:600;white-space:nowrap}
.sa-status{white-space:nowrap;font-size:10px;padding:1px 6px;border-radius:3px}
.sa-completed{color:var(--success)}
.sa-failed{color:var(--error)}
.sa-running,.sa-unknown{color:var(--warning)}
.sa-task{flex:1;color:var(--fg1);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.sa-meta{color:var(--fg2);font-size:10px;white-space:nowrap}
.subagent-detail{display:none;padding:4px 0 4px 8px;margin-top:4px;border-top:1px dashed var(--bd)}
.substep{margin:4px 0;padding:4px 6px;background:var(--bg2);border-radius:3px}
.substep-hdr{font-size:10px;color:var(--fg2);margin-bottom:2px}
.substep-tool{font-size:11px;display:grid;grid-template-columns:1fr 70px;gap:8px;padding:2px 0;color:var(--fg1)}
.substep-tool.terr{color:var(--error)}
.sa-missing{font-size:11px;color:var(--warning);padding:6px}
.msg-content{white-space:pre-wrap;word-break:break-word}
</style>
</head>
<body>

<div class="nav">
  <span class="nav-title">ISAA OBS</span>
  <span class="nav-sep">/</span>
  <span id="nav-agent" style="color:var(--fg)">agent</span>
  <span class="nav-sep">/</span>
  <span id="nav-crumb" style="color:var(--fg1)">dashboard</span>
</div>
<div class="tabs">
  <button class="tab active" data-view="dashboard">> dashboard</button>
  <button class="tab" data-view="replay">session replay</button>
  <button class="tab" data-view="timing">time analysis</button>
</div>

<div id="dashboard" class="view active"></div>
<div id="replay" class="view"><div id="replay-c"><div style="color:var(--fg1);font-size:11px">Select a run from dashboard.</div></div></div>
<div id="timing" class="view"></div>

<script>
const DATA = __OBS_DATA_PLACEHOLDER__;
const runs = (DATA.runs||[]).map(r=>({...r,_src:'complete'}));
const interrupted = (DATA.interrupted||[]).map(r=>({...r,_src:'interrupted'}));
const allRuns = [...runs,...interrupted];
document.getElementById('nav-agent').textContent = DATA.agent_name||'unknown';

// ── Sub-agent lineage helpers ──
const RUN_BY_ID = {};
allRuns.forEach(r => { if (r.run_id) RUN_BY_ID[r.run_id] = r; });
const SPAWN_TOOLS = new Set(['spawn_sub_agent','wait_for','resume_sub_agent']);
// Match a spawn-type tool call to the next unconsumed sub_agent_runs entry of
// its parent run. The obs layer records sub_agent_runs in spawn order, so we
// walk them positionally per render.
function makeSubMatcher(run){
  const queue = (run && run.sub_agent_runs ? run.sub_agent_runs.slice() : []);
  return function(toolName){
    if (!SPAWN_TOOLS.has(toolName)) return null;
    return queue.length ? queue.shift() : null;
  };
}

// ── Utils ──
const F = {
  dur(s){if(s==null||s===0) return '—'; return s<1?`${(s*1000).toFixed(0)}ms`:`${s.toFixed(2)}s`},
  durX(s){if(s==null) return '—'; if(s===0) return '0'; return s<1?`${(s*1000).toFixed(0)}ms`:`${s.toFixed(2)}s`},
  ts(t){return t?new Date(t*1000).toLocaleString():'—'},
  tok(n){return n==null?'—':n>=1000?`${(n/1000).toFixed(1)}k`:String(n)},
  pct(v,t){return t>0?`${((v/t)*100).toFixed(1)}%`:'0%'},
};
function el(t,c,h){const e=document.createElement(t);if(c)e.className=c;if(h!==undefined)e.innerHTML=h;return e}

// ── Render a linked sub-agent run, collapsed under its spawn tool-call ──
function renderSubAgentRun(subEntry, depth){
  if (!subEntry) return '';
  const rid = subEntry.run_id || '';
  const sub = RUN_BY_ID[rid];
  const status = subEntry.status || (sub ? (sub.success ? 'completed' : 'failed') : 'unknown');
  const task = esc((subEntry.task || (sub && sub.query) || '').substring(0,100));
  const subId = 'subrun-'+rid;
  const statusIcon = status==='completed' ? '✓' : (status==='failed' ? '✕' : '◷');
  const nSteps = sub ? (sub.steps||[]).length : 0;
  const nTools = sub ? (sub.tool_call_count||0) : 0;
  let h = `<div class="subagent-wrap" style="margin-left:${depth*8}px">`;
  h += `<div class="subagent-row" onclick="event.stopPropagation();toggleDetail('${subId}')" title="click to expand sub-agent">`;
  h += `<span class="sa-badge">⊕ sub-agent</span><span class="sa-status sa-${status}">${statusIcon} ${status}</span>`;
  h += `<span class="sa-task">${task||'(no task)'}</span>`;
  h += sub ? `<span class="sa-meta">${nSteps} steps · ${nTools} tools · ${F.dur(sub.duration_s)}</span>`
           : `<span class="sa-meta">run ${rid} not persisted</span>`;
  h += `</div>`;
  h += `<div class="subagent-detail" id="${subId}">`;
  if (sub) {
    const sra = analyzeRun(sub);
    sra.analyzed.forEach(({step, a}) => {
      h += renderSubStep(sub, step, a, depth+1);
    });
  } else {
    h += `<div class="sa-missing">Sub-agent run ${esc(rid)} is not in this snapshot. Open the live obs viewer for its full trajectory.</div>`;
  }
  h += `</div></div>`;
  return h;
}

// Compact step render for nested sub-agent steps (tools + recursive sub-agents)
function renderSubStep(run, step, a, depth){
  let h = `<div class="substep">`;
  h += `<div class="substep-hdr">step ${step.step_id} · ${F.dur(a.total)}</div>`;
  const matchSub = makeSubMatcher(run);
  if (step.tool_calls?.length){
    step.tool_calls.forEach(tc => {
      const isErr = tc.status==='error';
      h += `<div class="substep-tool${isErr?' terr':''}"><span>${isErr?'✕':'▸'} ${esc(tc.name)}</span><span>${F.dur(tc.duration_s)}</span></div>`;
      const childSub = matchSub(tc.name);
      if (childSub) h += renderSubAgentRun(childSub, depth+1);
    });
  }
  h += `</div>`;
  return h;
}

// ── Deep step analysis ──
function analyzeStep(step, runStart) {
  const d = step.duration_s || 0;
  const llm = step.llm;
  let llmDur = 0, preLlm = 0, postLlm = 0, toolDur = 0;

  if (llm && llm.t_start && step.t_start) {
    preLlm = Math.max(0, llm.t_start - step.t_start);
    llmDur = llm.duration_s || 0;
    const llmEnd = llm.t_end || (llm.t_start + llmDur);
    // Tool time after LLM
    (step.tool_calls||[]).forEach(tc => { toolDur += tc.duration_s || 0; });
    postLlm = Math.max(0, d - preLlm - llmDur - toolDur);
  } else {
    // No LLM data
    (step.tool_calls||[]).forEach(tc => { toolDur += tc.duration_s || 0; });
    postLlm = Math.max(0, d - toolDur);
  }

  return {
    total: d, llm: llmDur, tool: toolDur,
    pre: preLlm, post: postLlm,
    ttft: llm?.ttft_s || 0,
    ttftMissing: llm && (!llm.t_first_token || llm.t_first_token === 0),
    tokIn: llm?.tokens_in || 0, tokOut: llm?.tokens_out || 0,
    tokSec: llm?.tokens_per_sec || 0,
    model: llm?.model || '',
    hasError: (step.tool_calls||[]).some(tc => tc.status==='error'),
  };
}

function analyzeRun(r) {
  const steps = r.steps || [];
  let totalLlm=0, totalTool=0, totalPre=0, totalPost=0;
  let ttfts=[], models=new Set();
  const analyzed = steps.map(s => {
    const a = analyzeStep(s);
    totalLlm += a.llm; totalTool += a.tool;
    totalPre += a.pre; totalPost += a.post;
    if (a.ttft > 0) ttfts.push(a.ttft);
    if (a.model) models.add(a.model);
    return { step: s, a };
  });
  const totalDur = r.duration_s || 0;

  // Init = time before first step started (run.t_start → step1.t_start)
  const initTime = (steps.length && r.t_start && steps[0].t_start)
    ? Math.max(0, steps[0].t_start - r.t_start) : 0;

  // End = time after last step ended (stepN.t_end → run.t_end) — finalize/bg learning
  const endTime = (steps.length && r.t_end && steps[steps.length-1].t_end)
    ? Math.max(0, r.t_end - steps[steps.length-1].t_end) : 0;

  // Inter-step = gaps between consecutive steps
  let interStep = 0;
  for (let i = 1; i < steps.length; i++) {
    if (steps[i].t_start && steps[i-1].t_end) {
      interStep += Math.max(0, steps[i].t_start - steps[i-1].t_end);
    }
  }

  const avgTtft = ttfts.length ? ttfts.reduce((a,b)=>a+b,0)/ttfts.length : 0;
  return {
    analyzed, totalDur, totalLlm, totalTool, totalPre, totalPost,
    initTime, endTime, interStep,
    avgTtft, ttftCount: ttfts.length, ttftTotal: steps.length,
    models: [...models],
  };
}

// ── Tabs ──
document.querySelectorAll('.tab').forEach(t=>{
  t.addEventListener('click',()=>{
    document.querySelectorAll('.tab').forEach(x=>{x.classList.remove('active');x.textContent=x.dataset.view});
    document.querySelectorAll('.view').forEach(x=>x.classList.remove('active'));
    t.classList.add('active'); t.textContent='> '+t.dataset.view;
    document.getElementById(t.dataset.view).classList.add('active');
    document.getElementById('nav-crumb').textContent=t.dataset.view;
  });
});

// ══════════════════════════════════════════════════════════════════
// VIEW 1: DASHBOARD
// ══════════════════════════════════════════════════════════════════
function buildDash() {
  const root = document.getElementById('dashboard');
  let totalTokIn=0,totalTokOut=0,totalDur=0,totalToolCalls=0,totalIter=0,successN=0;
  let allTtft=[],allModels=new Set();

  runs.forEach(r=>{
    const a = analyzeRun(r);
    totalTokIn+=r.total_tokens_in||0; totalTokOut+=r.total_tokens_out||0;
    totalDur+=r.duration_s||0; totalToolCalls+=r.tool_call_count||0;
    totalIter+=r.total_iterations||0; if(r.success) successN++;
    if(a.avgTtft>0) allTtft.push(a.avgTtft);
    a.models.forEach(m=>allModels.add(m));
  });

  const avgTtft = allTtft.length ? allTtft.reduce((a,b)=>a+b,0)/allTtft.length : 0;
  const avgDur = runs.length ? totalDur/runs.length : 0;

  const stats = [
    {l:'RUNS',v:runs.length,s:`${interrupted.length} interrupted`},
    {l:'SUCCESS',v:runs.length?F.pct(successN,runs.length):'—',a:true},
    {l:'TOKENS',v:F.tok(totalTokIn+totalTokOut),s:`in:${F.tok(totalTokIn)} out:${F.tok(totalTokOut)}`},
    {l:'AVG DURATION',v:F.dur(avgDur),s:`total: ${F.dur(totalDur)}`},
    {l:'TOOL CALLS',v:totalToolCalls||'0',s:`${totalIter} iterations total`},
    {l:'AVG TTFT',v:F.dur(avgTtft),s:`${allTtft.length}/${runs.length} runs measured`},
  ];

  let h = '<div class="stats-row">';
  stats.forEach(s=>{h+=`<div class="stat"><div class="stat-l">${s.l}</div><div class="stat-v${s.a?' acc':''}">${s.v}</div>${s.s?`<div class="stat-s">${s.s}</div>`:''}</div>`});
  h += '</div>';

  if (allModels.size) {
    h += `<div class="sec-label" style="margin-top:8px">Models Used</div>`;
    h += `<div style="font-size:11px;color:var(--fg1);margin-bottom:16px">${[...allModels].join(' │ ')}</div>`;
  }

  // Tool calls = 0 warning
  if (totalToolCalls === 0 && totalIter > 0) {
    h += `<div class="warn-box"><div class="wt">▲ data gap</div>tool_call_count = 0 across all runs. record_tool_start/end hooks likely not wired in ExecutionEngine. Tool execution time is hidden inside "pre-LLM" and "post-LLM" segments.</div>`;
  }

  h += '<div class="sec-label">Runs</div><div class="run-grid" id="run-grid"></div>';
  root.innerHTML = h;

  const grid = document.getElementById('run-grid');
  allRuns.forEach(r=>{
    const isInt = r._src==='interrupted';
    const status = isInt?'int':(r.success?'ok':'fail');
    const glyph = isInt?'◐':(r.success?'●':'✕');
    const nSteps = (r.steps||[]).length;
    const card = el('div',`rc${isInt?' int':''}`);
    card.innerHTML=`<div class="rc-h"><span class="rc-id">${r.run_id||'?'}</span><span class="rc-st ${status}">${glyph} ${status}</span></div>
      <div class="rc-q">${(r.query||'—').substring(0,80)}</div>
      <div class="rc-m"><span><b>${F.dur(r.duration_s)}</b></span><span>${nSteps} steps</span>
      <span>${F.tok((r.total_tokens_in||0)+(r.total_tokens_out||0))} tok</span>
      <span>${r.tool_call_count||0} tools</span>
      ${r.persona?`<span>${r.persona}</span>`:''}</div>`;
    card.addEventListener('click',()=>openReplay(r));
    grid.appendChild(card);
  });
}

// ══════════════════════════════════════════════════════════════════
// VIEW 2: SESSION REPLAY
// ══════════════════════════════════════════════════════════════════
function openReplay(run) {
  document.querySelectorAll('.tab').forEach(t=>{t.classList.toggle('active',t.dataset.view==='replay');t.textContent=t.dataset.view==='replay'?'> session replay':t.dataset.view});
  document.querySelectorAll('.view').forEach(v=>v.classList.toggle('active',v.id==='replay'));
  document.getElementById('nav-crumb').textContent=`replay / ${run.run_id}`;

  const c = document.getElementById('replay-c');
  const ra = analyzeRun(run);
  const matchSub = makeSubMatcher(run);
  const steps = run.steps||[];

  let h = `<div style="padding:12px 0;border-bottom:1px solid var(--bd);margin-bottom:16px">
    <button class="replay-back" onclick="backToDash()">← dashboard</button>
    <div style="display:flex;gap:16px;flex-wrap:wrap;margin-top:6px">
      <span class="kv"><span class="k">run </span><span class="v" style="color:var(--primary)">${run.run_id}</span></span>
      <span class="kv"><span class="k">query </span><span class="v">${(run.query||'—').substring(0,120)}</span></span>
      <span class="kv"><span class="k">duration </span><span class="v">${F.dur(run.duration_s)}</span></span>
      <span class="kv"><span class="k">model </span><span class="v">${ra.models.join(', ')||'?'}</span></span>
      <span class="kv"><span class="k">persona </span><span class="v">${run.persona||'default'}</span></span>
      ${run.skills_matched?.length?`<span class="kv"><span class="k">skills </span><span class="v">${run.skills_matched.join(', ')}</span></span>`:''}
    </div>
  </div>`;

  // Collapse controls
  h += `<div class="collapse-bar">
    <button class="collapse-btn" onclick="toggleAllSteps(true)">[ Min All ]</button>
    <button class="collapse-btn" onclick="toggleAllSteps(false)">[ Open All ]</button>
  </div>`;

  // Time decomposition stats
  h += `<div class="stats-row">
    <div class="stat"><div class="stat-l">LLM TIME</div><div class="stat-v">${F.dur(ra.totalLlm)}</div><div class="stat-s">${F.pct(ra.totalLlm,ra.totalDur)} of total</div></div>
    <div class="stat"><div class="stat-l">TOOL TIME</div><div class="stat-v">${F.durX(ra.totalTool)}</div><div class="stat-s">${F.pct(ra.totalTool,ra.totalDur)} of total</div></div>
    <div class="stat"><div class="stat-l">PRE-LLM</div><div class="stat-v">${F.dur(ra.totalPre)}</div><div class="stat-s">${F.pct(ra.totalPre,ra.totalDur)} setup within steps</div></div>
    <div class="stat"><div class="stat-l">POST-LLM</div><div class="stat-v">${F.dur(ra.totalPost)}</div><div class="stat-s">${F.pct(ra.totalPost,ra.totalDur)} parse/dispatch</div></div>
    <div class="stat"><div class="stat-l">INIT</div><div class="stat-v">${F.dur(ra.initTime)}</div><div class="stat-s">${F.pct(ra.initTime,ra.totalDur)} before step 1</div></div>
    <div class="stat"><div class="stat-l">END</div><div class="stat-v">${F.dur(ra.endTime)}</div><div class="stat-s">${F.pct(ra.endTime,ra.totalDur)} finalize/bg learn</div></div>
    <div class="stat"><div class="stat-l">INTER-STEP</div><div class="stat-v">${F.dur(ra.interStep)}</div><div class="stat-s">${F.pct(ra.interStep,ra.totalDur)} gaps</div></div>
    <div class="stat"><div class="stat-l">AVG TTFT</div><div class="stat-v">${F.dur(ra.avgTtft)}</div><div class="stat-s">${ra.ttftCount}/${ra.ttftTotal} steps measured</div></div>
  </div>`;

  // Files modified
  if (run.files_modified && Object.keys(run.files_modified).length) {
    h += `<div class="sec-label">Files Modified</div>`;
    for (const [path,info] of Object.entries(run.files_modified)) {
      const d = info.net_lines_delta||0;
      h += `<div style="font-size:11px;padding:2px 0"><span class="file-path">${path}</span> <span class="file-d${d<0?' neg':''}">${d>=0?'+':''}${d} lines</span> <span style="color:var(--fg2)">${(info.actions||[]).join(', ')}</span></div>`;
    }
    h += '<div style="margin-bottom:12px"></div>';
  }

  // Step timeline
  h += `<div class="sec-label">Step Timeline</div>`;
  h += `<div class="legend">
    <span class="legend-i"><span class="legend-s" style="background:var(--warning);opacity:.6"></span> pre-LLM</span>
    <span class="legend-i"><span class="legend-s" style="background:var(--primary)"></span> LLM</span>
    <span class="legend-i"><span class="legend-s" style="background:var(--success)"></span> tool</span>
    <span class="legend-i"><span class="legend-s" style="background:var(--error);opacity:.4"></span> post-LLM</span>
  </div>`;
  h += '<div class="step-tl">';

  ra.analyzed.forEach(({step, a}) => {
    const cls = a.hasError ? ' err' : (a.pre > 2 || a.post > 2 ? ' warn' : '');
    const maxT = a.total || 1;
    const prePct = (a.pre/maxT*100).toFixed(1);
    const llmPct = (a.llm/maxT*100).toFixed(1);
    const toolPct = (a.tool/maxT*100).toFixed(1);
    const postPct = (a.post/maxT*100).toFixed(1);
    const preOff = 0;
    const llmOff = parseFloat(prePct);
    const toolOff = llmOff + parseFloat(llmPct);
    const postOff = toolOff + parseFloat(toolPct);

    h += `<div class="step${cls}" id="step-${step.step_id}">
      <div class="step-hdr" onclick="toggleStep('step-${step.step_id}')"><div class="step-h"><span class="step-id"><span class="chevron">▼</span>step ${step.step_id}</span><span class="step-dur">${F.dur(a.total)}</span></div></div>`;

    h += '<div class="step-body">';

    // Timing bar
    h += `<div class="t-bar-row">
      <span class="t-bar-label">timing</span>
      <div class="t-bar-track">
        ${a.pre>0?`<div class="t-bar-seg pre" style="left:${preOff}%;width:${prePct}%" title="pre-LLM ${F.dur(a.pre)}"></div>`:''}
        <div class="t-bar-seg llm" style="left:${llmOff}%;width:${Math.max(parseFloat(llmPct),0.5)}%" title="LLM ${F.dur(a.llm)}"></div>
        ${a.tool>0?`<div class="t-bar-seg tool" style="left:${toolOff}%;width:${toolPct}%" title="tools ${F.dur(a.tool)}"></div>`:''}
        ${a.post>0.01?`<div class="t-bar-seg post" style="left:${postOff}%;width:${Math.max(parseFloat(postPct),0.5)}%" title="post-LLM ${F.dur(a.post)}"></div>`:''}
      </div>
      <span class="t-bar-val"></span>
    </div>`;

    // Decomposition numbers
    h += `<div style="display:flex;flex-wrap:wrap;gap:12px;font-size:10px;color:var(--fg2);margin:4px 0">`;
    if (a.pre > 0.01) h += `<span style="color:var(--warning)">pre: ${F.dur(a.pre)}</span>`;
    h += `<span style="color:var(--primary)">llm: ${F.dur(a.llm)}</span>`;
    if (a.tool > 0) h += `<span style="color:var(--success)">tool: ${F.dur(a.tool)}</span>`;
    if (a.post > 0.01) h += `<span style="color:var(--error);opacity:.7">post: ${F.dur(a.post)}</span>`;
    h += `</div>`;

    // LLM details — clickable to expand input/output
    if (step.llm) {
      const l = step.llm;
      const llmId = 'llm-d-'+step.step_id;
      const ttftNote = a.ttftMissing ? ' <span style="color:var(--warning)">▲ not captured</span>' : '';
      h += `<div class="step-sec"><div class="step-sec-l">LLM</div>
        <div class="llm-toggle" onclick="event.stopPropagation();toggleDetail('${llmId}')" title="click to show input/output">
        <div style="display:flex;flex-wrap:wrap;gap:12px;font-size:11px;color:var(--fg1)">
          <span class="kv"><span class="k">model </span><span class="v">${l.model||'?'}</span></span>
          <span class="kv"><span class="k">ttft </span><span class="v" style="color:var(--primary)">${a.ttft>0?F.dur(a.ttft):'—'}</span>${ttftNote}</span>
          <span class="kv"><span class="k">duration </span><span class="v">${F.dur(l.duration_s)}</span></span>
          <span class="kv"><span class="k">in </span><span class="v">${F.tok(l.tokens_in)}</span></span>
          <span class="kv"><span class="k">out </span><span class="v">${F.tok(l.tokens_out)}</span></span>
          <span class="kv"><span class="k">tok/s </span><span class="v">${a.tokSec?a.tokSec.toFixed(1):'—'}</span></span>
        </div></div>
        <div class="llm-detail" id="${llmId}">
          <div class="llm-detail-tab-bar">
            <button class="llm-detail-tab active" onclick="event.stopPropagation();switchLlmTab('${llmId}','input')">Input</button>
            <button class="llm-detail-tab" onclick="event.stopPropagation();switchLlmTab('${llmId}','output')">Output</button>
          </div>
          <div class="llm-detail-content" id="${llmId}-input">${renderLlmMessages(l.input_messages||l.messages||l.input)}</div>
          <div class="llm-detail-content" id="${llmId}-output" style="display:none">${renderLlmOutput(l.output||l.response||l.output_text)}</div>
        </div></div>`;
    }

    // Tool calls — each row expandable for args/result
    if (step.tool_calls?.length) {
      h += `<div class="step-sec"><div class="step-sec-l">Tools</div>`;
      step.tool_calls.forEach((tc,ti) => {
        const isErr = tc.status==='error';
        const tcId = 'tc-'+step.step_id+'-'+ti;
        h += `<div class="tool-row-wrap">`;
        h += `<div class="tool-row${isErr?' terr':''}" onclick="toggleDetail('${tcId}')" title="click to expand"><span>${isErr?'✕ ':'▸ '}${tc.name}</span><span>${F.dur(tc.duration_s)}</span><span>${tc.status}</span><span style="color:var(--fg2)">${esc((tc.result_summary||tc.error||tc.args_summary||'').substring(0,80))}</span></div>`;
        h += `<div class="tool-detail" id="${tcId}">`;
        if (tc.args || tc.args_summary || tc.input) {
          h += `<div class="tool-detail-l">Input / Args</div><pre>${esc(fmtJson(tc.args||tc.input||tc.args_summary))}</pre>`;
        }
        if (tc.result || tc.result_summary || tc.output || tc.error) {
          h += `<div class="tool-detail-l">${isErr?'Error':'Result'}</div><pre>${esc(fmtJson(tc.result||tc.output||tc.result_summary||tc.error))}</pre>`;
        }
        h += `</div></div>`;
        // Linked sub-agent run — nested, collapsed, under its spawn tool-call
        const subEntry = matchSub(tc.name);
        if (subEntry) h += renderSubAgentRun(subEntry, 1);
      });
      h += '</div>';
    } else if (a.pre > 1) {
      h += `<div class="step-sec"><div class="step-sec-l">Tools</div>
        <div style="font-size:11px;color:var(--warning)">▲ 0 tool calls recorded, but ${F.dur(a.pre)} spent before LLM call. Likely untracked tool execution or init.</div></div>`;
    }

    // VFS
    if (step.vfs_deltas?.length) {
      h += `<div class="step-sec"><div class="step-sec-l">VFS</div>`;
      step.vfs_deltas.forEach(vd => {
        const linesAdded = vd.lines_added != null
          ? vd.lines_added
          : (vd.after_content != null ? vd.after_content.split('\n').length : 0);
        const linesRemoved = vd.lines_removed != null
          ? vd.lines_removed
          : (vd.before_content != null ? vd.before_content.split('\n').length : 0);
        h += `<div class="vfs-row"><span class="file-path">${vd.path}</span><span>${vd.action}</span><span class="file-d">+${linesAdded}</span><span class="file-d neg">-${linesRemoved}</span></div>`;
      });
      h += '</div>';
    }

    // Compression
    if (step.compression) {
      h += `<div class="step-sec"><div class="step-sec-l">Compression</div>
        <div style="font-size:11px;color:var(--fg1)">kept: ${step.compression.kept||0} │ summarized: ${step.compression.summarized||0} │ dropped: ${step.compression.dropped||0}</div></div>`;
    }

    h += '</div>'; // step-body
    h += '</div>'; // step
  });

  h += '</div>'; // step-tl
  c.innerHTML = h;
}

function backToDash() {
  document.querySelectorAll('.tab').forEach(t=>{t.classList.toggle('active',t.dataset.view==='dashboard');t.textContent=t.dataset.view==='dashboard'?'> dashboard':t.dataset.view});
  document.querySelectorAll('.view').forEach(v=>v.classList.toggle('active',v.id==='dashboard'));
  document.getElementById('nav-crumb').textContent='dashboard';
}

// ══════════════════════════════════════════════════════════════════
// VIEW 3: TIME ANALYSIS
// ══════════════════════════════════════════════════════════════════
function buildTiming() {
  const root = document.getElementById('timing');
  if (!runs.length) { root.innerHTML='<div style="color:var(--fg1);font-size:11px">No completed runs.</div>'; return; }

  let h = '';
  // Aggregate across all runs
  let aLlm=0,aTool=0,aPre=0,aPost=0,aInit=0,aEnd=0,aInter=0,aTotal=0;
  const runAnalyses = runs.map(r => {
    const ra = analyzeRun(r);
    aLlm+=ra.totalLlm; aTool+=ra.totalTool; aPre+=ra.totalPre; aPost+=ra.totalPost;
    aInit+=ra.initTime; aEnd+=ra.endTime; aInter+=ra.interStep; aTotal+=ra.totalDur;
    return { run: r, ra };
  });

  // Section 1: Full decomposition
  h += '<div class="time-sec"><div class="time-sec-t">Full Time Decomposition</div>';
  h += `<div class="legend">
    <span class="legend-i"><span class="legend-s" style="background:var(--primary)"></span> LLM</span>
    <span class="legend-i"><span class="legend-s" style="background:var(--success)"></span> Tool exec</span>
    <span class="legend-i"><span class="legend-s" style="background:var(--warning)"></span> Pre-LLM (setup within step)</span>
    <span class="legend-i"><span class="legend-s" style="background:var(--error);opacity:.6"></span> Post-LLM (parse/dispatch)</span>
    <span class="legend-i"><span class="legend-s" style="background:var(--info)"></span> Init (before step 1)</span>
    <span class="legend-i"><span class="legend-s" style="background:var(--fg2)"></span> End (finalize/bg learn)</span>
    <span class="legend-i"><span class="legend-s" style="background:rgba(255,255,255,0.1)"></span> Inter-step gaps</span>
  </div>`;
  const maxT = aTotal||1;
  const categories = [
    {l:'LLM',v:aLlm,c:'llm'},{l:'Tool exec',v:aTool,c:'tool'},
    {l:'Pre-LLM',v:aPre,c:'pre'},{l:'Post-LLM',v:aPost,c:'post'},
    {l:'Init',v:aInit,c:'llm',extra:'opacity:0.5'},{l:'End',v:aEnd,c:'other'},
    {l:'Inter-step',v:aInter,c:'other',extra:'opacity:0.4'},
  ];
  h += '<div class="bar-chart">';
  categories.forEach(b=>{
    const pct=(b.v/maxT*100).toFixed(1);
    h+=`<div class="bar-r"><span class="bar-l">${b.l}</span><div class="bar-t"><div class="bar-f ${b.c}" style="width:${pct}%"></div></div><span class="bar-v">${F.dur(b.v)} (${pct}%)</span></div>`;
  });
  h += '</div></div>';

  // Section 2: Per-run table
  h += '<div class="time-sec"><div class="time-sec-t">Per-Run Breakdown</div>';
  h += `<table class="tbl"><tr><th>RUN</th><th class="n">TOTAL</th><th class="n">LLM</th><th class="n">PRE-LLM</th><th class="n">POST-LLM</th><th class="n">TOOLS</th><th class="n">INIT</th><th class="n">END</th><th class="n">AVG TTFT</th><th>STATUS</th></tr>`;
  runAnalyses.forEach(({run:r, ra}, idx)=>{
    let tokSecs=[];
    ra.analyzed.forEach(({a})=>{if(a.tokSec>0)tokSecs.push(a.tokSec)});
    const avgTS=tokSecs.length?(tokSecs.reduce((a,b)=>a+b,0)/tokSecs.length).toFixed(1):'—';
    const st=r.success?'<span style="color:var(--success)">●</span>':'<span style="color:var(--error)">✕</span>';
    h+=`<tr style="cursor:pointer" onclick="showWaterfall(${idx})">
      <td style="color:var(--primary)">${r.run_id}</td><td class="n">${F.dur(ra.totalDur)}</td>
      <td class="n">${F.dur(ra.totalLlm)}</td><td class="n">${F.dur(ra.totalPre)}</td>
      <td class="n">${F.dur(ra.totalPost)}</td><td class="n">${F.durX(ra.totalTool)}</td>
      <td class="n">${F.dur(ra.initTime)}</td><td class="n">${F.dur(ra.endTime)}</td>
      <td class="n">${F.dur(ra.avgTtft)}</td><td>${st}</td></tr>`;
  });
  h += '</table></div>';

  // Waterfall container (rendered dynamically on row click)
  h += '<div id="wf-container"></div>';

  // Section 4: Speed calculator
  h += `<div class="time-sec"><div class="time-sec-t">Speed Calculator</div>
    <div style="font-size:11px;color:var(--fg1);margin-bottom:12px">What if the model or tools were faster? Simulates across ${runs.length} run(s).</div>
    <div class="calc-grid">
      <div class="calc-card">
        <div class="calc-card-t">LLM Speed</div>
        <div class="slider-r"><label>multiplier</label><input type="range" id="sl-llm" min="1" max="10" step="0.5" value="1"><span class="val" id="sv-llm">1.0x</span></div>
        <div class="calc-res" id="cr-llm"></div>
      </div>
      <div class="calc-card">
        <div class="calc-card-t">Pre-LLM (init/processing)</div>
        <div class="slider-r"><label>multiplier</label><input type="range" id="sl-pre" min="1" max="10" step="0.5" value="1"><span class="val" id="sv-pre">1.0x</span></div>
        <div class="calc-res" id="cr-pre"></div>
      </div>
    </div>
    <div class="calc-card" style="margin-top:8px">
      <div class="calc-card-t">Combined Impact</div>
      <div class="calc-res" id="cr-comb"></div>
    </div>
  </div>`;

  root.innerHTML = h;

  // Store for click handler
  window._runAnalyses = runAnalyses;

  // Show latest run waterfall by default
  if (runAnalyses.length) showWaterfall(runAnalyses.length - 1);

  // Calculator logic
  const sLlm=document.getElementById('sl-llm'), sPre=document.getElementById('sl-pre');
  function updCalc(){
    const lx=parseFloat(sLlm.value), px=parseFloat(sPre.value);
    document.getElementById('sv-llm').textContent=lx.toFixed(1)+'x';
    document.getElementById('sv-pre').textContent=px.toFixed(1)+'x';
    const nLlm=aLlm/lx, nPre=aPre/px;
    const nTotal=nLlm+aTool+nPre+aPost+aInit+aEnd+aInter;
    const saved=aTotal-nTotal;
    document.getElementById('cr-llm').innerHTML=`LLM: ${F.dur(aLlm)} → <span class="hi">${F.dur(nLlm)}</span> (−${F.dur(aLlm-nLlm)})`;
    document.getElementById('cr-pre').innerHTML=`Pre-LLM: ${F.dur(aPre)} → <span class="hi">${F.dur(nPre)}</span> (−${F.dur(aPre-nPre)})`;
    document.getElementById('cr-comb').innerHTML=`Total: ${F.dur(aTotal)} → <span class="hi">${F.dur(nTotal)}</span> │ saved <span class="hi">${F.dur(saved)}</span> (${F.pct(saved,aTotal)})`;
  }
  sLlm.addEventListener('input',updCalc);
  sPre.addEventListener('input',updCalc);
  updCalc();

  // Section 5: Top slowest tools
  let allToolCalls=[];
  runs.forEach(r=>{(r.steps||[]).forEach(s=>{(s.tool_calls||[]).forEach(tc=>{allToolCalls.push({run:r.run_id,step:s.step_id,...tc})})})});
  if (allToolCalls.length) {
    allToolCalls.sort((a,b)=>(b.duration_s||0)-(a.duration_s||0));
    const top = allToolCalls.slice(0,10);
    let th = '<div class="time-sec"><div class="time-sec-t">Top Slowest Tool Calls</div>';
    th += '<table class="tbl"><tr><th>TOOL</th><th class="n">DURATION</th><th>RUN</th><th class="n">STEP</th><th>STATUS</th></tr>';
    top.forEach(t=>{
      const st=t.status==='error'?'<span style="color:var(--error)">✕</span>':'<span style="color:var(--success)">●</span>';
      th+=`<tr><td style="color:var(--primary)">${t.name}</td><td class="n">${F.dur(t.duration_s)}</td><td>${t.run}</td><td class="n">${t.step}</td><td>${st}</td></tr>`;
    });
    th += '</table></div>';
    root.insertAdjacentHTML('beforeend', th);
  }
}

// ── showWaterfall: renders waterfall + step table for a selected run ──
function showWaterfall(idx) {
  const {run: lr, ra} = window._runAnalyses[idx];
  const container = document.getElementById('wf-container');
  const steps = lr.steps || [];
  if (!steps.length) { container.innerHTML = ''; return; }

  const runStart = steps[0]?.t_start || 0;
  const runEnd = steps[steps.length-1]?.t_end || runStart + 1;
  const span = runEnd - runStart || 1;

  // Highlight selected row in table
  document.querySelectorAll('#timing .tbl tr').forEach((tr,i) => {
    if (i === 0) return; // header
    tr.style.background = (i-1 === idx) ? 'var(--sel)' : '';
  });

  let h = `<div class="time-sec"><div class="time-sec-t">Step Waterfall — ${lr.run_id}</div>`;
  h += `<div class="legend">
    <span class="legend-i"><span class="legend-s" style="background:var(--warning);opacity:.5"></span> pre-LLM</span>
    <span class="legend-i"><span class="legend-s" style="background:var(--primary)"></span> LLM</span>
    <span class="legend-i"><span class="legend-s" style="background:var(--success)"></span> tool</span>
    <span class="legend-i"><span class="legend-s" style="background:var(--error);opacity:.35"></span> post-LLM</span>
  </div>`;

  ra.analyzed.forEach(({step: s, a}) => {
    h += `<div class="wf-row"><span class="wf-label">step ${s.step_id}</span><div class="wf-track">`;
    const sOff = s.t_start - runStart;
    if (a.pre > 0.01 && s.llm?.t_start) {
      const x = (sOff/span*100).toFixed(2);
      const w = (a.pre/span*100).toFixed(2);
      h += `<div class="wf-seg pre" style="left:${x}%;width:${Math.max(parseFloat(w),0.3)}%" title="pre-LLM ${F.dur(a.pre)}"></div>`;
    }
    if (s.llm?.t_start) {
      const x = ((s.llm.t_start-runStart)/span*100).toFixed(2);
      const w = (a.llm/span*100).toFixed(2);
      h += `<div class="wf-seg llm" style="left:${x}%;width:${Math.max(parseFloat(w),0.3)}%" title="LLM ${F.dur(a.llm)}"></div>`;
    }
    (s.tool_calls||[]).forEach(tc => {
      if (tc.t_start) {
        const x = ((tc.t_start-runStart)/span*100).toFixed(2);
        const w = (tc.duration_s/span*100).toFixed(2);
        h += `<div class="wf-seg tool" style="left:${x}%;width:${Math.max(parseFloat(w),0.3)}%" title="${tc.name} ${F.dur(tc.duration_s)}"></div>`;
      }
    });
    if (a.post > 0.01) {
      const postStart = (s.llm?.t_end || s.t_start) + a.tool;
      const x = ((postStart-runStart)/span*100).toFixed(2);
      const w = (a.post/span*100).toFixed(2);
      h += `<div class="wf-seg post" style="left:${x}%;width:${Math.max(parseFloat(w),0.3)}%" title="post-LLM ${F.dur(a.post)}"></div>`;
    }
    h += '</div></div>';
  });
  h += '</div>';

  // Step detail table
  h += '<div class="time-sec"><div class="time-sec-t">Step Detail — ' + lr.run_id + '</div>';
  h += `<table class="tbl"><tr><th>STEP</th><th class="n">TOTAL</th><th class="n">PRE</th><th class="n">LLM</th><th class="n">TOOL</th><th class="n">POST</th><th class="n">TTFT</th><th class="n">TOK/S</th><th class="n">IN</th><th class="n">OUT</th></tr>`;
  ra.analyzed.forEach(({step: s, a}) => {
    const preWarn = a.pre > 2 ? ' style="color:var(--warning)"' : '';
    const postWarn = a.post > 2 ? ' style="color:var(--error)"' : '';
    const ttftNote = a.ttftMissing ? ' ▲' : '';
    h += `<tr><td style="color:var(--primary)">step ${s.step_id}</td>
      <td class="n">${F.dur(a.total)}</td>
      <td class="n"${preWarn}>${F.dur(a.pre)}</td>
      <td class="n">${F.dur(a.llm)}</td>
      <td class="n">${F.durX(a.tool)}</td>
      <td class="n"${postWarn}>${F.dur(a.post)}</td>
      <td class="n">${a.ttft>0?F.dur(a.ttft):'—'}${ttftNote}</td>
      <td class="n">${a.tokSec?a.tokSec.toFixed(1):'—'}</td>
      <td class="n">${F.tok(a.tokIn)}</td>
      <td class="n">${F.tok(a.tokOut)}</td></tr>`;
  });
  h += '</table></div>';

  container.innerHTML = h;
  container.scrollIntoView({behavior: 'smooth', block: 'nearest'});
}

// ── Collapse/expand helpers ──
function esc(s) {
  if (s == null) return '';
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
function fmtJson(v) {
  if (v == null) return '';
  if (typeof v === 'string') {
    try { return JSON.stringify(JSON.parse(v), null, 2); } catch(e) { return v; }
  }
  try { return JSON.stringify(v, null, 2); } catch(e) { return String(v); }
}
// Decode strings that were double-serialized (literal \n \t \" \\ but no
// real newlines). Leaves genuine multi-line content untouched, so real code
// with actual newlines is never mangled.
function decodeMaybe(s) {
  if (typeof s !== 'string') return s;
  for (let i = 0; i < 3; i++) {
    const lit = (s.match(/\\n/g) || []).length;
    const real = (s.match(/\n/g) || []).length;
    if (!lit || lit <= real) break;
    s = s.replace(/\\(n|t|r|"|\\)/g, (m, c) =>
      ({n:'\n', t:'\t', r:'\r', '"':'"', '\\':'\\'}[c]));
  }
  return s;
}

function renderLlmMessages(msgs) {
  if (!msgs) return '<span style="color:var(--fg2)">no input data captured</span>';
  if (typeof msgs === 'string') return '<pre>'+esc(msgs)+'</pre>';
  if (!Array.isArray(msgs)) return '<pre>'+esc(fmtJson(msgs))+'</pre>';
  return msgs.map(m => {
    const role = m.role || 'unknown';
    let content = '';
    if (typeof m.content === 'string') content = m.content;
    else if (Array.isArray(m.content)) content = m.content.map(c => typeof c === 'string' ? c : (c.text || JSON.stringify(c))).join('\n');
    else content = fmtJson(m.content);
    return `<div class="msg role-${role}"><div class="msg-role ${role}">${esc(role)}</div><div class="msg-content">${esc(decodeMaybe(content))}</div></div>`;
  }).join('');
}
function renderLlmOutput(out) {
  if (!out) return '<span style="color:var(--fg2)">no output data captured</span>';
  if (typeof out === 'string') return '<pre>'+esc(decodeMaybe(out))+'</pre>';
  return '<pre>'+esc(fmtJson(out))+'</pre>';
}
function toggleStep(id) {
  document.getElementById(id)?.classList.toggle('minimized');
}
function toggleAllSteps(minimize) {
  document.querySelectorAll('.step-tl .step').forEach(el => {
    el.classList.toggle('minimized', minimize);
  });
}
function toggleDetail(id) {
  document.getElementById(id)?.classList.toggle('open');
}
function switchLlmTab(llmId, tab) {
  const parent = document.getElementById(llmId);
  if (!parent) return;
  parent.querySelectorAll('.llm-detail-tab').forEach(t => t.classList.toggle('active', t.textContent.toLowerCase() === tab));
  const inp = document.getElementById(llmId+'-input');
  const out = document.getElementById(llmId+'-output');
  if (inp) inp.style.display = tab==='input' ? '' : 'none';
  if (out) out.style.display = tab==='output' ? '' : 'none';
}

// ── INIT ──
buildDash();
buildTiming();
</script>
</body>
</html>
"""
