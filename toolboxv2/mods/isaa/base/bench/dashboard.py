"""
Dashboard V3 — Binary Check Dashboard

Features:
- Leaderboard with scrollable dimension columns (no overflow)
- Click dimension cell → see which tasks contribute, pass/fail per check
- Compare two models side-by-side on any dimension
- Cost & performance stats panel with tooltips
- Tooltips on all columns explaining what they mean
"""

import json
from typing import List, Any
from pathlib import Path


class Dashboard:

    @staticmethod
    def generate(reports: List[Any], title: str = "Benchmark Comparison") -> str:
        data = []
        for r in reports:
            if hasattr(r, "to_dict"):
                d = r.to_dict()
            elif isinstance(r, dict):
                d = r
            else:
                continue
            data.append(d)

        if not data:
            return "<html><body>No valid reports</body></html>"

        all_dims = set()
        for d in data:
            all_dims.update(d.get("dimensions", {}).keys())
        dims_sorted = sorted(all_dims)

        data_json = json.dumps(data, default=str, indent=None)
        # Escape for safe embedding in <script> tag
        data_json_safe = data_json.replace("</script>", "<\\/script>")

        dim_headers = "".join(
            f'<th class="dim-th sortable" data-sort="{d}" title="Tag: {d} — average pass rate of all tasks with this tag">{d[:7]}</th>'
            for d in dims_sorted
        )

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet"/>
<style>
:root {{
  --bg:#08080d; --surface:#0e0e14; --border:rgba(255,255,255,0.04);
  --text:#e2e2e8; --muted:rgba(255,255,255,0.3); --accent:#00e6d2;
  --green:#10b981; --amber:#f59e0b; --red:#ef4444; --blue:#6366f1;
}}
*{{margin:0;padding:0;box-sizing:border-box;}}
body{{background:var(--bg);color:var(--text);font-family:'IBM Plex Sans',sans-serif;padding:20px;font-size:13px;}}
.wrap{{max-width:1600px;margin:0 auto;}}
h1{{font-size:20px;font-weight:300;margin-bottom:2px;}}
.sub{{font-size:9px;color:var(--muted);letter-spacing:2px;text-transform:uppercase;margin-bottom:20px;}}
.card{{background:var(--surface);border:1px solid var(--border);border-radius:6px;padding:16px;margin-bottom:14px;}}
.card-t{{font-size:9px;font-weight:600;letter-spacing:2px;text-transform:uppercase;color:var(--muted);margin-bottom:10px;}}

/* Table wrapper — horizontal scroll prevents left overflow */
.table-wrap{{overflow-x:auto;-webkit-overflow-scrolling:touch;margin:0 -4px;padding:0 4px;}}
table{{width:100%;border-collapse:collapse;min-width:600px;table-layout:auto;}}
th{{text-align:left;padding:6px 8px;border-bottom:1px solid var(--border);font-size:8px;color:var(--muted);
    letter-spacing:1.5px;text-transform:uppercase;white-space:nowrap;user-select:none;}}
th.sortable{{cursor:pointer;}}
th.sortable:hover{{color:var(--accent);}}
td{{padding:6px 8px;border-bottom:1px solid rgba(255,255,255,0.015);font-size:11px;white-space:nowrap;}}
tr{{transition:background .1s;}}
tr:hover{{background:rgba(0,230,210,0.015);}}
tr.sel{{background:rgba(0,230,210,0.03);}}
.rk{{font-weight:600;width:24px;color:var(--muted);text-align:center;}}
.rk.g{{color:#fbbf24;}}.rk.s{{color:#94a3b8;}}.rk.b{{color:#d97706;}}
.mn{{font-family:'IBM Plex Mono',monospace;font-weight:500;max-width:180px;overflow:hidden;text-overflow:ellipsis;}}
.sc{{font-family:'IBM Plex Mono',monospace;font-weight:600;}}
.hi{{color:var(--green);}}.md{{color:var(--amber);}}.lo{{color:var(--red);}}
.bar{{width:60px;height:5px;background:rgba(255,255,255,0.04);border-radius:3px;display:inline-block;vertical-align:middle;margin-left:6px;}}
.bar>span{{display:block;height:100%;border-radius:3px;}}
.mono{{font-family:'IBM Plex Mono',monospace;font-size:10px;}}
.dim-cell{{cursor:pointer;position:relative;}}
.dim-cell:hover{{text-decoration:underline;text-underline-offset:2px;}}

/* Controls */
.ctrl{{display:flex;gap:10px;margin-bottom:14px;align-items:center;flex-wrap:wrap;}}
.ctrl input,.ctrl select{{background:rgba(0,0,0,.3);border:1px solid var(--border);border-radius:4px;
  padding:5px 8px;color:var(--text);font-size:10px;font-family:'IBM Plex Mono',monospace;}}
.ctrl input:focus{{border-color:rgba(0,230,210,.3);outline:none;}}
.ctrl label{{font-size:9px;color:var(--muted);letter-spacing:1px;text-transform:uppercase;}}

/* Stats grid */
.stats-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px;}}
.stat{{background:var(--bg);border:1px solid var(--border);border-radius:4px;padding:10px 12px;}}
.stat-label{{font-size:8px;color:var(--muted);letter-spacing:1.5px;text-transform:uppercase;margin-bottom:4px;}}
.stat-val{{font-family:'IBM Plex Mono',monospace;font-size:16px;font-weight:600;}}

/* Panels */
.panel{{display:none;}}.panel.active{{display:block;}}
.panel-header{{display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;
  padding-bottom:10px;border-bottom:1px solid var(--border);}}
.panel-title{{font-size:14px;font-weight:500;color:var(--accent);}}
.close-btn{{cursor:pointer;color:var(--muted);font-size:16px;padding:2px 6px;border-radius:3px;}}
.close-btn:hover{{color:var(--text);background:rgba(255,255,255,0.03);}}

/* Dimension drill-down */
.dim-task{{display:grid;grid-template-columns:20px 1fr 60px;gap:6px;align-items:center;
  padding:5px 0;border-bottom:1px solid rgba(255,255,255,0.02);font-size:11px;}}
.dim-task:last-child{{border-bottom:none;}}

/* Checks */
.chk{{display:grid;grid-template-columns:16px 120px 1fr;gap:6px;align-items:center;
  padding:3px 0;font-size:10px;}}
.chk-i{{font-size:12px;}}.chk-n{{font-family:'IBM Plex Mono',monospace;color:var(--accent);}}
.chk-d{{color:var(--muted);font-family:'IBM Plex Mono',monospace;font-size:9px;overflow:hidden;text-overflow:ellipsis;}}

/* Agent trace */
.trace{{margin-top:8px;border:1px solid rgba(99,102,241,0.12);border-radius:4px;overflow:hidden;}}
.trace-title{{font-size:8px;color:var(--blue);letter-spacing:1.5px;text-transform:uppercase;
  padding:6px 10px;background:rgba(99,102,241,0.04);cursor:pointer;user-select:none;display:flex;justify-content:space-between;}}
.trace-body{{display:none;padding:6px 0;}}.trace.open .trace-body{{display:block;}}
.trace-msg{{padding:4px 10px;font-size:9px;font-family:'IBM Plex Mono',monospace;border-bottom:1px solid rgba(255,255,255,0.015);}}
.trace-msg:last-child{{border-bottom:none;}}
.trace-role{{display:inline-block;width:70px;font-weight:600;text-transform:uppercase;letter-spacing:0.5px;}}
.trace-role.user{{color:var(--accent);}}.trace-role.assistant{{color:var(--green);}}
.trace-role.tool{{color:var(--amber);}}.trace-role.system{{color:var(--muted);}}
.trace-content{{color:var(--muted);white-space:pre-wrap;max-height:100px;overflow-y:auto;display:inline;}}
.tool-badge{{display:inline-block;padding:1px 5px;border-radius:2px;font-size:8px;
  background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.15);color:var(--amber);margin-right:4px;}}

/* Compare */
.cmp-grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px;}}
.cmp-col{{background:var(--bg);border:1px solid var(--border);border-radius:4px;padding:12px;}}
.cmp-model{{font-family:'IBM Plex Mono',monospace;font-size:12px;font-weight:500;color:var(--accent);margin-bottom:8px;}}
.cmp-task{{padding:4px 0;border-bottom:1px solid rgba(255,255,255,0.02);font-size:10px;display:flex;justify-content:space-between;}}

/* Tooltip */
[title]{{position:relative;}}

/* Response */
.resp{{background:rgba(0,0,0,.3);padding:8px 10px;border-radius:4px;font-family:'IBM Plex Mono',monospace;
  font-size:9px;color:var(--muted);max-height:120px;overflow-y:auto;white-space:pre-wrap;margin-top:6px;}}
.lbl{{font-size:8px;color:var(--muted);letter-spacing:1px;text-transform:uppercase;margin-top:8px;margin-bottom:3px;}}

/* Task card (in detail panel) */
.tc{{background:var(--bg);border:1px solid var(--border);border-radius:4px;margin-bottom:6px;overflow:hidden;}}
.tc-h{{display:flex;align-items:center;gap:8px;padding:8px 12px;cursor:pointer;user-select:none;}}
.tc-h:hover{{background:rgba(255,255,255,.01);}}
.tc-b{{display:none;padding:10px 12px;border-top:1px solid var(--border);}}
.tc.open .tc-b{{display:block;}}
.tag{{font-size:8px;padding:1px 5px;border-radius:2px;background:rgba(0,230,210,.05);
  border:1px solid rgba(0,230,210,.12);color:var(--accent);margin-left:auto;}}
</style>
</head>
<body>
<div class="wrap">
<div class="sub">bench framework</div>
<h1>{title}</h1>
<p style="font-size:10px;color:var(--muted);margin-bottom:16px;" id="summary"></p>

<div class="ctrl">
  <input type="text" id="q" placeholder="Filter models..." oninput="render()"/>
  <label>Compare:</label>
  <select id="cmpA" onchange="renderCompare()"><option value="">Model A</option></select>
  <span style="color:var(--muted);font-size:10px;">vs</span>
  <select id="cmpB" onchange="renderCompare()"><option value="">Model B</option></select>
</div>

<!-- Stats -->
<div class="card" id="statsCard">
  <div class="card-t">Performance Overview</div>
  <div class="stats-grid" id="statsGrid"></div>
</div>

<!-- Leaderboard -->
<div class="card">
  <div class="card-t">Leaderboard <span style="font-weight:400;letter-spacing:0;text-transform:none;font-size:9px;color:var(--muted);">— click dimension scores to drill down</span></div>
  <div class="table-wrap">
  <table>
    <thead><tr>
      <th title="Rank by total score">#</th>
      <th class="sortable" data-sort="model" title="Model identifier">Model</th>
      <th class="sortable" data-sort="total" title="Average pass rate across all tasks (0-100%)">Score</th>
      {dim_headers}
      <th class="sortable" data-sort="cost" title="Total API cost in USD">Cost</th>
      <th class="sortable" data-sort="time" title="Total wall-clock time in seconds">Time</th>
      <th class="sortable" data-sort="tokens" title="Total tokens (in + out)">Tokens</th>
      <th class="sortable" data-sort="probes" title="Number of tasks executed">Tasks</th>
    </tr></thead>
    <tbody id="tb"></tbody>
  </table>
  </div>
</div>

<!-- Dimension drill-down panel -->
<div class="card panel" id="dimPanel">
  <div class="panel-header">
    <span class="panel-title" id="dimTitle"></span>
    <span class="close-btn" onclick="closeDim()">✕</span>
  </div>
  <div id="dimContent"></div>
</div>

<!-- Model detail panel -->
<div class="card panel" id="detailPanel">
  <div class="panel-header">
    <span class="panel-title" id="detailTitle"></span>
    <span class="close-btn" onclick="closeDetail()">✕</span>
  </div>
  <div class="ctrl" style="margin-bottom:10px;">
    <input type="text" id="tq" placeholder="Filter tasks..." oninput="renderTasks()"/>
    <select id="tf" onchange="renderTasks()">
      <option value="all">All</option>
      <option value="pass">✓ Passed</option>
      <option value="fail">✗ Failed</option>
    </select>
  </div>
  <div id="detailTasks"></div>
</div>

<!-- Compare panel -->
<div class="card panel" id="cmpPanel">
  <div class="panel-header">
    <span class="panel-title">Model Comparison</span>
    <span class="close-btn" onclick="closeCmp()">✕</span>
  </div>
  <div id="cmpContent"></div>
</div>

</div>
<script>
const R=__REPORT_DATA_PLACEHOLDER__;
const dims=__DIMS_DATA_PLACEHOLDER__;
let sortCol='total',sortDir='desc',selModel=null,selDim=null;

function sc(v){{return v>=75?'hi':v>=50?'md':'lo';}}
function fmt$(v){{return v>0?'$'+v.toFixed(4):'—';}}
function fmtT(v){{return v>0?v.toFixed(1)+'s':'—';}}
function fmtN(v){{return v>0?v.toLocaleString():'—';}}
function esc(s){{
  let v = typeof s === 'string' ? s : JSON.stringify(s || '');
  return v.replace(/</g,'&lt;').replace(/>/g,'&gt;');
}}
function getM(name){{return R.find(x=>x.model===name);}}

// Sort
document.querySelectorAll('th.sortable').forEach(th=>{{
  th.onclick=()=>{{
    const c=th.dataset.sort;
    if(sortCol===c)sortDir=sortDir==='desc'?'asc':'desc';
    else{{sortCol=c;sortDir='desc';}}
    render();
  }};
}});

function getData(){{
  let d=[...R];
  const q=document.getElementById('q').value.toLowerCase();
  if(q)d=d.filter(x=>x.model.toLowerCase().includes(q));
  d.sort((a,b)=>{{
    let av,bv;
    if(sortCol==='total'){{av=a.total;bv=b.total;}}
    else if(sortCol==='model'){{return sortDir==='asc'?a.model.localeCompare(b.model):b.model.localeCompare(a.model);}}
    else if(sortCol==='cost'){{av=(a.cost||{{}}).total_cost||0;bv=(b.cost||{{}}).total_cost||0;}}
    else if(sortCol==='time'){{av=(a.cost||{{}}).total_time_s||0;bv=(b.cost||{{}}).total_time_s||0;}}
    else if(sortCol==='tokens'){{av=(a.cost||{{}}).total_tokens||0;bv=(b.cost||{{}}).total_tokens||0;}}
    else if(sortCol==='probes'){{av=a.probes||0;bv=b.probes||0;}}
    else{{av=(a.dimensions||{{}})[sortCol]||0;bv=(b.dimensions||{{}})[sortCol]||0;}}
    return sortDir==='desc'?bv-av:av-bv;
  }});
  return d;
}}

// Stats
function renderStats(){{
  const d=R;
  if(!d.length)return;
  const best=d.reduce((a,b)=>a.total>b.total?a:b);
  const worst=d.reduce((a,b)=>a.total<b.total?a:b);
  const avgScore=d.reduce((s,m)=>s+m.total,0)/d.length;
  const totalCost=d.reduce((s,m)=>s+(m.cost||{{}}).total_cost||0,0);
  const totalTime=d.reduce((s,m)=>s+(m.cost||{{}}).total_time_s||0,0);
  const totalTokens=d.reduce((s,m)=>s+(m.cost||{{}}).total_tokens||0,0);
  const avgTime=totalTime/d.length;
  const avgCost=totalCost/d.length;

  document.getElementById('statsGrid').innerHTML=`
    <div class="stat" title="Highest scoring model"><div class="stat-label">Best Model</div><div class="stat-val hi">${{best.model}}</div><div class="mono">${{best.total.toFixed(1)}}%</div></div>
    <div class="stat" title="Lowest scoring model"><div class="stat-label">Worst Model</div><div class="stat-val lo">${{worst.model}}</div><div class="mono">${{worst.total.toFixed(1)}}%</div></div>
    <div class="stat" title="Mean score across all models"><div class="stat-label">Avg Score</div><div class="stat-val ${{sc(avgScore)}}">${{avgScore.toFixed(1)}}%</div></div>
    <div class="stat" title="Sum of all API costs across all models"><div class="stat-label">Total Cost</div><div class="stat-val" style="color:var(--green)">${{fmt$(totalCost)}}</div></div>
    <div class="stat" title="Average cost per model run"><div class="stat-label">Avg Cost/Model</div><div class="stat-val" style="color:var(--green)">${{fmt$(avgCost)}}</div></div>
    <div class="stat" title="Average wall-clock time per model"><div class="stat-label">Avg Time/Model</div><div class="stat-val">${{fmtT(avgTime)}}</div></div>
    <div class="stat" title="Total tokens consumed across all runs"><div class="stat-label">Total Tokens</div><div class="stat-val mono">${{fmtN(totalTokens)}}</div></div>
    <div class="stat" title="Number of models tested"><div class="stat-label">Models</div><div class="stat-val">${{d.length}}</div></div>
  `;
}}

// Leaderboard
function render(){{
  const d=getData();
  document.getElementById('summary').textContent=`${{d.length}} models · ${{d.length>0?(d[0].probes||0):0}} tasks · binary checks only`;

  // Populate compare dropdowns
  const opts='<option value="">—</option>'+R.map(m=>`<option value="${{m.model}}">${{m.model}}</option>`).join('');
  document.getElementById('cmpA').innerHTML=opts;
  document.getElementById('cmpB').innerHTML=opts;

  const tb=document.getElementById('tb');
  tb.innerHTML=d.map((m,i)=>{{
    const rk=i+1;
    const rc=rk===1?'g':rk===2?'s':rk===3?'b':'';
    const cls=sc(m.total);
    const sel=m.model===selModel?'sel':'';
    const cost=m.cost||{{}};

    const dimCells=dims.map(dim=>{{
      const v=(m.dimensions||{{}})[dim];
      if(v===undefined)return'<td class="mono" style="color:var(--muted)">—</td>';
      return`<td class="dim-cell ${{sc(v)}}" onclick="event.stopPropagation();showDim('${{m.model}}','${{dim}}')" title="Click to see which tasks make up this score"><span class="mono">${{v.toFixed(0)}}</span></td>`;
    }}).join('');

    return`<tr class="${{sel}}" onclick="showDetail('${{m.model}}')" style="cursor:pointer" title="Click for full task breakdown">
      <td class="rk ${{rc}}">${{rk}}</td>
      <td class="mn">${{m.model}}</td>
      <td><span class="sc ${{cls}}">${{m.total.toFixed(1)}}</span><span class="bar"><span class="${{cls}}" style="width:${{m.total}}%"></span></span></td>
      ${{dimCells}}
      <td class="mono" style="color:var(--green)" title="API cost: ${{fmt$(cost.total_cost)}}">${{fmt$(cost.total_cost)}}</td>
      <td class="mono" title="Wall time: ${{fmtT(cost.total_time_s)}}">${{fmtT(cost.total_time_s)}}</td>
      <td class="mono" title="In: ${{fmtN(cost.tokens_in)}} Out: ${{fmtN(cost.tokens_out)}}">${{fmtN(cost.total_tokens)}}</td>
      <td class="mono">${{m.probes||0}}</td>
    </tr>`;
  }}).join('');
}}

// Dimension drill-down
function showDim(model,dim){{
  selDim={{model,dim}};
  const m=getM(model);if(!m)return;
  const probes=m.probe_details||m.results||[];
  // Find tasks that have this dim in their scores
  const relevant=probes.filter(p=>{{
    const scores=p.scores||{{}};
    return scores[dim]!==undefined;
  }});

  document.getElementById('dimTitle').textContent=`${{model}} → ${{dim}} (${{(m.dimensions||{{}})[dim]||0}}%)`;

  let html=`<div style="font-size:10px;color:var(--muted);margin-bottom:10px;">
    ${{relevant.length}} tasks contribute to this dimension score.
    Score = average pass rate of checks across these tasks.</div>`;

  relevant.forEach(p=>{{
    const checks=p.checks||[];
    const passed=checks.filter(c=>c.passed).length;
    const total=checks.length;
    const pct=total>0?Math.round(passed/total*100):0;
    const icon=passed===total?'✓':'✗';
    const iconC=passed===total?'var(--green)':'var(--red)';

    html+=`<div class="tc">
      <div class="tc-h" onclick="this.parentElement.classList.toggle('open')">
        <span style="color:${{iconC}};font-size:12px">${{icon}}</span>
        <span class="mono" style="font-weight:500">${{p.probe_id}}</span>
        <span class="mono ${{sc(pct)}}">${{passed}}/${{total}}</span>
        <span class="mono" style="color:var(--muted)">${{p.latency_ms||0}}ms</span>
      </div>
      <div class="tc-b">
        ${{checks.map(c=>`<div class="chk">
          <span class="chk-i" style="color:${{c.passed?'var(--green)':'var(--red)'}}">${{c.passed?'✓':'✗'}}</span>
          <span class="chk-n">${{c.name||'?'}}</span>
          <span class="chk-d" title="${{esc(c.detail)}}">${{esc(c.detail)}}</span>
        </div>`).join('')}}
        <div class="lbl">Response</div>
        <div class="resp">${{esc(p.response)}}</div>
        ${{renderTrace(p)}}
      </div>
    </div>`;
  }});

  document.getElementById('dimContent').innerHTML=html;
  document.getElementById('dimPanel').classList.add('active');
  document.getElementById('dimPanel').scrollIntoView({{behavior:'smooth',block:'start'}});
}}
function closeDim(){{document.getElementById('dimPanel').classList.remove('active');selDim=null;}}

// Full model detail
function showDetail(model){{
  selModel=model;
  document.getElementById('detailTitle').textContent=model;
  document.getElementById('detailPanel').classList.add('active');
  render();
  renderTasks();
  document.getElementById('detailPanel').scrollIntoView({{behavior:'smooth',block:'start'}});
}}
function closeDetail(){{
  selModel=null;
  document.getElementById('detailPanel').classList.remove('active');
  render();
}}

function renderTasks(){{
  const m=getM(selModel);if(!m)return;
  const probes=m.probe_details||m.results||[];
  const q=(document.getElementById('tq').value||'').toLowerCase();
  const f=document.getElementById('tf').value;
  const container=document.getElementById('detailTasks');

  let html='';let shown=0;
  probes.forEach((p,i)=>{{
    const pid=p.probe_id||`task_${{i}}`;
    const checks=p.checks||[];
    const passed=checks.filter(c=>c.passed).length;
    const total=checks.length;
    const pct=total>0?(passed/total*100):0;
    const allP=passed===total;

    if(q&&!pid.toLowerCase().includes(q))return;
    if(f==='pass'&&!allP)return;
    if(f==='fail'&&allP)return;
    shown++;

    html+=`<div class="tc">
      <div class="tc-h" onclick="this.parentElement.classList.toggle('open')">
        <span style="color:${{allP?'var(--green)':'var(--red)'}};font-size:12px">${{allP?'✓':'✗'}}</span>
        <span class="mono" style="font-weight:500">${{pid}}</span>
        <span class="mono ${{sc(pct)}}">${{passed}}/${{total}}</span>
        <span class="mono" style="color:var(--muted)">${{p.latency_ms||0}}ms · ${{p.tokens_in||0}}→${{p.tokens_out||0}}tok${{p.cost>0?' · $'+p.cost.toFixed(5):''}}</span>
      </div>
      <div class="tc-b">
        ${{checks.map(c=>`<div class="chk">
          <span class="chk-i" style="color:${{c.passed?'var(--green)':'var(--red)'}}">${{c.passed?'✓':'✗'}}</span>
          <span class="chk-n">${{c.name||'?'}}</span>
          <span class="chk-d" title="${{esc(c.detail)}}">${{esc(c.detail)}}</span>
        </div>`).join('')}}
        ${{p.prompt?`<div class="lbl">Prompt</div><div class="resp">${{esc(p.prompt)}}</div>`:''}}
        <div class="lbl">Response</div>
        <div class="resp">${{esc(p.response)||'—'}}</div>
        ${{renderTrace(p)}}
      </div>
    </div>`;
  }});

  if(!shown)html='<div style="padding:16px;color:var(--muted);text-align:center">No matching tasks</div>';
  container.innerHTML=html;
}}

// Compare two models
function renderCompare(){{
  const a=document.getElementById('cmpA').value;
  const b=document.getElementById('cmpB').value;
  const panel=document.getElementById('cmpPanel');
  const content=document.getElementById('cmpContent');

  if(!a||!b||a===b){{panel.classList.remove('active');return;}}

  const mA=getM(a),mB=getM(b);
  if(!mA||!mB)return;

  panel.classList.add('active');

  // Dimension comparison
  let dimHtml='<div style="margin-bottom:16px"><div class="card-t">Dimension Scores</div>';
  dimHtml+='<table><thead><tr><th>Dimension</th><th>'+esc(a)+'</th><th>'+esc(b)+'</th><th>Δ</th></tr></thead><tbody>';
  dims.forEach(dim=>{{
    const vA=(mA.dimensions||{{}})[dim]||0;
    const vB=(mB.dimensions||{{}})[dim]||0;
    const delta=vA-vB;
    const dC=delta>0?'var(--green)':delta<0?'var(--red)':'var(--muted)';
    const dS=delta>0?'+'+delta.toFixed(0):delta.toFixed(0);
    dimHtml+=`<tr>
      <td class="mono">${{dim}}</td>
      <td class="mono ${{sc(vA)}}">${{vA.toFixed(0)}}</td>
      <td class="mono ${{sc(vB)}}">${{vB.toFixed(0)}}</td>
      <td class="mono" style="color:${{dC}}">${{dS}}</td>
    </tr>`;
  }});
  dimHtml+='</tbody></table></div>';

  // Cost comparison
  const cA=mA.cost||{{}},cB=mB.cost||{{}};
  let costHtml='<div style="margin-bottom:16px"><div class="card-t">Cost & Performance</div>';
  costHtml+='<table><thead><tr><th>Metric</th><th>'+esc(a)+'</th><th>'+esc(b)+'</th></tr></thead><tbody>';
  [['Total Score',mA.total.toFixed(1)+'%',mB.total.toFixed(1)+'%'],
   ['Cost',fmt$(cA.total_cost),fmt$(cB.total_cost)],
   ['Time',fmtT(cA.total_time_s),fmtT(cB.total_time_s)],
   ['Tokens',fmtN(cA.total_tokens),fmtN(cB.total_tokens)],
   ['Tokens In',fmtN(cA.tokens_in),fmtN(cB.tokens_in)],
   ['Tokens Out',fmtN(cA.tokens_out),fmtN(cB.tokens_out)],
   ['Tasks',mA.probes||0,mB.probes||0],
  ].forEach(([label,va,vb])=>{{
    costHtml+=`<tr><td class="mono">${{label}}</td><td class="mono">${{va}}</td><td class="mono">${{vb}}</td></tr>`;
  }});
  costHtml+='</tbody></table></div>';

  // Per-task comparison: show where they differ
  const pA=mA.probe_details||mA.results||[];
  const pB=mB.probe_details||mB.results||[];
  const taskMap={{}};
  pA.forEach(p=>{{taskMap[p.probe_id]=taskMap[p.probe_id]||{{}};taskMap[p.probe_id].a=p;}});
  pB.forEach(p=>{{taskMap[p.probe_id]=taskMap[p.probe_id]||{{}};taskMap[p.probe_id].b=p;}});

  let diffHtml='<div><div class="card-t">Task Differences <span style="font-weight:400;text-transform:none;letter-spacing:0;font-size:9px;color:var(--muted);">— tasks where models scored differently</span></div>';
  let diffCount=0;
  Object.entries(taskMap).forEach(([tid,{{a:pa,b:pb}}])=>{{
    if(!pa||!pb)return;
    const cA=pa.checks||[],cB=pb.checks||[];
    const sA=cA.length>0?cA.filter(c=>c.passed).length/cA.length:0;
    const sB=cB.length>0?cB.filter(c=>c.passed).length/cB.length:0;
    if(Math.abs(sA-sB)<0.01)return;
    diffCount++;
    const pctA=Math.round(sA*100),pctB=Math.round(sB*100);
    diffHtml+=`<div class="tc">
      <div class="tc-h" onclick="this.parentElement.classList.toggle('open')">
        <span class="mono" style="font-weight:500">${{tid}}</span>
        <span class="mono ${{sc(pctA)}}">${{esc(a)}}: ${{pctA}}%</span>
        <span style="color:var(--muted)">vs</span>
        <span class="mono ${{sc(pctB)}}">${{esc(b)}}: ${{pctB}}%</span>
      </div>
      <div class="tc-b">
        <div class="cmp-grid">
          <div class="cmp-col">
            <div class="cmp-model">${{esc(a)}}</div>
            ${{cA.map(c=>`<div class="chk"><span class="chk-i" style="color:${{c.passed?'var(--green)':'var(--red)'}}">${{c.passed?'✓':'✗'}}</span><span class="chk-n">${{c.name}}</span><span class="chk-d">${{esc(c.detail)}}</span></div>`).join('')}}
            <div class="lbl">Response</div><div class="resp">${{esc(pa.response)}}</div>
          </div>
          <div class="cmp-col">
            <div class="cmp-model">${{esc(b)}}</div>
            ${{cB.map(c=>`<div class="chk"><span class="chk-i" style="color:${{c.passed?'var(--green)':'var(--red)'}}">${{c.passed?'✓':'✗'}}</span><span class="chk-n">${{c.name}}</span><span class="chk-d">${{esc(c.detail)}}</span></div>`).join('')}}
            <div class="lbl">Response</div><div class="resp">${{esc(pb.response)}}</div>
          </div>
        </div>
        ${{pa.prompt?`<div class="lbl">Prompt (shared)</div><div class="resp">${{esc(pa.prompt)}}</div>`:''}}
      </div>
    </div>`;
  }});
  if(!diffCount)diffHtml+='<div style="padding:12px;color:var(--muted);text-align:center">No differences found — models scored identically on all tasks</div>';
  diffHtml+='</div>';

  content.innerHTML=dimHtml+costHtml+diffHtml;
  panel.scrollIntoView({{behavior:'smooth',block:'start'}});
}}
function closeCmp(){{document.getElementById('cmpPanel').classList.remove('active');}}

function renderTrace(p){{
  const tc=p.tool_calls||[];
  const wh=p.working_history||[];
  if(!tc.length&&!wh.length)return '';

  let html=`<div class="trace" onclick="this.classList.toggle('open')">`;

  // Tool calls summary
  const toolCount=tc.length;
  const histLen=wh.length;
  html+=`<div class="trace-title"><span>Agent Trace — ${{toolCount}} tool call${{toolCount!==1?'s':''}} · ${{histLen}} messages</span><span>▸</span></div>`;
  html+='<div class="trace-body">';

  // Working history (full conversation trace)
  if(wh.length){{
    wh.forEach(msg=>{{
      const role=msg.role||'?';
      const roleClass=role==='user'?'user':role==='assistant'?'assistant':role==='tool'?'tool':'system';
      const content=esc(msg.content||'');
      html+=`<div class="trace-msg"><span class="trace-role ${{roleClass}}">${{role}}</span> <span class="trace-content">${{content}}</span></div>`;
    }});
  }}

  // Tool calls (if no working_history, show tool_calls as fallback)
  if(!wh.length&&tc.length){{
    tc.forEach(t=>{{
      const args=Object.entries(t.args||{{}}).map(([k,v])=>`${{k}}=${{esc(String(v))}}`).join(', ');
      html+=`<div class="trace-msg">
        <span class="tool-badge">${{esc(t.name)}}</span>
        <span class="trace-content" style="color:var(--muted)">${{args}}</span>
      </div>`;
      html+=`<div class="trace-msg" style="padding-left:80px">
        <span class="trace-content">→ ${{esc(t.result||'')}}</span>
      </div>`;
    }});
  }}

  html+='</div></div>';
  return html;
}}

renderStats();
render();
</script>
</body>
</html>"""
        # Replace placeholders with actual data (avoids f-string breaking on {{ }} in JSON)
        html = html.replace("__REPORT_DATA_PLACEHOLDER__", data_json_safe)
        html = html.replace("__DIMS_DATA_PLACEHOLDER__", json.dumps(dims_sorted))
        return html

    @staticmethod
    def save(reports: List[Any], path: str, title: str = "Benchmark Comparison"):
        html = Dashboard.generate(reports, title)
        Path(path).write_text(html, encoding="utf-8")
