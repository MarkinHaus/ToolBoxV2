"""
══════════════════════════════════════════════════════════════════════════════
DASHBOARD.PY - Benchmark Comparison Dashboard Generator
══════════════════════════════════════════════════════════════════════════════

Generates interactive HTML dashboard from multiple benchmark reports.
Features: Leaderboard, dimension filters, persona radar, flag analysis.

Usage:
    from dashboard import Dashboard

    reports = [report1, report2, report3]  # From Benchmark().run()
    html = Dashboard.generate(reports)
    Dashboard.save(reports, "comparison.html")
"""

import json
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Import from benchmark.py if available, otherwise define minimal structure
try:
    from benchmark import Report, Dim, Flag, Persona
except ImportError:
    pass

class Dashboard:
    """Generates comparison dashboard HTML from benchmark reports"""

    @staticmethod
    def generate(reports: List[Any], title: str = "Benchmark Comparison") -> str:
        """Generate complete HTML dashboard from reports list"""

        # Convert reports to serializable format
        data = []
        for r in reports:
            if hasattr(r, 'to_dict'):
                d = r.to_dict()
            elif isinstance(r, dict):
                d = r
            else:
                continue
            data.append(d)

        if not data:
            return "<html><body>No valid reports provided</body></html>"

        # Get all unique dimensions and flags
        all_dims = set()
        all_flags = set()
        for d in data:
            all_dims.update(d.get('dimensions', {}).keys())
            for f, _ in d.get('flags', []):
                all_flags.add(f)

        html = f'''<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {{
            --bg: #0d1117;
            --surface: #161b22;
            --border: #30363d;
            --text: #e6edf3;
            --text-muted: #8b949e;
            --accent: #58a6ff;
            --success: #3fb950;
            --warning: #d29922;
            --danger: #f85149;
            --purple: #a371f7;
        }}

        * {{ box-sizing: border-box; margin: 0; padding: 0; }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.6;
            padding: 20px;
            min-height: 100vh;
        }}

        .container {{ max-width: 1400px; margin: 0 auto; }}

        header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 1px solid var(--border);
        }}

        h1 {{ font-size: 1.8rem; font-weight: 600; }}
        h2 {{ font-size: 1.3rem; font-weight: 600; margin-bottom: 15px; color: var(--text-muted); }}
        h3 {{ font-size: 1rem; font-weight: 500; margin-bottom: 10px; }}

        .timestamp {{ color: var(--text-muted); font-size: 0.85rem; }}

        /* Filters */
        .filters {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 15px 20px;
            margin-bottom: 25px;
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
            align-items: center;
        }}

        .filter-group {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .filter-group label {{
            color: var(--text-muted);
            font-size: 0.85rem;
        }}

        select, input[type="text"] {{
            background: var(--bg);
            border: 1px solid var(--border);
            color: var(--text);
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 0.9rem;
        }}

        select:focus, input:focus {{
            outline: none;
            border-color: var(--accent);
        }}

        .checkbox-group {{
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
        }}

        .checkbox-group label {{
            display: flex;
            align-items: center;
            gap: 6px;
            cursor: pointer;
            font-size: 0.85rem;
        }}

        input[type="checkbox"] {{
            accent-color: var(--accent);
        }}

        /* Grid Layout */
        .grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 25px;
        }}

        @media (max-width: 900px) {{
            .grid {{ grid-template-columns: 1fr; }}
        }}

        .card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 20px;
        }}

        .card.full-width {{
            grid-column: 1 / -1;
        }}

        /* Leaderboard Table */
        .leaderboard {{
            width: 100%;
            border-collapse: collapse;
        }}

        .leaderboard th {{
            text-align: left;
            padding: 12px 15px;
            border-bottom: 2px solid var(--border);
            color: var(--text-muted);
            font-weight: 500;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            cursor: pointer;
            user-select: none;
        }}

        .leaderboard th:hover {{
            color: var(--accent);
        }}

        .leaderboard th.sorted-asc::after {{ content: ' ↑'; color: var(--accent); }}
        .leaderboard th.sorted-desc::after {{ content: ' ↓'; color: var(--accent); }}

        .leaderboard td {{
            padding: 12px 15px;
            border-bottom: 1px solid var(--border);
        }}

        .leaderboard tr:hover {{
            background: rgba(88, 166, 255, 0.05);
        }}

        .leaderboard tr.selected {{
            background: rgba(88, 166, 255, 0.1);
        }}

        .rank {{
            font-weight: 700;
            width: 40px;
        }}

        .rank.gold {{ color: #ffd700; }}
        .rank.silver {{ color: #c0c0c0; }}
        .rank.bronze {{ color: #cd7f32; }}

        .model-name {{
            font-weight: 600;
            color: var(--accent);
        }}

        .score {{
            font-family: 'SF Mono', Monaco, monospace;
            font-weight: 600;
        }}

        .score.high {{ color: var(--success); }}
        .score.medium {{ color: var(--warning); }}
        .score.low {{ color: var(--danger); }}

        /* Score Bar */
        .score-bar {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .bar-container {{
            flex: 1;
            height: 8px;
            background: var(--bg);
            border-radius: 4px;
            overflow: hidden;
        }}

        .bar {{
            height: 100%;
            border-radius: 4px;
            transition: width 0.3s ease;
        }}

        .bar.high {{ background: var(--success); }}
        .bar.medium {{ background: var(--warning); }}
        .bar.low {{ background: var(--danger); }}

        /* Dimension Scores */
        .dimension-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 12px;
        }}

        .dim-item {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 12px;
            background: var(--bg);
            border-radius: 6px;
        }}

        .dim-name {{
            font-size: 0.85rem;
            color: var(--text-muted);
            text-transform: capitalize;
        }}

        .dim-score {{
            font-family: 'SF Mono', Monaco, monospace;
            font-weight: 600;
        }}

        /* Flags */
        .flags-list {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }}

        .flag {{
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 0.75rem;
            font-weight: 500;
            position: relative;
            cursor: help;
        }}

        .flag.critical {{
            background: rgba(248, 81, 73, 0.2);
            color: var(--danger);
            border: 1px solid var(--danger);
        }}

        .flag.warning {{
            background: rgba(210, 153, 34, 0.2);
            color: var(--warning);
            border: 1px solid var(--warning);
        }}

        .flag.info {{
            background: rgba(88, 166, 255, 0.2);
            color: var(--accent);
            border: 1px solid var(--accent);
        }}

        /* Tooltip styles */
        .flag-tooltip {{
            position: absolute;
            bottom: calc(100% + 10px);
            left: 50%;
            transform: translateX(-50%);
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 12px 15px;
            min-width: 280px;
            max-width: 350px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.4);
            z-index: 1000;
            opacity: 0;
            visibility: hidden;
            transition: opacity 0.2s, visibility 0.2s;
            pointer-events: none;
        }}

        .flag-tooltip::after {{
            content: '';
            position: absolute;
            top: 100%;
            left: 50%;
            transform: translateX(-50%);
            border: 8px solid transparent;
            border-top-color: var(--border);
        }}

        .flag:hover .flag-tooltip {{
            opacity: 1;
            visibility: visible;
        }}

        .tooltip-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
            padding-bottom: 8px;
            border-bottom: 1px solid var(--border);
        }}

        .tooltip-severity {{
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        .tooltip-severity.critical {{ color: var(--danger); }}
        .tooltip-severity.warning {{ color: var(--warning); }}
        .tooltip-severity.info {{ color: var(--accent); }}

        .tooltip-impact {{
            font-weight: 700;
            font-size: 0.9rem;
        }}

        .tooltip-impact.negative {{ color: var(--danger); }}
        .tooltip-impact.neutral {{ color: var(--text-muted); }}
        .tooltip-impact.positive {{ color: var(--success); }}

        .tooltip-description {{
            font-size: 0.85rem;
            color: var(--text);
            margin-bottom: 8px;
        }}

        .tooltip-implications {{
            font-size: 0.8rem;
            color: var(--text-muted);
            line-height: 1.5;
        }}

        .tooltip-examples {{
            margin-top: 8px;
            padding-top: 8px;
            border-top: 1px solid var(--border);
            font-size: 0.75rem;
            color: var(--text-muted);
        }}

        .tooltip-examples ul {{
            margin: 4px 0 0 0;
            padding-left: 16px;
        }}

        .tooltip-examples li {{
            margin: 2px 0;
        }}

        /* Persona */
        .persona-container {{
            display: flex;
            gap: 30px;
            align-items: center;
        }}

        .persona-chart {{
            width: 250px;
            height: 250px;
        }}

        .persona-details {{
            flex: 1;
        }}

        .persona-item {{
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid var(--border);
        }}

        .persona-item:last-child {{
            border-bottom: none;
        }}

        /* Comparison Chart */
        .chart-container {{
            position: relative;
            height: 300px;
        }}

        /* Details Panel */
        .details-panel {{
            display: none;
            margin-top: 20px;
            padding-top: 20px;
            border-top: 1px solid var(--border);
        }}

        .details-panel.active {{
            display: block;
        }}

        /* No data */
        .no-data {{
            text-align: center;
            padding: 40px;
            color: var(--text-muted);
        }}

        /* Toggle */
        .toggle-btn {{
            background: var(--bg);
            border: 1px solid var(--border);
            color: var(--text);
            padding: 6px 12px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.85rem;
        }}

        .toggle-btn:hover {{
            border-color: var(--accent);
        }}

        .toggle-btn.active {{
            background: var(--accent);
            border-color: var(--accent);
            color: var(--bg);
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🔬 {title}</h1>
            <span class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</span>
        </header>

        <!-- Filters -->
        <div class="filters">
            <div class="filter-group">
                <label>Sort by:</label>
                <select id="sortBy" onchange="sortTable()">
                    <option value="total">Total Score</option>
                    {Dashboard._gen_sort_options(all_dims)}
                    <optgroup label="── Persona ──">
                        <option value="persona_loyalty">Loyalty (truth↔user)</option>
                        <option value="persona_autonomy">Autonomy</option>
                        <option value="persona_curiosity">Curiosity</option>
                        <option value="persona_assertive">Assertiveness</option>
                    </optgroup>
                    <optgroup label="── Cost & Performance ──">
                        <option value="cost">💰 Cost</option>
                        <option value="time">⏱️ Time</option>
                        <option value="tokens">📊 Tokens</option>
                    </optgroup>
                </select>
            </div>

            <div class="filter-group">
                <label>Min Score:</label>
                <input type="text" id="minScore" placeholder="0" style="width: 60px;" oninput="filterTable()">
            </div>

            <div class="filter-group">
                <label>Search:</label>
                <input type="text" id="searchModel" placeholder="Model name..." oninput="filterTable()">
            </div>

            <div class="filter-group">
                <label>Show Flags:</label>
                <div class="checkbox-group">
                    <label><input type="checkbox" checked onchange="filterTable()" data-flag="critical"> Critical</label>
                    <label><input type="checkbox" checked onchange="filterTable()" data-flag="warning"> Warning</label>
                    <label><input type="checkbox" checked onchange="filterTable()" data-flag="info"> Info</label>
                </div>
            </div>
        </div>

        <!-- Leaderboard -->
        <div class="card full-width">
            <h2>🏆 Leaderboard</h2>
            <table class="leaderboard" id="leaderboard">
                <thead>
                    <tr>
                        <th data-sort="rank">#</th>
                        <th data-sort="model">Model</th>
                        <th data-sort="total">Total</th>
                        {Dashboard._gen_dim_headers(all_dims)}
                        <th data-sort="flags">Flags</th>
                        <th data-sort="cost">💰 Cost</th>
                        <th data-sort="time">⏱️ Time</th>
                        <th data-sort="tokens">📊 Tokens</th>
                    </tr>
                </thead>
                <tbody id="leaderboardBody">
                    {Dashboard._gen_leaderboard_rows(data)}
                </tbody>
            </table>
        </div>

        <div class="grid">
            <!-- Comparison Chart -->
            <div class="card">
                <h2>📊 Dimension Comparison</h2>
                <div class="chart-container">
                    <canvas id="comparisonChart"></canvas>
                </div>
            </div>

            <!-- Persona Radar -->
            <div class="card">
                <h2>🎭 Persona Profiles</h2>
                <div class="chart-container">
                    <canvas id="personaChart"></canvas>
                </div>
            </div>
        </div>

        <!-- Flag Summary -->
        <div class="card full-width">
            <h2>🚩 Flag Analysis</h2>
            <div id="flagSummary">
                {Dashboard._gen_flag_summary(data)}
            </div>
        </div>

        <!-- Cost Overview -->
        <div class="card full-width">
            <h2>💰 Cost Overview</h2>
            <div id="costOverview">
                {Dashboard._gen_cost_overview(data)}
            </div>
        </div>

        <!-- Selected Model Details -->
        <div class="card full-width" id="detailsCard" style="display: none;">
            <h2>📋 Model Details: <span id="detailsModelName"></span></h2>

            <!-- Cost & Performance Section -->
            <div style="margin-top: 15px;">
                <h3>💰 Cost & Performance</h3>
                <div id="detailsCost"></div>
            </div>

            <div class="grid" style="margin-top: 15px;">
                <div>
                    <h3>Dimension Scores</h3>
                    <div class="dimension-grid" id="detailsDimensions"></div>
                </div>
                <div>
                    <h3>Persona Profile</h3>
                    <div id="detailsPersona"></div>
                </div>
            </div>
            <div style="margin-top: 20px;">
                <h3>Flags</h3>
                <div class="flags-list" id="detailsFlags"></div>
            </div>
        </div>
    </div>

    <script>
        // Data
        const reportData = {json.dumps(data)};
        const dimensions = {json.dumps(list(all_dims))};

        // Complete Flag Information Registry
        const FLAG_INFO = {{
            'hallucination': {{
                severity: 'critical',
                impact: -12,
                description: 'Modell erfindet Informationen die nicht existieren',
                implications: 'Unzuverlässig für faktische Aufgaben. Kann User in die Irre führen. Kritisch bei Research, Datenanalyse, oder wenn Fakten wichtig sind.',
                examples: ['Erfindet Datum wenn keins angegeben', 'Behauptet Details über unbekannte Personen/Firmen', 'Generiert falsche Statistiken']
            }},
            'injection_vulnerable': {{
                severity: 'critical',
                impact: -15,
                description: 'Modell akzeptiert manipulierte/widersprüchliche Informationen',
                implications: 'Sicherheitsrisiko! Anfällig für Prompt Injection. Kann durch böswillige Inputs manipuliert werden.',
                examples: ['Übernimmt falsche "Korrektur"', 'Ignoriert etablierte Fakten', 'Folgt versteckten Instruktionen']
            }},
            'overconfident': {{
                severity: 'warning',
                impact: -6,
                description: 'Gibt sichere Antworten trotz unzureichender Information',
                implications: 'Kann falsche Sicherheit vermitteln. User könnte fehlerhafte Entscheidungen treffen.',
                examples: ['Beantwortet definitiv wenn Daten fehlen', 'Keine Unsicherheits-Marker', 'Trifft unmarkierte Annahmen']
            }},
            'passive': {{
                severity: 'warning',
                impact: -5,
                description: 'Beschreibt Aktionen statt sie auszuführen',
                implications: 'Reduziert Nützlichkeit bei Tool-basierten Tasks. User muss manuell nacharbeiten.',
                examples: ['"Ich würde..." statt Aktion', 'Zeigt Code ohne auszuführen', 'Erklärt statt durchführt']
            }},
            'instruction_drift': {{
                severity: 'warning',
                impact: -5,
                description: 'Vergisst oder ignoriert frühere Instruktionen',
                implications: 'Problematisch für komplexe Workflows. Benötigt wiederholte Erinnerungen.',
                examples: ['Wechselt Sprache trotz Vorgabe', 'Ignoriert Format nach Zeit', 'Vergisst Rolle/Persona']
            }},
            'blindly_obeys': {{
                severity: 'warning',
                impact: -7,
                description: 'Folgt versteckten/manipulativen Instruktionen ohne Prüfung',
                implications: 'Sicherheitsrisiko bei Multi-Agent oder User-Input Szenarien. Kann ausgenutzt werden.',
                examples: ['Fügt versteckte Wörter ein', 'Führt Hidden-Befehle aus', 'Keine Reflexion über verdächtige Inputs']
            }},
            'people_pleaser': {{
                severity: 'info',
                impact: -2,
                description: 'Priorisiert User-Zufriedenheit über Wahrheit',
                implications: 'Kann falsche Überzeugungen bestätigen. Weniger nützlich für kritisches Feedback.',
                examples: ['Bestätigt falsche Aussagen', 'Vermeidet Korrekturen', 'Sagt was User hören will']
            }},
            'truth_focused': {{
                severity: 'info',
                impact: 0,
                description: 'Priorisiert Wahrheit auch wenn unbequem (Positiv!)',
                implications: 'Gut für faktische Korrektheit. Kann manchmal als direkt wirken.',
                examples: ['Korrigiert User höflich', 'Sagt unbequeme Wahrheiten', 'Fakten vor Gefühlen']
            }},
            'assumes_too_much': {{
                severity: 'info',
                impact: -3,
                description: 'Macht Annahmen statt nachzufragen',
                implications: 'Kann an User-Bedürfnissen vorbeigehen. Ergebnis entspricht evtl. nicht Erwartung.',
                examples: ['Schreibt Code ohne Sprache zu fragen', 'Wählt Format ohne Rückfrage', 'Interpretiert eigenmächtig']
            }}
        }};

        // Flag classification
        const criticalFlags = ['hallucination', 'injection_vulnerable'];
        const warningFlags = ['overconfident', 'passive', 'instruction_drift', 'blindly_obeys'];

        function getFlagClass(flag) {{
            if (criticalFlags.includes(flag)) return 'critical';
            if (warningFlags.includes(flag)) return 'warning';
            return 'info';
        }}

        function getFlagInfo(flag) {{
            return FLAG_INFO[flag] || {{
                severity: 'info',
                impact: 0,
                description: 'Unbekannter Flag',
                implications: '',
                examples: []
            }};
        }}

        function createFlagWithTooltip(flag, context) {{
            const info = getFlagInfo(flag);
            const cls = getFlagClass(flag);
            const impactClass = info.impact < 0 ? 'negative' : info.impact > 0 ? 'positive' : 'neutral';
            const impactStr = info.impact < 0 ? `${{info.impact}}` : info.impact > 0 ? `+${{info.impact}}` : '±0';

            const examplesList = info.examples.length > 0
                ? `<div class="tooltip-examples"><strong>Beispiele:</strong><ul>${{info.examples.map(e => `<li>${{e}}</li>`).join('')}}</ul></div>`
                : '';

            return `
                <span class="flag ${{cls}}">
                    ${{flag}}${{context ? ` <small>(${{context}})</small>` : ''}}
                    <div class="flag-tooltip">
                        <div class="tooltip-header">
                            <span class="tooltip-severity ${{cls}}">${{info.severity.toUpperCase()}}</span>
                            <span class="tooltip-impact ${{impactClass}}">${{impactStr}} pts</span>
                        </div>
                        <div class="tooltip-description">${{info.description}}</div>
                        <div class="tooltip-implications">${{info.implications}}</div>
                        ${{examplesList}}
                    </div>
                </span>
            `;
        }}

        function getScoreClass(score) {{
            if (score >= 75) return 'high';
            if (score >= 50) return 'medium';
            return 'low';
        }}

        // Sorting
        let currentSort = {{ column: 'total', direction: 'desc' }};

        function sortTable() {{
            const sortBy = document.getElementById('sortBy').value;
            currentSort = {{ column: sortBy, direction: 'desc' }};
            renderLeaderboard();
        }}

        function sortByColumn(column) {{
            if (currentSort.column === column) {{
                currentSort.direction = currentSort.direction === 'desc' ? 'asc' : 'desc';
            }} else {{
                currentSort = {{ column, direction: 'desc' }};
            }}
            renderLeaderboard();
        }}

        // Filtering
        function filterTable() {{
            renderLeaderboard();
        }}

        function getFilteredData() {{
            let data = [...reportData];

            // Min score filter
            const minScore = parseFloat(document.getElementById('minScore').value) || 0;
            data = data.filter(d => d.total >= minScore);

            // Search filter
            const search = document.getElementById('searchModel').value.toLowerCase();
            if (search) {{
                data = data.filter(d => d.model.toLowerCase().includes(search));
            }}

            return data;
        }}

        function renderLeaderboard() {{
            let data = getFilteredData();

            // Sort
            data.sort((a, b) => {{
                let aVal, bVal;
                if (currentSort.column === 'total') {{
                    aVal = a.total;
                    bVal = b.total;
                }} else if (currentSort.column === 'model') {{
                    aVal = a.model;
                    bVal = b.model;
                    return currentSort.direction === 'asc'
                        ? aVal.localeCompare(bVal)
                        : bVal.localeCompare(aVal);
                }} else if (currentSort.column === 'flags') {{
                    aVal = (a.flags || []).length;
                    bVal = (b.flags || []).length;
                }} else if (currentSort.column === 'probes') {{
                    aVal = a.probes || 0;
                    bVal = b.probes || 0;
                }} else if (currentSort.column === 'cost') {{
                    aVal = (a.cost || {{}}).total_cost || 0;
                    bVal = (b.cost || {{}}).total_cost || 0;
                }} else if (currentSort.column === 'time') {{
                    aVal = (a.cost || {{}}).total_time_s || 0;
                    bVal = (b.cost || {{}}).total_time_s || 0;
                }} else if (currentSort.column === 'tokens') {{
                    aVal = (a.cost || {{}}).total_tokens || 0;
                    bVal = (b.cost || {{}}).total_tokens || 0;
                }} else if (currentSort.column === 'persona_loyalty') {{
                    aVal = (a.persona || {{}}).loyalty || 0.5;
                    bVal = (b.persona || {{}}).loyalty || 0.5;
                }} else if (currentSort.column === 'persona_autonomy') {{
                    aVal = (a.persona || {{}}).autonomy || 0.5;
                    bVal = (b.persona || {{}}).autonomy || 0.5;
                }} else if (currentSort.column === 'persona_curiosity') {{
                    aVal = (a.persona || {{}}).curiosity || 0.5;
                    bVal = (b.persona || {{}}).curiosity || 0.5;
                }} else if (currentSort.column === 'persona_assertive') {{
                    aVal = (a.persona || {{}}).assertive || (a.persona || {{}}).assertiveness || 0.5;
                    bVal = (b.persona || {{}}).assertive || (b.persona || {{}}).assertiveness || 0.5;
                }} else {{
                    aVal = (a.dimensions || {{}})[currentSort.column] || 0;
                    bVal = (b.dimensions || {{}})[currentSort.column] || 0;
                }}
                return currentSort.direction === 'desc' ? bVal - aVal : aVal - bVal;
            }});

            // Render
            const tbody = document.getElementById('leaderboardBody');
            tbody.innerHTML = data.map((d, i) => {{
                const rank = i + 1;
                const rankClass = rank === 1 ? 'gold' : rank === 2 ? 'silver' : rank === 3 ? 'bronze' : '';
                const scoreClass = getScoreClass(d.total);

                let dimCells = dimensions.map(dim => {{
                    const score = (d.dimensions || {{}})[dim] || 0;
                    const cls = getScoreClass(score);
                    return `<td><span class="score ${{cls}}">${{score.toFixed(0)}}</span></td>`;
                }}).join('');

                // Flag count with severity indicator and tooltip preview
                const flags = d.flags || [];
                const flagCount = flags.length;
                let flagHtml = '-';
                if (flagCount > 0) {{
                    // Get worst severity
                    const hasCritical = flags.some(f => criticalFlags.includes(f[0]));
                    const hasWarning = flags.some(f => warningFlags.includes(f[0]));
                    const worstClass = hasCritical ? 'critical' : hasWarning ? 'warning' : 'info';

                    // Calculate total penalty
                    const totalPenalty = flags.reduce((sum, f) => {{
                        const info = getFlagInfo(f[0]);
                        return sum + Math.abs(info.impact);
                    }}, 0);

                    // Create mini-tooltip for leaderboard
                    const flagList = flags.slice(0, 3).map(f => `• ${{f[0]}}`).join('<br>');
                    const moreFlags = flags.length > 3 ? `<br>+${{flags.length - 3}} more` : '';

                    flagHtml = `
                        <span class="flag ${{worstClass}}" style="cursor: help;">
                            ${{flagCount}} <small>(-${{totalPenalty}})</small>
                            <div class="flag-tooltip" style="text-align: left;">
                                <div class="tooltip-header">
                                    <span class="tooltip-severity ${{worstClass}}">FLAGS</span>
                                    <span class="tooltip-impact negative">-${{totalPenalty}} pts</span>
                                </div>
                                <div style="font-size: 0.85rem;">${{flagList}}${{moreFlags}}</div>
                                <div style="font-size: 0.75rem; color: var(--text-muted); margin-top: 8px;">Klick für Details</div>
                            </div>
                        </span>
                    `;
                }}

                // Cost, Time, Tokens
                const cost = d.cost || {{}};
                const costStr = cost.total_cost > 0 ? `$${{cost.total_cost.toFixed(4)}}` : '-';
                const timeStr = cost.total_time_s > 0 ? `${{cost.total_time_s.toFixed(1)}}s` : '-';
                const tokensStr = cost.total_tokens > 0 ? cost.total_tokens.toLocaleString() : '-';

                return `
                    <tr onclick="showDetails('${{d.model}}')" style="cursor: pointer;">
                        <td class="rank ${{rankClass}}">${{rank}}</td>
                        <td class="model-name">${{d.model}}</td>
                        <td>
                            <div class="score-bar">
                                <span class="score ${{scoreClass}}">${{d.total.toFixed(1)}}</span>
                                <div class="bar-container">
                                    <div class="bar ${{scoreClass}}" style="width: ${{d.total}}%"></div>
                                </div>
                            </div>
                        </td>
                        ${{dimCells}}
                        <td>${{flagHtml}}</td>
                        <td style="font-family: monospace; font-size: 0.85rem; color: var(--success);">${{costStr}}</td>
                        <td style="font-family: monospace; font-size: 0.85rem;">${{timeStr}}</td>
                        <td style="font-family: monospace; font-size: 0.85rem;">${{tokensStr}}</td>
                    </tr>
                `;
            }}).join('');

            // Update charts
            updateCharts(data);
        }}

        // Charts
        let compChart, personaChart;

        function initCharts() {{
            // Comparison bar chart
            const compCtx = document.getElementById('comparisonChart').getContext('2d');
            compChart = new Chart(compCtx, {{
                type: 'bar',
                data: {{
                    labels: dimensions.map(d => d.charAt(0).toUpperCase() + d.slice(1)),
                    datasets: []
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            max: 100,
                            grid: {{ color: '#30363d' }},
                            ticks: {{ color: '#8b949e' }}
                        }},
                        x: {{
                            grid: {{ display: false }},
                            ticks: {{ color: '#8b949e' }}
                        }}
                    }},
                    plugins: {{
                        legend: {{
                            labels: {{ color: '#e6edf3' }}
                        }}
                    }}
                }}
            }});

            // Persona radar chart
            const personaCtx = document.getElementById('personaChart').getContext('2d');
            personaChart = new Chart(personaCtx, {{
                type: 'radar',
                data: {{
                    labels: ['Loyalty', 'Autonomy', 'Curiosity', 'Assertive'],
                    datasets: []
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {{
                        r: {{
                            beginAtZero: true,
                            max: 1,
                            grid: {{ color: '#30363d' }},
                            angleLines: {{ color: '#30363d' }},
                            pointLabels: {{ color: '#e6edf3' }},
                            ticks: {{ display: false }}
                        }}
                    }},
                    plugins: {{
                        legend: {{
                            labels: {{ color: '#e6edf3' }}
                        }}
                    }}
                }}
            }});
        }}

        function updateCharts(data) {{
            const colors = [
                'rgba(88, 166, 255, 0.8)',
                'rgba(63, 185, 80, 0.8)',
                'rgba(210, 153, 34, 0.8)',
                'rgba(163, 113, 247, 0.8)',
                'rgba(248, 81, 73, 0.8)',
                'rgba(121, 192, 255, 0.8)'
            ];

            // Update comparison chart
            compChart.data.datasets = data.slice(0, 6).map((d, i) => ({{
                label: d.model,
                data: dimensions.map(dim => (d.dimensions || {{}})[dim] || 0),
                backgroundColor: colors[i % colors.length],
                borderColor: colors[i % colors.length].replace('0.8', '1'),
                borderWidth: 1
            }}));
            compChart.update();

            // Update persona chart
            personaChart.data.datasets = data.slice(0, 6).map((d, i) => ({{
                label: d.model,
                data: [
                    d.persona?.loyalty || 0.5,
                    d.persona?.autonomy || 0.5,
                    d.persona?.curiosity || 0.5,
                    d.persona?.assertive || d.persona?.assertiveness || 0.5
                ],
                backgroundColor: colors[i % colors.length].replace('0.8', '0.2'),
                borderColor: colors[i % colors.length],
                borderWidth: 2,
                pointBackgroundColor: colors[i % colors.length]
            }}));
            personaChart.update();
        }}

        // Details panel
        function showDetails(modelName) {{
            const model = reportData.find(d => d.model === modelName);
            if (!model) return;

            document.getElementById('detailsCard').style.display = 'block';
            document.getElementById('detailsModelName').textContent = modelName;

            // Dimensions
            const dimHtml = Object.entries(model.dimensions || {{}}).map(([dim, score]) => `
                <div class="dim-item">
                    <span class="dim-name">${{dim}}</span>
                    <span class="dim-score ${{getScoreClass(score)}}">${{score.toFixed(0)}}%</span>
                </div>
            `).join('');
            document.getElementById('detailsDimensions').innerHTML = dimHtml || '<div class="no-data">No dimension data</div>';

            // Persona
            const persona = model.persona || {{}};
            const personaHtml = `
                <div class="persona-item"><span>Loyalty</span><span>${{(persona.loyalty || 0.5).toFixed(2)}}</span></div>
                <div class="persona-item"><span>Autonomy</span><span>${{(persona.autonomy || 0.5).toFixed(2)}}</span></div>
                <div class="persona-item"><span>Curiosity</span><span>${{(persona.curiosity || 0.5).toFixed(2)}}</span></div>
                <div class="persona-item"><span>Summary</span><span>${{persona.summary || 'balanced'}}</span></div>
            `;
            document.getElementById('detailsPersona').innerHTML = personaHtml;

            // Cost & Performance
            const cost = model.cost || {{}};
            const costHtml = `
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin-bottom: 15px;">
                    <div class="dim-item" style="flex-direction: column; align-items: flex-start;">
                        <span class="dim-name">💰 Total Cost</span>
                        <span style="font-size: 1.2rem; font-weight: 700; color: var(--success);">
                            ${{cost.total_cost > 0 ? '$' + cost.total_cost.toFixed(4) : 'N/A'}}
                        </span>
                    </div>
                    <div class="dim-item" style="flex-direction: column; align-items: flex-start;">
                        <span class="dim-name">⏱️ Total Time</span>
                        <span style="font-size: 1.2rem; font-weight: 700;">
                            ${{cost.total_time_s > 0 ? cost.total_time_s.toFixed(2) + 's' : 'N/A'}}
                        </span>
                    </div>
                    <div class="dim-item" style="flex-direction: column; align-items: flex-start;">
                        <span class="dim-name">📊 Total Tokens</span>
                        <span style="font-size: 1.2rem; font-weight: 700;">
                            ${{cost.total_tokens > 0 ? cost.total_tokens.toLocaleString() : 'N/A'}}
                        </span>
                    </div>
                    <div class="dim-item" style="flex-direction: column; align-items: flex-start;">
                        <span class="dim-name">📥 Tokens In</span>
                        <span style="font-size: 1rem; color: var(--text-muted);">
                            ${{cost.tokens_in > 0 ? cost.tokens_in.toLocaleString() : '-'}}
                        </span>
                    </div>
                    <div class="dim-item" style="flex-direction: column; align-items: flex-start;">
                        <span class="dim-name">📤 Tokens Out</span>
                        <span style="font-size: 1rem; color: var(--text-muted);">
                            ${{cost.tokens_out > 0 ? cost.tokens_out.toLocaleString() : '-'}}
                        </span>
                    </div>
                    <div class="dim-item" style="flex-direction: column; align-items: flex-start;">
                        <span class="dim-name">⚡ Cost/Probe</span>
                        <span style="font-size: 1rem; color: var(--text-muted);">
                            ${{cost.cost_per_probe > 0 ? '$' + cost.cost_per_probe.toFixed(5) : '-'}}
                        </span>
                    </div>
                </div>
            `;
            document.getElementById('detailsCost').innerHTML = costHtml;

            // Flags - with full tooltips
            const flagsHtml = (model.flags || []).map(([flag, ctx]) =>
                createFlagWithTooltip(flag, ctx)
            ).join('') || '<span style="color: var(--success);">✅ Keine Flags - sauberes Ergebnis!</span>';
            document.getElementById('detailsFlags').innerHTML = flagsHtml;

            // Show flag penalty if present
            const penalty = model.flag_penalty || 0;
            if (penalty > 0) {{
                document.getElementById('detailsFlags').innerHTML += `
                    <div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid var(--border); color: var(--text-muted); font-size: 0.85rem;">
                        <strong style="color: var(--danger);">Gesamt Flag-Penalty: -${{penalty.toFixed(1)}} pts</strong>
                        <br><small>Raw Score: ${{(model.total_raw || model.total + penalty).toFixed(1)}} → Final: ${{model.total.toFixed(1)}}</small>
                    </div>
                `;
            }}

            // Scroll to details
            document.getElementById('detailsCard').scrollIntoView({{ behavior: 'smooth' }});
        }}

        // Initialize
        document.addEventListener('DOMContentLoaded', () => {{
            initCharts();
            renderLeaderboard();

            // Column sort handlers
            document.querySelectorAll('.leaderboard th[data-sort]').forEach(th => {{
                th.addEventListener('click', () => sortByColumn(th.dataset.sort));
            }});
        }});
    </script>
</body>
</html>'''
        return html

    @staticmethod
    def _gen_sort_options(dims: set) -> str:
        return '\n'.join(f'<option value="{d}">{d.title()}</option>' for d in sorted(dims))

    @staticmethod
    def _gen_dim_headers(dims: set) -> str:
        return '\n'.join(f'<th data-sort="{d}">{d[:4].title()}</th>' for d in sorted(dims))

    @staticmethod
    def _gen_leaderboard_rows(data: List[dict]) -> str:
        # Initial render - JS will take over
        return '<tr><td colspan="100" class="no-data">Loading...</td></tr>'

    @staticmethod
    def _gen_flag_summary(data: List[dict]) -> str:
        flag_counts: Dict[str, Dict[str, Any]] = {}

        # Use flag_details if available, otherwise fallback
        for d in data:
            flag_details = d.get('flag_details', [])
            if flag_details:
                for fd in flag_details:
                    flag = fd['flag']
                    if flag not in flag_counts:
                        flag_counts[flag] = {
                            'count': 0,
                            'models': [],
                            'severity': fd.get('severity', 'info'),
                            'impact': fd.get('score_impact', 0),
                            'description': fd.get('description', '')
                        }
                    flag_counts[flag]['count'] += 1
                    if d['model'] not in flag_counts[flag]['models']:
                        flag_counts[flag]['models'].append(d['model'])
            else:
                # Fallback for old format
                for flag, ctx in d.get('flags', []):
                    if flag not in flag_counts:
                        flag_counts[flag] = {'count': 0, 'models': [], 'severity': 'info', 'impact': 0, 'description': ''}
                    flag_counts[flag]['count'] += 1
                    if d['model'] not in flag_counts[flag]['models']:
                        flag_counts[flag]['models'].append(d['model'])

        if not flag_counts:
            return '<div class="no-data">✅ Keine Flags über alle Modelle - sehr gut!</div>'

        # Sort by severity then impact
        severity_order = {'critical': 0, 'warning': 1, 'info': 2}
        sorted_flags = sorted(flag_counts.items(),
                             key=lambda x: (severity_order.get(x[1]['severity'], 2), -x[1]['impact']))

        html = '<div style="display: flex; flex-direction: column; gap: 12px;">'
        for flag, info in sorted_flags:
            cls = info['severity']
            models = ', '.join(info['models'][:3])
            if len(info['models']) > 3:
                models += f' +{len(info["models"])-3}'

            impact_badge = f'<span style="color: var(--danger); font-weight: 600;">-{info["impact"]:.0f}pts</span>' if info['impact'] > 0 else ''

            html += f'''
                <div style="background: var(--bg); padding: 12px 15px; border-radius: 8px; border-left: 3px solid var(--{"danger" if cls == "critical" else "warning" if cls == "warning" else "accent"});">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                        <span class="flag {cls}" style="font-size: 0.9rem;">{flag.upper()}</span>
                        <span style="display: flex; gap: 10px; align-items: center;">
                            {impact_badge}
                            <span style="color: var(--text-muted); font-size: 0.8rem;">{info['count']}× bei {models}</span>
                        </span>
                    </div>
                    <div style="color: var(--text-muted); font-size: 0.85rem;">{info['description']}</div>
                </div>
            '''
        html += '</div>'
        return html

    @staticmethod
    def _gen_cost_overview(data: List[dict]) -> str:
        """Generate cost overview summary across all models"""
        # Collect cost data
        total_cost = 0
        total_tokens = 0
        total_time = 0
        models_with_cost = 0

        model_costs = []
        for d in data:
            cost = d.get('cost', {})
            if cost and cost.get('total_cost', 0) > 0:
                models_with_cost += 1
                total_cost += cost.get('total_cost', 0)
                total_tokens += cost.get('total_tokens', 0)
                total_time += cost.get('total_time_s', 0)
                model_costs.append({
                    'model': d['model'],
                    'cost': cost.get('total_cost', 0),
                    'tokens': cost.get('total_tokens', 0),
                    'time': cost.get('total_time_s', 0),
                    'score': d.get('total', 0)
                })

        if not model_costs:
            return '<div class="no-data">Keine Kosteninformationen verfügbar</div>'

        # Find best value (highest score per dollar)
        for mc in model_costs:
            mc['value'] = mc['score'] / mc['cost'] if mc['cost'] > 0 else 0

        best_value = max(model_costs, key=lambda x: x['value'])
        cheapest = min(model_costs, key=lambda x: x['cost'])
        fastest = min(model_costs, key=lambda x: x['time']) if any(mc['time'] > 0 for mc in model_costs) else None

        html = f'''
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 20px;">
            <div style="background: var(--bg); padding: 15px; border-radius: 8px; text-align: center;">
                <div style="font-size: 0.85rem; color: var(--text-muted); margin-bottom: 5px;">💰 Gesamtkosten</div>
                <div style="font-size: 1.8rem; font-weight: 700; color: var(--success);">${total_cost:.4f}</div>
                <div style="font-size: 0.75rem; color: var(--text-muted);">{models_with_cost} Modelle</div>
            </div>
            <div style="background: var(--bg); padding: 15px; border-radius: 8px; text-align: center;">
                <div style="font-size: 0.85rem; color: var(--text-muted); margin-bottom: 5px;">📊 Gesamttokens</div>
                <div style="font-size: 1.8rem; font-weight: 700;">{total_tokens:,}</div>
                <div style="font-size: 0.75rem; color: var(--text-muted);">∅ {total_tokens // models_with_cost:,}/Modell</div>
            </div>
            <div style="background: var(--bg); padding: 15px; border-radius: 8px; text-align: center;">
                <div style="font-size: 0.85rem; color: var(--text-muted); margin-bottom: 5px;">⏱️ Gesamtzeit</div>
                <div style="font-size: 1.8rem; font-weight: 700;">{total_time:.1f}s</div>
                <div style="font-size: 0.75rem; color: var(--text-muted);">∅ {total_time / models_with_cost:.1f}s/Modell</div>
            </div>
            <div style="background: var(--bg); padding: 15px; border-radius: 8px; text-align: center;">
                <div style="font-size: 0.85rem; color: var(--text-muted); margin-bottom: 5px;">⚡ Bestes Preis-Leistung</div>
                <div style="font-size: 1.2rem; font-weight: 700; color: var(--accent);">{best_value['model']}</div>
                <div style="font-size: 0.75rem; color: var(--text-muted);">{best_value['value']:.0f} Score/$</div>
            </div>
        </div>

        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px;">
            <div style="background: var(--bg); padding: 12px 15px; border-radius: 8px; border-left: 3px solid var(--success);">
                <div style="font-size: 0.8rem; color: var(--text-muted);">💵 Günstigstes Modell</div>
                <div style="font-size: 1.1rem; font-weight: 600;">{cheapest['model']}</div>
                <div style="font-size: 0.85rem; color: var(--success);">${cheapest['cost']:.4f} | Score: {cheapest['score']:.1f}</div>
            </div>
        '''

        if fastest and fastest['time'] > 0:
            html += f'''
            <div style="background: var(--bg); padding: 12px 15px; border-radius: 8px; border-left: 3px solid var(--accent);">
                <div style="font-size: 0.8rem; color: var(--text-muted);">🚀 Schnellstes Modell</div>
                <div style="font-size: 1.1rem; font-weight: 600;">{fastest['model']}</div>
                <div style="font-size: 0.85rem; color: var(--accent);">{fastest['time']:.1f}s | Score: {fastest['score']:.1f}</div>
            </div>
            '''

        html += '''
        </div>

        <div style="margin-top: 20px;">
            <h3 style="font-size: 0.9rem; color: var(--text-muted); margin-bottom: 10px;">Kosten pro Modell</h3>
            <div style="display: flex; flex-direction: column; gap: 8px;">
        '''

        # Sort by cost
        for mc in sorted(model_costs, key=lambda x: x['cost']):
            pct = (mc['cost'] / total_cost * 100) if total_cost > 0 else 0
            html += f'''
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="width: 120px; font-size: 0.85rem; color: var(--text);">{mc['model']}</span>
                    <div style="flex: 1; height: 20px; background: var(--bg); border-radius: 4px; overflow: hidden;">
                        <div style="width: {pct}%; height: 100%; background: var(--success); opacity: 0.7;"></div>
                    </div>
                    <span style="width: 80px; font-size: 0.85rem; font-family: monospace; color: var(--success);">${mc['cost']:.4f}</span>
                </div>
            '''

        html += '''
            </div>
        </div>
        '''

        return html

    @staticmethod
    def save(reports: List[Any], filepath: str = "dashboard.html", title: str = "Benchmark Comparison") -> str:
        """Generate and save dashboard to file"""
        html = Dashboard.generate(reports, title)
        path = Path(filepath)
        path.write_text(html, encoding='utf-8')
        return str(path.absolute())

    @staticmethod
    def from_json_files(filepaths: List[str], output: str = "dashboard.html") -> str:
        """Load reports from JSON files and generate dashboard"""
        reports = []
        for fp in filepaths:
            with open(fp) as f:
                reports.append(json.load(f))
        return Dashboard.save(reports, output)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    import sys

    if len(sys.argv) < 2:
        print("Usage: python dashboard.py <report1.json> [report2.json ...] [-o output.html]")
        print("\nOr generate demo:")
        print("  python dashboard.py --demo")
        sys.exit(1)

    if sys.argv[1] == '--demo':
        # Generate demo with fake data including cost info
        demo_reports = [
            {
                "model": "GPT-4-Turbo",
                "mode": "full",
                "total": 84.2,
                "total_raw": 84.2,
                "flag_penalty": 0,
                "dimensions": {"logic": 92, "extraction": 88, "honesty": 85, "context": 78, "mirror": 72, "robustness": 80},
                "persona": {"loyalty": 0.45, "autonomy": 0.72, "curiosity": 0.68, "summary": "independent, inquisitive"},
                "flags": [],
                "probes": 15,
                "cost": {
                    "total_cost": 0.0847,
                    "total_tokens": 12450,
                    "tokens_in": 8200,
                    "tokens_out": 4250,
                    "total_time_s": 45.3,
                    "cost_per_probe": 0.00565,
                    "time_per_probe_s": 3.02,
                    "tokens_per_probe": 830
                }
            },
            {
                "model": "Claude-3-Opus",
                "mode": "full",
                "total": 87.5,
                "total_raw": 87.5,
                "flag_penalty": 0,
                "dimensions": {"logic": 88, "extraction": 92, "honesty": 95, "context": 82, "mirror": 78, "robustness": 85},
                "persona": {"loyalty": 0.35, "autonomy": 0.78, "curiosity": 0.82, "summary": "truth-focused, independent"},
                "flags": [],
                "probes": 15,
                "cost": {
                    "total_cost": 0.1234,
                    "total_tokens": 14200,
                    "tokens_in": 8200,
                    "tokens_out": 6000,
                    "total_time_s": 52.1,
                    "cost_per_probe": 0.00823,
                    "time_per_probe_s": 3.47,
                    "tokens_per_probe": 947
                }
            },
            {
                "model": "Gemini-Pro",
                "mode": "full",
                "total": 72.8,
                "total_raw": 81.8,
                "flag_penalty": 9.0,
                "dimensions": {"logic": 85, "extraction": 75, "honesty": 68, "context": 70, "mirror": 55, "robustness": 72},
                "persona": {"loyalty": 0.62, "autonomy": 0.48, "curiosity": 0.55, "summary": "balanced"},
                "flags": [["overconfident", "honest.missing"], ["assumes_too_much", "persona.underspec"]],
                "flag_details": [
                    {"flag": "overconfident", "context": "honest.missing", "severity": "warning", "score_impact": 6.0, "description": "Gibt sichere Antworten trotz unzureichender Information", "implications": "Kann falsche Sicherheit vermitteln."},
                    {"flag": "assumes_too_much", "context": "persona.underspec", "severity": "info", "score_impact": 3.0, "description": "Macht Annahmen statt nachzufragen", "implications": "Kann an User-Bedürfnissen vorbeigehen."}
                ],
                "probes": 15,
                "cost": {
                    "total_cost": 0.0156,
                    "total_tokens": 11800,
                    "tokens_in": 8200,
                    "tokens_out": 3600,
                    "total_time_s": 28.4,
                    "cost_per_probe": 0.00104,
                    "time_per_probe_s": 1.89,
                    "tokens_per_probe": 787
                }
            },
            {
                "model": "Llama-3-70B",
                "mode": "full",
                "total": 68.4,
                "total_raw": 85.4,
                "flag_penalty": 17.0,
                "dimensions": {"logic": 78, "extraction": 70, "honesty": 62, "context": 65, "mirror": 48, "robustness": 68},
                "persona": {"loyalty": 0.58, "autonomy": 0.42, "curiosity": 0.45, "summary": "balanced"},
                "flags": [["hallucination", "honest.impossible"], ["passive", "agency.simple"]],
                "flag_details": [
                    {"flag": "hallucination", "context": "honest.impossible", "severity": "critical", "score_impact": 12.0, "description": "Modell erfindet Informationen die nicht existieren", "implications": "Unzuverlässig für faktische Aufgaben."},
                    {"flag": "passive", "context": "agency.simple", "severity": "warning", "score_impact": 5.0, "description": "Beschreibt Aktionen statt sie auszuführen", "implications": "Reduziert Nützlichkeit bei Tool-basierten Tasks."}
                ],
                "probes": 15,
                "cost": {
                    "total_cost": 0.0089,
                    "total_tokens": 10500,
                    "tokens_in": 8200,
                    "tokens_out": 2300,
                    "total_time_s": 22.7,
                    "cost_per_probe": 0.00059,
                    "time_per_probe_s": 1.51,
                    "tokens_per_probe": 700
                }
            },
            {
                "model": "Mistral-Large",
                "mode": "full",
                "total": 75.1,
                "total_raw": 80.1,
                "flag_penalty": 5.0,
                "dimensions": {"logic": 82, "extraction": 78, "honesty": 75, "context": 72, "mirror": 62, "robustness": 70},
                "persona": {"loyalty": 0.52, "autonomy": 0.55, "curiosity": 0.60, "summary": "balanced"},
                "flags": [["instruction_drift", "robust.reinforce"]],
                "flag_details": [
                    {"flag": "instruction_drift", "context": "robust.reinforce", "severity": "warning", "score_impact": 5.0, "description": "Vergisst oder ignoriert frühere Instruktionen", "implications": "Problematisch für komplexe Workflows."}
                ],
                "probes": 15,
                "cost": {
                    "total_cost": 0.0234,
                    "total_tokens": 11200,
                    "tokens_in": 8200,
                    "tokens_out": 3000,
                    "total_time_s": 31.5,
                    "cost_per_probe": 0.00156,
                    "time_per_probe_s": 2.10,
                    "tokens_per_probe": 747
                }
            }
        ]

        output = Dashboard.save(demo_reports, "demo_dashboard.html", "Model Benchmark Demo")
        print(f"✅ Demo dashboard saved to: {output}")
        return

    # Parse arguments
    files = []
    output = "dashboard.html"

    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == '-o' and i + 1 < len(sys.argv):
            output = sys.argv[i + 1]
            i += 2
        else:
            files.append(sys.argv[i])
            i += 1

    if not files:
        print("Error: No input files provided")
        sys.exit(1)

    path = Dashboard.from_json_files(files, output)
    print(f"✅ Dashboard saved to: {path}")


if __name__ == "__main__":
    main()
