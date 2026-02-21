"""
Memory Graph Visualizer for HybridMemoryStore (V2)

Provides advanced, highly performant visualization capabilities for entity relations
and knowledge graphs natively extracted from the SQLite backend.

Generates a standalone, interactive HTML dashboard using D3.js.
Strictly adheres to professional UI/UX guidelines:
- Dark mode (#08080d)
- IBM Plex typography
- Multi-edge support via curved links
- Degree-centrality based node scaling
- Accordion-style data inspection (no modals)
"""

import json
from typing import Any, Dict, List, Optional, Tuple

class MemoryGraphVisualizer:
    """
    Visualizes entity relations from HybridMemoryStore V2.
    Extracts directly from SQLite and generates a standalone D3.js dashboard.
    """

    def __init__(self, store, max_depth: int = 3):
        """
        Initialize visualizer with a HybridMemoryStore instance.

        Args:
            store: HybridMemoryStore instance (V2)
            max_depth: Default maximum depth for subgraph traversal
        """
        self.store = store
        self.max_depth = max_depth

    def get_all_entities(self) -> List[Dict[str, Any]]:
        """Retrieve all entities from the V2 store."""
        entities = []
        with self.store._tx() as conn:
            cursor = conn.execute(
                "SELECT id, entity_type, name, meta FROM entities ORDER BY name"
            )
            for row in cursor:
                entities.append({
                    "id": row[0],
                    "type": row[1],
                    "name": row[2],
                    "meta": json.loads(row[3]) if row[3] else {},
                })
        return entities

    def get_all_relations(self) -> List[Dict[str, Any]]:
        """Retrieve all relations from the V2 store."""
        relations = []
        with self.store._tx() as conn:
            cursor = conn.execute(
                "SELECT source_id, target_id, rel_type, weight, meta FROM relations"
            )
            for row in cursor:
                relations.append({
                    "source": row[0],
                    "target": row[1],
                    "type": row[2],
                    "weight": row[3],
                    "meta": json.loads(row[4]) if row[4] else {},
                })
        return relations

    def get_entity_network(
        self, entity_id: str, depth: int = 2
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Get entities and relations within N hops of a given entity.
        Uses a Recursive CTE in SQLite for maximum performance and O(1) memory overhead.
        """
        with self.store._tx() as conn:
            # 1. Recursive CTE to find all node IDs within `depth` hops (undirected traversal)
            query = """
            WITH RECURSIVE traverse(id, current_depth) AS (
                SELECT ?, 0
                UNION
                SELECT r.target_id, t.current_depth + 1
                FROM relations r JOIN traverse t ON r.source_id = t.id
                WHERE t.current_depth < ?
                UNION
                SELECT r.source_id, t.current_depth + 1
                FROM relations r JOIN traverse t ON r.target_id = t.id
                WHERE t.current_depth < ?
            )
            SELECT DISTINCT id FROM traverse
            """
            cursor = conn.execute(query, (entity_id, depth, depth))
            node_ids = [r[0] for r in cursor]

        if not node_ids:
            return [], []

        # 2. Fetch Entity Details
        placeholders = ",".join(["?"] * len(node_ids))
        entities = []
        with self.store._tx() as conn:
            e_cursor = conn.execute(
                f"SELECT id, entity_type, name, meta FROM entities WHERE id IN ({placeholders})",
                node_ids
            )
            for row in e_cursor:
                entities.append({
                    "id": row[0],
                    "type": row[1],
                    "name": row[2],
                    "meta": json.loads(row[3]) if row[3] else {},
                })

        # 3. Fetch all edges exclusively between the discovered nodes
        relations = []
        with self.store._tx() as conn:
            r_cursor = conn.execute(
                f"""SELECT source_id, target_id, rel_type, weight, meta
                    FROM relations
                    WHERE source_id IN ({placeholders}) AND target_id IN ({placeholders})""",
                node_ids + node_ids
            )
            for row in r_cursor:
                relations.append({
                    "source": row[0],
                    "target": row[1],
                    "type": row[2],
                    "weight": row[3],
                    "meta": json.loads(row[4]) if row[4] else {},
                })

        return entities, relations

    def to_json(
        self,
        entities: Optional[List[Dict]] = None,
        relations: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """Format data perfectly for the D3.js frontend."""
        if entities is None:
            entities = self.get_all_entities()
        if relations is None:
            relations = self.get_all_relations()

        # Deduplicate nodes to ensure valid graph state
        seen_nodes = set()
        nodes = []
        for e in entities:
            if e["id"] not in seen_nodes:
                seen_nodes.add(e["id"])
                nodes.append({
                    "id": e["id"],
                    "label": e["name"],
                    "type": e["type"],
                    "meta": e["meta"],
                })

        # Ensure edges only reference existing nodes
        edges = []
        for r in relations:
            if r["source"] in seen_nodes and r["target"] in seen_nodes:
                edges.append({
                    "source": r["source"],
                    "target": r["target"],
                    "type": r["type"],
                    "weight": r["weight"],
                    "meta": r["meta"],
                })

        return {
            "nodes": nodes,
            "edges": edges,
            "metadata": {
                "node_count": len(nodes),
                "edge_count": len(edges)
            }
        }

    def to_html(
        self,
        entities: Optional[List[Dict]] = None,
        relations: Optional[List[Dict]] = None,
        title: str = "Memory Intelligence Graph"
    ) -> str:
        """
        Generate standalone HTML with an interactive, professional D3.js visualization.
        Adheres strictly to UI rules: IBM Plex, CSS Grid, No Modals, Subdued styling.
        """
        graph_data = self.to_json(entities, relations)
        json_str = json.dumps(graph_data)

        # Using a raw string block with string replacement to avoid { } escaping hell in python f-strings.
        html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>__TITLE__</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        :root {
            --bg: #08080d;
            --text: #e2e2e8;
            --text-muted: rgba(255, 255, 255, 0.45);
            --text-faint: rgba(255, 255, 255, 0.25);
            --accent: #6366f1;
            --accent-light: #a5b4fc;
            --border: rgba(255, 255, 255, 0.05);
            --border-active: rgba(99, 102, 241, 0.2);
            --surface: rgba(255, 255, 255, 0.02);
            --surface-hover: rgba(99, 102, 241, 0.06);
        }

        body {
            margin: 0;
            padding: 0;
            background: var(--bg);
            color: var(--text);
            font-family: 'IBM Plex Sans', sans-serif;
            overflow: hidden;
            font-size: 14px;
        }

        /* Layout */
        .dashboard {
            display: grid;
            grid-template-columns: 1fr 400px;
            height: 100vh;
        }

        /* Graph Canvas */
        .graph-container {
            position: relative;
            outline: none;
            cursor: grab;
        }
        .graph-container:active { cursor: grabbing; }

        svg { width: 100%; height: 100%; }

        .node circle {
            stroke: var(--bg);
            stroke-width: 2px;
            cursor: pointer;
            transition: stroke 0.2s ease;
        }
        .node:hover circle { stroke: var(--accent); stroke-width: 3px; }

        .node text {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 11px;
            fill: var(--text);
            pointer-events: none;
            text-shadow: 0 1px 3px rgba(0,0,0,0.8);
        }

        .link {
            fill: none;
            stroke-opacity: 0.3;
            transition: stroke-opacity 0.2s ease, stroke-width 0.2s ease;
            cursor: pointer;
        }
        .link:hover {
            stroke-opacity: 0.8 !important;
            stroke-width: 3px !important;
        }

        /* Controls Overlays */
        .top-bar {
            position: absolute;
            top: 20px;
            left: 20px;
            pointer-events: none;
        }
        .label-micro {
            font-size: 9px;
            letter-spacing: 2px;
            text-transform: uppercase;
            color: var(--text-faint);
            margin-bottom: 4px;
        }
        .title {
            font-size: 22px;
            font-weight: 300;
            margin: 0;
        }

        .controls {
            position: absolute;
            bottom: 20px;
            left: 20px;
            display: flex;
            gap: 8px;
        }
        button {
            background: var(--surface);
            border: 1px solid var(--border);
            color: var(--text-muted);
            padding: 6px 12px;
            font-family: 'IBM Plex Mono', monospace;
            font-size: 11px;
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.15s ease;
        }
        button:hover {
            background: var(--surface-hover);
            color: var(--accent-light);
            border-color: var(--border-active);
        }

        /* Sidebar UI */
        .sidebar {
            border-left: 1px solid var(--border);
            background: var(--surface);
            display: grid;
            grid-template-rows: auto 1fr;
            overflow: hidden;
        }

        .sidebar-header {
            padding: 24px;
            border-bottom: 1px solid var(--border);
        }

        .sidebar-content {
            padding: 24px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 24px;
        }

        /* Typography & Data */
        .mono-text { font-family: 'IBM Plex Mono', monospace; }
        .text-dim { color: var(--text-muted); }
        h3 { font-size: 16px; font-weight: 500; margin: 0 0 12px 0; }

        .key-val {
            display: grid;
            grid-template-columns: 100px 1fr;
            font-size: 12px;
            margin-bottom: 8px;
            line-height: 1.5;
        }
        .key-val .key { color: var(--text-muted); }
        .key-val .val { font-family: 'IBM Plex Mono', monospace; color: var(--text); word-break: break-all; }

        /* Accordion List */
        .accordion-list {
            display: flex;
            flex-direction: column;
            border-top: 1px solid var(--border);
        }
        .accordion-item {
            border-bottom: 1px solid var(--border);
        }
        .accordion-header {
            padding: 12px 0;
            cursor: pointer;
            display: grid;
            grid-template-columns: 1fr auto;
            align-items: center;
            transition: all 0.15s ease;
        }
        .accordion-header:hover {
            background: var(--surface-hover);
            padding-left: 8px;
            padding-right: 8px;
            border-radius: 4px;
        }
        .accordion-title { font-size: 12px; font-weight: 500; }
        .accordion-body {
            display: none;
            padding: 12px;
            background: var(--bg);
            border-radius: 4px;
            margin-bottom: 12px;
            border: 1px solid var(--border);
        }
        .accordion-item.active .accordion-body { display: block; }

        /* Score Bar (Weights) */
        .score-wrap { display: flex; align-items: center; gap: 8px; }
        .score-bar { display: flex; gap: 2px; }
        .score-segment {
            width: 14px;
            height: 6px;
            background: rgba(255,255,255,0.05);
            border-radius: 1px;
        }
        .score-segment.active { background: var(--text-muted); }

        /* Utility */
        .empty-state {
            color: var(--text-faint);
            text-align: center;
            margin-top: 50px;
            font-family: 'IBM Plex Mono', monospace;
            font-size: 12px;
        }
    </style>
</head>
<body>
    <div class="dashboard">
        <!-- Canvas -->
        <div class="graph-container" id="graph-container">
            <div class="top-bar">
                <div class="label-micro">Graph Visualization</div>
                <h1 class="title">__TITLE__</h1>
            </div>
            <div class="controls">
                <button onclick="zoomIn()">ZOOM IN</button>
                <button onclick="zoomOut()">ZOOM OUT</button>
                <button onclick="resetZoom()">RESET</button>
            </div>
            <svg id="graph-svg"></svg>
        </div>

        <!-- Inspector -->
        <div class="sidebar">
            <div class="sidebar-header" id="sidebar-header">
                <div class="label-micro">Inspector</div>
                <h3 class="text-dim" style="font-weight:300;">Select a node or edge</h3>
            </div>
            <div class="sidebar-content" id="sidebar-content">
                <div class="empty-state">// awaiting selection...</div>
            </div>
        </div>
    </div>

    <script>
        const GRAPH_DATA = __GRAPH_DATA__;

        // --- Utils: Badges & Colors ---
        const colorPalette = ["#6366f1", "#10b981", "#f59e0b", "#ef4444", "#a5b4fc", "#ec4899", "#8b5cf6", "#14b8a6"];

        function getHashColor(str) {
            let hash = 0;
            for(let i=0; i<str.length; i++) hash = str.charCodeAt(i) + ((hash << 5) - hash);
            return colorPalette[Math.abs(hash) % colorPalette.length];
        }

        function createBadge(text, color) {
            const hex = color || getHashColor(text);
            return `<span style="padding: 2px 7px; border-radius: 4px; font-size: 9px; font-weight: 600; background: ${hex}18; color: ${hex}; border: 1px solid ${hex}30; display: inline-block; text-transform: uppercase; letter-spacing: 1px;">${text}</span>`;
        }

        function renderScoreBar(weight) {
            const val = parseFloat(weight) || 0;
            const activeCount = Math.max(1, Math.round(val * 5));
            let html = '<div class="score-wrap"><div class="score-bar">';
            for(let i=1; i<=5; i++) {
                html += `<div class="score-segment ${i <= activeCount ? 'active' : ''}"></div>`;
            }
            html += `</div><span class="mono-text text-dim" style="font-size:10px">${val.toFixed(2)}</span></div>`;
            return html;
        }

        function renderMeta(metaObj) {
            if (!metaObj || Object.keys(metaObj).length === 0) return '<div class="text-dim mono-text" style="font-size:11px">null</div>';
            let html = '';
            for (const [k, v] of Object.entries(metaObj)) {
                const valStr = typeof v === 'object' ? JSON.stringify(v) : v;
                html += `<div class="key-val"><div class="key">${k}</div><div class="val">${valStr}</div></div>`;
            }
            return html;
        }

        // --- Data Preprocessing ---
        // Calculate degree centrality for node sizing
        const degrees = {};
        GRAPH_DATA.edges.forEach(e => {
            degrees[e.source] = (degrees[e.source] || 0) + 1;
            degrees[e.target] = (degrees[e.target] || 0) + 1;
        });

        GRAPH_DATA.nodes.forEach(n => {
            n.degree = degrees[n.id] || 0;
            n.radius = Math.max(4, Math.min(24, 4 + Math.sqrt(n.degree) * 3));
            n.color = getHashColor(n.type);
        });

        // Calculate multi-edges (Curved links logic)
        const edgeCounts = {};
        GRAPH_DATA.edges.forEach(e => {
            // Strictly directed key to separate A->B and B->A curves if needed,
            // but for UI clarity we group undirected bounds to spread arcs.
            const s = e.source < e.target ? e.source : e.target;
            const t = e.source < e.target ? e.target : e.source;
            const key = `${s}|||${t}`;
            if (!edgeCounts[key]) edgeCounts[key] = 0;
            e.linknum = edgeCounts[key]++;
        });

        // --- D3 Graph Setup ---
        const container = document.getElementById("graph-container");
        const width = container.clientWidth;
        const height = container.clientHeight;

        const svg = d3.select("#graph-svg");
        const g = svg.append("g");

        // Zoom & Pan
        const zoomBehavior = d3.zoom()
            .scaleExtent([0.1, 8])
            .on("zoom", (e) => g.attr("transform", e.transform));
        svg.call(zoomBehavior);

        window.zoomIn = () => svg.transition().duration(300).call(zoomBehavior.scaleBy, 1.4);
        window.zoomOut = () => svg.transition().duration(300).call(zoomBehavior.scaleBy, 0.7);
        window.resetZoom = () => svg.transition().duration(300).call(zoomBehavior.transform, d3.zoomIdentity.translate(width/2, height/2).scale(0.8));

        // Define Arrow Markers
        svg.append("defs").selectAll("marker")
            .data(["arrow"])
            .enter().append("marker")
            .attr("id", String)
            .attr("viewBox", "0 -5 10 10")
            .attr("refX", 15) // Will be updated dynamically per node radius
            .attr("refY", 0)
            .attr("markerWidth", 5)
            .attr("markerHeight", 5)
            .attr("orient", "auto")
            .append("path")
            .attr("d", "M0,-5L10,0L0,5")
            .attr("fill", "var(--text-muted)");

        // Force Simulation
        const simulation = d3.forceSimulation(GRAPH_DATA.nodes)
            .force("link", d3.forceLink(GRAPH_DATA.edges).id(d => d.id).distance(120))
            .force("charge", d3.forceManyBody().strength(-300))
            .force("center", d3.forceCenter(0, 0))
            .force("collide", d3.forceCollide().radius(d => d.radius + 15).iterations(2));

        // Edges (Paths for curves)
        const link = g.append("g")
            .selectAll("path")
            .data(GRAPH_DATA.edges)
            .enter().append("path")
            .attr("class", "link")
            .attr("stroke", d => getHashColor(d.type))
            .attr("stroke-width", d => Math.max(1, d.weight * 3))
            .attr("marker-end", "url(#arrow)")
            .on("click", (event, d) => inspectEdge(d, event));

        // Nodes
        const node = g.append("g")
            .selectAll("g")
            .data(GRAPH_DATA.nodes)
            .enter().append("g")
            .attr("class", "node")
            .call(d3.drag()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended))
            .on("click", (event, d) => inspectNode(d, event));

        node.append("circle")
            .attr("r", d => d.radius)
            .attr("fill", d => d.color);

        node.append("text")
            .attr("dy", d => d.radius + 12)
            .attr("text-anchor", "middle")
            .text(d => d.label);

        // Simulation Tick Update
        simulation.on("tick", () => {
            link.attr("d", d => {
                const dx = d.target.x - d.source.x;
                const dy = d.target.y - d.source.y;
                const dr = Math.sqrt(dx * dx + dy * dy);

                // Adjust arrow position to edge of circle
                const targetRadius = d.target.radius + 2;
                const offsetX = (dx * targetRadius) / dr;
                const offsetY = (dy * targetRadius) / dr;
                const tx = d.target.x - offsetX;
                const ty = d.target.y - offsetY;

                if (d.linknum === 0) {
                    return `M${d.source.x},${d.source.y}L${tx},${ty}`;
                }

                // Multiple edges: Draw arcs
                const sweep = d.linknum % 2 === 0 ? 0 : 1;
                const scale = 1 + Math.floor((d.linknum - 1) / 2) * 0.4;
                const r = dr * scale;
                return `M${d.source.x},${d.source.y}A${r},${r} 0 0,${sweep} ${tx},${ty}`;
            });

            node.attr("transform", d => `translate(${d.x},${d.y})`);
        });

        // Initial Center
        setTimeout(() => resetZoom(), 100);

        function dragstarted(event, d) {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x; d.fy = d.y;
        }
        function dragged(event, d) {
            d.fx = event.x; d.fy = event.y;
        }
        function dragended(event, d) {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null; d.fy = null;
        }

        // --- UI Interaction / Sidebar ---
        const sHeader = document.getElementById("sidebar-header");
        const sContent = document.getElementById("sidebar-content");

        function toggleAccordion(el) {
            el.parentElement.classList.toggle('active');
        }
        window.toggleAccordion = toggleAccordion;

        function inspectNode(d, event) {
            event.stopPropagation();

            // Highlight
            node.style("opacity", o => (o.id === d.id || isConnected(o, d)) ? 1 : 0.2);
            link.style("stroke-opacity", o => (o.source.id === d.id || o.target.id === d.id) ? 0.8 : 0.05);

            sHeader.innerHTML = `
                <div class="label-micro">Entity Details</div>
                <h3 style="margin-bottom:8px">${d.label}</h3>
                ${createBadge(d.type, d.color)}
            `;

            // Find connected edges
            const edges = GRAPH_DATA.edges.filter(e => e.source.id === d.id || e.target.id === d.id);

            let connectionsHtml = '<div class="accordion-list">';
            edges.forEach(e => {
                const isOut = e.source.id === d.id;
                const other = isOut ? e.target : e.source;
                const dirLabel = isOut ? 'OUT' : 'IN ';
                const dirColor = isOut ? '#a5b4fc' : '#14b8a6';

                connectionsHtml += `
                    <div class="accordion-item">
                        <div class="accordion-header" onclick="toggleAccordion(this)">
                            <div class="accordion-title">
                                <span class="mono-text" style="color:${dirColor}; font-size:10px; margin-right:6px">${dirLabel}</span>
                                ${e.type}
                                <span class="text-dim" style="margin: 0 4px">→</span>
                                ${other.label}
                            </div>
                        </div>
                        <div class="accordion-body">
                            <div class="label-micro">Edge Weight</div>
                            ${renderScoreBar(e.weight)}
                            <div class="label-micro" style="margin-top:16px">Edge Metadata</div>
                            ${renderMeta(e.meta)}
                        </div>
                    </div>
                `;
            });
            connectionsHtml += '</div>';

            sContent.innerHTML = `
                <div>
                    <div class="label-micro">Node ID</div>
                    <div class="mono-text text-dim">${d.id}</div>
                </div>
                <div>
                    <div class="label-micro">Metadata</div>
                    <div style="background:var(--bg); border:1px solid var(--border); padding:12px; border-radius:4px;">
                        ${renderMeta(d.meta)}
                    </div>
                </div>
                <div>
                    <div class="label-micro">Connections (${edges.length})</div>
                    ${connectionsHtml}
                </div>
            `;
        }

        function inspectEdge(d, event) {
            event.stopPropagation();

            // Highlight
            link.style("stroke-opacity", o => o === d ? 1 : 0.05)
                .style("stroke-width", o => o === d ? Math.max(2, o.weight*4) : 1);
            node.style("opacity", o => (o.id === d.source.id || o.id === d.target.id) ? 1 : 0.2);

            const typeColor = getHashColor(d.type);

            sHeader.innerHTML = `
                <div class="label-micro">Relation Details</div>
                <h3 style="margin-bottom:8px">${d.source.label} <span class="text-dim">→</span> ${d.target.label}</h3>
                ${createBadge(d.type, typeColor)}
            `;

            sContent.innerHTML = `
                <div>
                    <div class="label-micro">Weight Score</div>
                    ${renderScoreBar(d.weight)}
                </div>
                <div>
                    <div class="label-micro">Source Node</div>
                    <div class="key-val"><div class="key">ID</div><div class="val text-dim">${d.source.id}</div></div>
                    <div class="key-val"><div class="key">Type</div><div class="val">${d.source.type}</div></div>
                </div>
                <div>
                    <div class="label-micro">Target Node</div>
                    <div class="key-val"><div class="key">ID</div><div class="val text-dim">${d.target.id}</div></div>
                    <div class="key-val"><div class="key">Type</div><div class="val">${d.target.type}</div></div>
                </div>
                <div>
                    <div class="label-micro">Edge Metadata</div>
                    <div style="background:var(--bg); border:1px solid var(--border); padding:12px; border-radius:4px;">
                        ${renderMeta(d.meta)}
                    </div>
                </div>
            `;
        }

        function isConnected(a, b) {
            return GRAPH_DATA.edges.some(e => (e.source.id === a.id && e.target.id === b.id) || (e.source.id === b.id && e.target.id === a.id));
        }

        // Reset click
        svg.on("click", () => {
            node.style("opacity", 1);
            link.style("stroke-opacity", 0.3).style("stroke-width", d => Math.max(1, d.weight * 3));
            sHeader.innerHTML = `
                <div class="label-micro">Inspector</div>
                <h3 class="text-dim" style="font-weight:300;">Select a node or edge</h3>
            `;
            sContent.innerHTML = '<div class="empty-state">// awaiting selection...</div>';
        });

    </script>
</body>
</html>"""

        # Replace template placeholders manually to avoid f-string syntax conflict with JS/CSS { } brackets
        html_out = html_template.replace("__TITLE__", title)
        html_out = html_out.replace("__GRAPH_DATA__", json_str)
        return html_out

    def save_html(self, filepath: str, **kwargs) -> None:
        """Save the generated dashboard to an HTML file."""
        html_content = self.to_html(**kwargs)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"✓ Memory Intelligence Dashboard saved to {filepath}")

    def subgraph(self, entity_id: str, depth: int = 2) -> "MemoryGraphVisualizer":
        """
        Create a subgraph visualizer for an N-hop neighborhood.
        Uses SQLite native traversal.
        """
        entities, relations = self.get_entity_network(entity_id, depth)

        class SubgraphVisualizer(MemoryGraphVisualizer):
            def __init__(self, parent, ent, rel):
                self.store = parent.store
                self._ent = ent
                self._rel = rel

            def get_all_entities(self): return self._ent
            def get_all_relations(self): return self._rel

        return SubgraphVisualizer(self, entities, relations)


# CLI interface
if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Visualize V2 HybridMemoryStore knowledge graphs")
    parser.add_argument("store_path", help="Path to HybridMemoryStore directory")
    parser.add_argument("--output", "-o", default="graph.html", help="Output HTML file path")
    parser.add_argument("--subgraph", help="Center on specific entity ID")
    parser.add_argument("--depth", type=int, default=2, help="N-hop traversal depth (default: 2)")

    args = parser.parse_args()

    try:
        from toolboxv2.mods.isaa.base.hybrid_memory import HybridMemoryStore
    except ImportError:
        print("Error: Could not import HybridMemoryStore. Ensure you are running within the toolboxv2 environment.")
        sys.exit(1)

    try:
        # Load the V2 store safely
        store = HybridMemoryStore(args.store_path, 768)
        visualizer = MemoryGraphVisualizer(store)

        if args.subgraph:
            print(f"Extracting subgraph for {args.subgraph} (depth={args.depth})...")
            visualizer = visualizer.subgraph(args.subgraph, args.depth)

        visualizer.save_html(args.output)
    except Exception as e:
        print(f"Fatal Error: {e}")
    finally:
        if 'store' in locals():
            store.close()
