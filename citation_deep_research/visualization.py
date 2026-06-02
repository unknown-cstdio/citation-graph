from __future__ import annotations

import json
from pathlib import Path

from .models import Manifest, PaperRecord
from .ranker import display_id


def write_graph_html(manifest: Manifest, out_dir: Path) -> None:
    view = build_visualization_data(manifest)
    html_text = HTML_TEMPLATE.replace("__GRAPH_DATA__", json.dumps(view, sort_keys=True))
    (out_dir / "graph.html").write_text(html_text, encoding="utf-8")


def build_visualization_data(manifest: Manifest) -> dict:
    rank_by_id = {display_id(paper): index + 1 for index, paper in enumerate(manifest.papers)}
    seed_id = display_id(manifest.seed)
    nodes = []
    for paper in manifest.papers:
        node_id = display_id(paper)
        nodes.append(
            {
                "id": node_id,
                "paper_id": paper.paper_id,
                "rank": rank_by_id[node_id],
                "title": paper.title or node_id,
                "year": paper.year,
                "venue": paper.venue,
                "doi": paper.doi,
                "citation_count": paper.citation_count,
                "selected": paper.selected_for_reading,
                "is_seed": node_id == seed_id,
                "graph_depth": paper.graph_depth,
                "parent_paper_id": paper.parent_paper_id,
                "parent_relation": paper.parent_relation,
                "expanded_for_crawl": paper.expanded_for_crawl,
                "status": paper_status(paper),
                "pdf_path": paper.pdf_path,
                "pdf_url": paper.pdf_url,
                "pdf_candidates": paper.pdf_candidates,
                "resolution_source": paper.resolution_source,
                "ranking_score": paper.ranking_score,
                "embedding_score": paper.embedding_score,
                "lexical_score": paper.lexical_score,
                "abstract_only": paper.abstract_only,
            }
        )
    return {
        "summary": {
            **manifest.to_dict()["counts"],
            "generated_at": manifest.generated_at,
            "ranker": manifest.config.get("effective_ranker") or manifest.config.get("ranker"),
            "embedding_model": manifest.config.get("embedding_model"),
            "query": manifest.config.get("query"),
            "seed_title": manifest.seed.title,
        },
        "nodes": nodes,
        "edges": [edge.to_dict() for edge in manifest.edges],
        "warnings": [warning.to_dict() for warning in manifest.warnings],
    }


def paper_status(paper: PaperRecord) -> str:
    if paper.pdf_path:
        return "downloaded"
    if paper.selected_for_reading and paper.pdf_url:
        return "pdf-url-only"
    if paper.selected_for_reading:
        return "abstract-only"
    return "not-selected"


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Citation Graph Run</title>
  <style>
    :root {
      --ink: #1f2933;
      --muted: #687887;
      --panel: #fffdf7;
      --line: #d8d0bf;
      --downloaded: #1f9d55;
      --pdf-url-only: #d97706;
      --abstract-only: #c2410c;
      --not-selected: #8a99a8;
      --seed: #155e75;
      --bg: #f4efe3;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background:
        radial-gradient(circle at top left, rgba(21, 94, 117, 0.16), transparent 34rem),
        linear-gradient(135deg, #f7f2e8 0%, #ece3d0 100%);
      color: var(--ink);
      font-family: ui-serif, Georgia, Cambria, "Times New Roman", serif;
    }
    header {
      padding: 2rem clamp(1rem, 4vw, 3rem) 1rem;
    }
    h1 { margin: 0 0 .5rem; font-size: clamp(2rem, 4vw, 3.4rem); line-height: 1; }
    .subtitle { color: var(--muted); max-width: 72rem; font-size: 1.05rem; }
    main {
      display: grid;
      grid-template-columns: minmax(18rem, 26rem) 1fr;
      gap: 1rem;
      padding: 1rem clamp(1rem, 4vw, 3rem) 2rem;
    }
    .panel {
      background: rgba(255, 253, 247, 0.86);
      border: 1px solid var(--line);
      border-radius: 18px;
      box-shadow: 0 22px 80px rgba(48, 39, 22, 0.12);
    }
    aside { padding: 1rem; }
    .metrics { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: .65rem; }
    .metric { padding: .8rem; border-radius: 14px; background: #fbf7ec; border: 1px solid #e6dcc8; }
    .metric strong { display: block; font-size: 1.4rem; }
    .metric span { color: var(--muted); font-size: .85rem; }
    .legend { display: grid; gap: .45rem; margin: 1rem 0; }
    .legend-item { display: flex; align-items: center; gap: .5rem; color: var(--muted); }
    .dot { width: .8rem; height: .8rem; border-radius: 50%; display: inline-block; }
    #graph-card { min-height: 72vh; position: relative; overflow: hidden; }
    svg { display: block; width: 100%; height: 72vh; }
    .edge { stroke: #8b7962; stroke-opacity: .42; stroke-width: 1.6; marker-end: url(#arrow); }
    .edge.reference { stroke: #155e75; }
    .edge.citation { stroke: #92400e; }
    .node circle { stroke: rgba(255,255,255,.9); stroke-width: 2.5; filter: drop-shadow(0 4px 5px rgba(0,0,0,.18)); }
    .node.seed circle { stroke: #0f172a; stroke-width: 4; }
    .node text { font-family: ui-sans-serif, system-ui, sans-serif; font-size: 11px; paint-order: stroke; stroke: #fffdf7; stroke-width: 4px; stroke-linecap: round; stroke-linejoin: round; }
    .table-wrap { max-height: 28rem; overflow: auto; border-radius: 14px; border: 1px solid var(--line); }
    table { width: 100%; border-collapse: collapse; font-family: ui-sans-serif, system-ui, sans-serif; font-size: .84rem; }
    th, td { padding: .55rem .6rem; border-bottom: 1px solid #e7dcc8; text-align: left; vertical-align: top; }
    th { position: sticky; top: 0; background: #f6ecd9; z-index: 1; }
    .pill { display: inline-block; padding: .15rem .45rem; border-radius: 999px; color: white; font-size: .75rem; white-space: nowrap; }
    .downloaded { background: var(--downloaded); }
    .pdf-url-only { background: var(--pdf-url-only); }
    .abstract-only { background: var(--abstract-only); }
    .not-selected { background: var(--not-selected); }
    .seed { background: var(--seed); }
    .warnings { margin-top: 1rem; color: #7c2d12; font-family: ui-sans-serif, system-ui, sans-serif; font-size: .9rem; }
    @media (max-width: 900px) {
      main { grid-template-columns: 1fr; }
      svg { height: 64vh; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Citation Graph Run</h1>
    <div class="subtitle" id="subtitle"></div>
  </header>
  <main>
    <aside class="panel">
      <h2>Run Summary</h2>
      <div class="metrics" id="metrics"></div>
      <h3>Status Legend</h3>
      <div class="legend">
        <div class="legend-item"><span class="dot" style="background: var(--seed)"></span> Seed paper</div>
        <div class="legend-item"><span class="dot" style="background: var(--downloaded)"></span> PDF downloaded</div>
        <div class="legend-item"><span class="dot" style="background: var(--pdf-url-only)"></span> PDF URL found, not downloaded</div>
        <div class="legend-item"><span class="dot" style="background: var(--abstract-only)"></span> Selected, abstract-only</div>
        <div class="legend-item"><span class="dot" style="background: var(--not-selected)"></span> Not selected</div>
      </div>
      <div class="warnings" id="warnings"></div>
    </aside>
    <section class="panel" id="graph-card">
      <svg id="graph" role="img" aria-label="Citation graph visualization"></svg>
    </section>
  </main>
  <main style="padding-top:0">
    <section class="panel" style="grid-column: 1 / -1; padding:1rem">
      <h2>Ranked Papers</h2>
      <div class="table-wrap">
        <table>
          <thead>
            <tr><th>Rank</th><th>Status</th><th>Title</th><th>Score</th><th>Embedding</th><th>Lexical</th><th>PDF</th><th>Attempts</th></tr>
          </thead>
          <tbody id="paper-rows"></tbody>
        </table>
      </div>
    </section>
  </main>
  <script>
    const data = __GRAPH_DATA__;
    const esc = (value) => String(value ?? "").replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
    const fmt = (value) => value == null ? "" : Number(value).toFixed(3);
    const truncate = (value, max = 34) => {
      const text = String(value ?? "");
      return text.length > max ? `${text.slice(0, max - 1)}…` : text;
    };
    const statusColor = (node) => {
      if (node.is_seed) return "var(--seed)";
      return `var(--${node.status})`;
    };

    document.getElementById("subtitle").innerHTML =
      `<strong>${esc(data.summary.seed_title)}</strong><br>${esc(data.summary.query || "")}`;
    document.getElementById("metrics").innerHTML = [
      ["Papers", data.summary.papers],
      ["Edges", data.summary.edges],
      ["Selected", data.summary.selected_for_reading],
      ["PDFs", data.summary.pdfs_downloaded],
      ["Warnings", data.summary.warnings],
      ["Ranker", data.summary.ranker || "unknown"],
    ].map(([label, value]) => `<div class="metric"><strong>${esc(value)}</strong><span>${esc(label)}</span></div>`).join("");
    document.getElementById("warnings").innerHTML = data.warnings.length
      ? `<strong>Warnings</strong><br>${data.warnings.map(w => esc(`${w.code}: ${w.message}`)).join("<br>")}`
      : "No crawl warnings recorded.";
    document.getElementById("paper-rows").innerHTML = data.nodes
      .sort((a, b) => a.rank - b.rank)
      .map(node => `
        <tr>
          <td>${node.rank}</td>
          <td><span class="pill ${node.is_seed ? "seed" : node.status}">${node.is_seed ? "seed" : node.status}</span></td>
          <td>${esc(node.title)}${node.year ? ` (${node.year})` : ""}</td>
          <td>${fmt(node.ranking_score)}</td>
          <td>${fmt(node.embedding_score)}</td>
          <td>${fmt(node.lexical_score)}</td>
          <td>${node.pdf_path ? esc(node.resolution_source || "downloaded") : node.pdf_url ? "url only" : ""}</td>
          <td>${(node.pdf_candidates || []).length}</td>
        </tr>`).join("");

    const svg = document.getElementById("graph");
    const width = 1200;
    const height = 760;
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    svg.innerHTML = `
      <defs>
        <marker id="arrow" markerWidth="9" markerHeight="9" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
          <path d="M0,0 L0,6 L8,3 z" fill="#7b6a56"></path>
        </marker>
      </defs>`;

    const nodes = data.nodes.map((node, index) => ({...node, index}));
    const byId = new Map(nodes.map(node => [node.id, node]));
    const seed = nodes.find(node => node.is_seed) || nodes[0];
    const visibleNodes = nodes.filter(node => node.is_seed || node.selected || node.expanded_for_crawl);
    const visibleById = new Map(visibleNodes.map(node => [node.id, node]));
    const edges = data.edges.filter(edge => visibleById.has(edge.source) && visibleById.has(edge.target));
    const referenceIds = new Set();
    const citationIds = new Set();
    if (seed) {
      for (const edge of data.edges) {
        if (edge.relation === "reference" && edge.source === seed.id) referenceIds.add(edge.target);
        if (edge.relation === "citation" && edge.target === seed.id) citationIds.add(edge.source);
      }
    }
    const sideFor = (node) => {
      if (referenceIds.has(node.id)) return "left";
      if (citationIds.has(node.id)) return "right";
      const parent = byId.get(node.parent_paper_id);
      if (parent) return sideFor(parent);
      return "right";
    };
    const groups = {
      references: visibleNodes.filter(node => !node.is_seed && sideFor(node) === "left" && (node.graph_depth || 1) <= 1).sort((a, b) => a.rank - b.rank),
      citations: visibleNodes.filter(node => !node.is_seed && sideFor(node) === "right" && (node.graph_depth || 1) <= 1).sort((a, b) => a.rank - b.rank),
      referenceDepth2: visibleNodes.filter(node => !node.is_seed && sideFor(node) === "left" && (node.graph_depth || 1) > 1).sort((a, b) => a.rank - b.rank),
      citationDepth2: visibleNodes.filter(node => !node.is_seed && sideFor(node) === "right" && (node.graph_depth || 1) > 1).sort((a, b) => a.rank - b.rank),
    };
    if (seed) { seed.x = width / 2; seed.y = height / 2; }

    function placeColumn(items, x, top, bottom) {
      const span = Math.max(1, bottom - top);
      const gap = span / Math.max(1, items.length);
      items.forEach((node, index) => {
        node.x = x;
        node.y = top + gap * (index + 0.5);
      });
    }
    placeColumn(groups.referenceDepth2, 150, 100, height - 100);
    placeColumn(groups.references, 370, 90, height - 90);
    placeColumn(groups.citations, width - 370, 90, height - 90);
    placeColumn(groups.citationDepth2, width - 150, 100, height - 100);

    const label = (text, side) => {
      const anchor = side === "left" ? "end" : "start";
      const offset = side === "left" ? -18 : 18;
      return {anchor, offset, text: truncate(text, 34)};
    };

    const bandLabel = (text, x) => {
      const t = document.createElementNS("http://www.w3.org/2000/svg", "text");
      t.setAttribute("x", x);
      t.setAttribute("y", 38);
      t.setAttribute("text-anchor", "middle");
      t.setAttribute("fill", "#4b5563");
      t.setAttribute("font-family", "ui-sans-serif, system-ui, sans-serif");
      t.setAttribute("font-size", "14");
      t.setAttribute("font-weight", "700");
      t.textContent = text;
      svg.appendChild(t);
    };
    bandLabel("Expanded references", 150);
    bandLabel("Cited by seed", 370);
    bandLabel("Seed", width / 2);
    bandLabel("Papers citing seed", width - 370);
    bandLabel("Expanded citations", width - 150);

    for (const edge of edges) {
      const a = visibleById.get(edge.source);
      const b = visibleById.get(edge.target);
      const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
      line.setAttribute("class", `edge ${edge.relation}`);
      line.setAttribute("x1", a.x);
      line.setAttribute("y1", a.y);
      line.setAttribute("x2", b.x);
      line.setAttribute("y2", b.y);
      line.innerHTML = `<title>${esc(edge.relation)}: ${esc(a.title)} -> ${esc(b.title)}</title>`;
      svg.appendChild(line);
    }
    for (const node of visibleNodes) {
      const group = document.createElementNS("http://www.w3.org/2000/svg", "g");
      group.setAttribute("class", `node ${node.is_seed ? "seed" : ""}`);
      const radius = node.is_seed ? 18 : node.selected ? 14 : 9;
      const side = sideFor(node);
      const labelInfo = label(`#${node.rank} ${node.title}`, side);
      group.innerHTML = `
        <circle cx="${node.x}" cy="${node.y}" r="${radius}" fill="${statusColor(node)}"></circle>
        <text x="${node.x + labelInfo.offset}" y="${node.y + 4}" text-anchor="${labelInfo.anchor}">${esc(labelInfo.text)}</text>
        <title>${esc(node.title)}
rank: ${node.rank}
status: ${node.is_seed ? "seed" : node.status}
ranking: ${fmt(node.ranking_score)}
embedding: ${fmt(node.embedding_score)}
lexical: ${fmt(node.lexical_score)}
pdf: ${esc(node.pdf_path || node.pdf_url || "")}
pdf attempts: ${esc((node.pdf_candidates || []).map(c => `${c.source}:${c.status}${c.reason ? `(${c.reason})` : ""}`).join(", "))}</title>`;
      svg.appendChild(group);
    }
  </script>
</body>
</html>
"""
