from __future__ import annotations

import json
import requests
import sys
from math import ceil
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

from .clients import HttpRequestError, OpenAlexClient, SemanticScholarClient, UnpaywallClient
from .institution import write_institution_links
from .models import CitationEdge, CrawlWarning, Manifest, PaperRecord
from .ollama import OllamaEmbeddingClient
from .ranker import dedupe_papers, display_id, rank_papers
from .resolver import PdfResolver
from .visualization import write_graph_html


class CitationClosurePipeline:
    def __init__(
        self,
        *,
        semantic_scholar: SemanticScholarClient,
        openalex: OpenAlexClient | None,
        unpaywall: UnpaywallClient | None,
    ) -> None:
        self.semantic_scholar = semantic_scholar
        self.openalex = openalex
        self.unpaywall = unpaywall

    def crawl(
        self,
        *,
        seed: str,
        query: str | None,
        out_dir: Path,
        depth: int = 1,
        max_candidates: int = 100,
        max_papers_to_read: int = 20,
        max_papers_to_expand: int | None = None,
        direction: str = "both",
        download_pdfs: bool = True,
        use_fallbacks: bool = True,
        semantic_scholar_requests_per_second: float = 1.0,
        semantic_scholar_max_retries: int = 8,
        semantic_scholar_429_cooldown: float = 30.0,
        ranker: str = "lexical",
        embedding_model: str = "nomic-embed-text",
        ollama_url: str = "http://localhost:11434",
        allow_ranker_fallback: bool = True,
        depth_caps: list[int] | None = None,
        progress: Callable[[str], None] | None = None,
    ) -> Manifest:
        out_dir.mkdir(parents=True, exist_ok=True)
        pdf_dir = out_dir / "pdfs"

        emit_progress(progress, f"Resolving seed: {seed}")
        seed_paper = self.semantic_scholar.resolve_seed(seed)
        if not seed_paper.paper_id:
            raise RuntimeError("Resolved seed has no Semantic Scholar paperId; cannot crawl citation graph")
        seed_paper.graph_depth = 0
        emit_progress(progress, f"Resolved seed paper: {short_title(seed_paper)} [{seed_paper.paper_id}]")

        papers: list[PaperRecord] = [seed_paper]
        edges: list[CitationEdge] = []
        warnings: list[CrawlWarning] = []
        frontier = [seed_paper]
        seen_frontier = {seed_paper.paper_id}
        ranking_query = query or " ".join(part for part in [seed_paper.title, seed_paper.abstract] if part)
        effective_ranker = ranker
        embedding_client = (
            OllamaEmbeddingClient(model=embedding_model, base_url=ollama_url) if ranker in {"ollama", "hybrid"} else None
        )

        for depth_index in range(max(0, depth)):
            next_frontier: list[PaperRecord] = []
            depth_cap = cap_for_depth(depth_caps, depth_index, max_candidates)
            remaining_budget = min(max_candidates + 1 - len(papers), depth_cap)
            if remaining_budget <= 0:
                emit_progress(progress, f"Stopping before depth {depth_index + 1}: candidate budget exhausted.")
                break
            emit_progress(
                progress,
                f"Depth {depth_index + 1}/{depth}: frontier={len(frontier)}, discovered={len(papers)}, "
                f"remaining_depth_budget={remaining_budget}",
            )
            if depth_index == 0:
                expansion_frontier = frontier
            else:
                emit_progress(progress, f"Ranking frontier for depth {depth_index + 1} expansion using {effective_ranker}.")
                expansion_frontier, effective_ranker, frontier_warning = rank_with_configured_ranker(
                    frontier,
                    ranking_query,
                    ranker=effective_ranker,
                    embedding_client=embedding_client,
                    original_ranker=ranker,
                    embedding_model=embedding_model,
                    ollama_url=ollama_url,
                    allow_fallback=allow_ranker_fallback,
                    warning_context="frontier_expansion",
                )
                if frontier_warning:
                    warnings.append(frontier_warning)
                    emit_progress(progress, f"Warning: {frontier_warning.code}: {frontier_warning.message}")
                expansion_limit = max_papers_to_expand or max_papers_to_read
                expansion_frontier = expansion_frontier[:expansion_limit]
                for paper in expansion_frontier:
                    if paper.paper_id:
                        paper.expanded_for_crawl = True
                emit_progress(progress, f"Expanding top {len(expansion_frontier)} ranked frontier papers.")
            parent_budget = max(1, ceil(remaining_budget / max(1, len(expansion_frontier))))
            for parent_index, current in enumerate(expansion_frontier, start=1):
                if not current.paper_id or len(papers) >= max_candidates + 1:
                    continue
                remaining_budget = min(max_candidates + 1 - len(papers), depth_cap - len(next_frontier))
                if remaining_budget <= 0:
                    break
                neighbor_limit = min(parent_budget, remaining_budget)
                emit_progress(
                    progress,
                    f"Expanding {parent_index}/{len(expansion_frontier)} at depth {depth_index + 1}: "
                    f"{short_title(current)} ({direction}, limit={neighbor_limit})",
                )
                before_count = len(papers)
                neighbors, relation_warnings = self._neighbors(current.paper_id, direction, neighbor_limit)
                warnings.extend(relation_warnings)
                for warning in relation_warnings:
                    emit_progress(progress, f"Warning: {warning.code}: {warning.message}")
                for relation, neighbor in neighbors:
                    if not neighbor.paper_id:
                        continue
                    edge = citation_edge(current.paper_id, neighbor.paper_id, relation)
                    edges.append(edge)
                    if neighbor.paper_id in seen_frontier:
                        continue
                    neighbor.graph_depth = depth_index + 1
                    neighbor.parent_paper_id = current.paper_id
                    neighbor.parent_relation = relation
                    papers.append(neighbor)
                    next_frontier.append(neighbor)
                    seen_frontier.add(neighbor.paper_id)
                    if len(papers) >= max_candidates + 1:
                        break
                emit_progress(
                    progress,
                    f"Finished expansion: added {len(papers) - before_count} new papers; discovered={len(papers)}.",
                )
            frontier = next_frontier
            if not frontier:
                emit_progress(progress, f"Stopping after depth {depth_index + 1}: no new frontier papers.")
                break

        deduped = dedupe_papers(papers)
        emit_progress(progress, f"Ranking {len(deduped)} deduplicated papers using {effective_ranker}.")
        ranked, effective_ranker, final_ranker_warning = rank_with_configured_ranker(
            deduped,
            ranking_query,
            ranker=effective_ranker,
            embedding_client=embedding_client,
            original_ranker=ranker,
            embedding_model=embedding_model,
            ollama_url=ollama_url,
            allow_fallback=allow_ranker_fallback,
            warning_context="final_ranking",
        )
        if final_ranker_warning:
            warnings.append(final_ranker_warning)
            emit_progress(progress, f"Warning: {final_ranker_warning.code}: {final_ranker_warning.message}")
        selected = ranked[:max_papers_to_read]
        emit_progress(progress, f"Selected top {len(selected)} papers for PDF resolution.")

        resolver = PdfResolver(
            openalex=self.openalex,
            unpaywall=self.unpaywall,
            download_pdfs=download_pdfs,
            use_fallbacks=use_fallbacks,
        )
        for paper_index, paper in enumerate(selected, start=1):
            paper.selected_for_reading = True
            emit_progress(progress, f"Resolving PDF {paper_index}/{len(selected)}: {short_title(paper)}")
            resolver.resolve(paper, pdf_dir)

        selected_ids = [display_id(paper) for paper in selected]
        manifest = Manifest(
            seed=seed_paper,
            papers=ranked,
            edges=dedupe_edges(edges),
            warnings=warnings,
            config={
                "seed": seed,
                "query": query,
                "depth": depth,
                "max_candidates": max_candidates,
                "max_papers_to_read": max_papers_to_read,
                "max_papers_to_expand": max_papers_to_expand or max_papers_to_read,
                "direction": direction,
                "download_pdfs": download_pdfs,
                "use_fallbacks": use_fallbacks,
                "semantic_scholar_requests_per_second": semantic_scholar_requests_per_second,
                "semantic_scholar_max_retries": semantic_scholar_max_retries,
                "semantic_scholar_429_cooldown": semantic_scholar_429_cooldown,
                "ranker": ranker,
                "effective_ranker": effective_ranker,
                "embedding_model": embedding_model if ranker in {"ollama", "hybrid"} else None,
                "ollama_url": ollama_url if ranker in {"ollama", "hybrid"} else None,
                "depth_caps": depth_caps,
                "expansion_policy": "selected_frontier_by_ranker",
            },
            selected_paper_ids=selected_ids,
            generated_at=datetime.now(UTC).isoformat(),
        )
        emit_progress(progress, "Writing manifest, graph, visualization, and institution-link artifacts.")
        write_outputs(manifest, out_dir)
        emit_progress(progress, f"Done: papers={len(ranked)}, edges={len(manifest.edges)}, warnings={len(warnings)}.")
        return manifest

    def _neighbors(
        self,
        paper_id: str,
        direction: str,
        max_candidates: int,
    ) -> tuple[list[tuple[str, PaperRecord]], list[CrawlWarning]]:
        neighbors: list[tuple[str, PaperRecord]] = []
        warnings: list[CrawlWarning] = []
        per_direction_limit = max(1, max_candidates) if direction == "both" else max_candidates
        references: list[PaperRecord] = []
        citations: list[PaperRecord] = []
        if direction in {"both", "references"}:
            try:
                references, reference_warnings = self.semantic_scholar.references(paper_id, limit=per_direction_limit)
                warnings.extend(reference_warnings)
            except RuntimeError as exc:
                warnings.append(relation_request_failed_warning(paper_id, "references", exc))
        if direction in {"both", "citations"}:
            try:
                citations, citation_warnings = self.semantic_scholar.citations(paper_id, limit=per_direction_limit)
                warnings.extend(citation_warnings)
            except RuntimeError as exc:
                warnings.append(relation_request_failed_warning(paper_id, "citations", exc))
        if direction == "both":
            neighbors.extend(interleave_relation_papers(references, citations, max_candidates))
        elif direction == "references":
            neighbors.extend(("reference", paper) for paper in references[:max_candidates])
        else:
            neighbors.extend(("citation", paper) for paper in citations[:max_candidates])
        return neighbors, warnings


def citation_edge(source_paper_id: str, neighbor_paper_id: str, relation: str) -> CitationEdge:
    if relation == "reference":
        return CitationEdge(source=source_paper_id, target=neighbor_paper_id, relation=relation)
    return CitationEdge(source=neighbor_paper_id, target=source_paper_id, relation=relation)


def relation_request_failed_warning(paper_id: str, relation: str, exc: RuntimeError) -> CrawlWarning:
    status_code = exc.status_code if isinstance(exc, HttpRequestError) else None
    code = f"{relation}_request_failed"
    if status_code == 429:
        code = f"{relation}_rate_limited"
    return CrawlWarning(
        code=code,
        message=f"Semantic Scholar {relation} request failed; skipped this expansion and continued crawl.",
        paper_id=paper_id,
        relation=relation,
        details={
            "status_code": status_code,
            "url": exc.url if isinstance(exc, HttpRequestError) else None,
            "error": str(exc),
        },
    )


def emit_progress(progress: Callable[[str], None] | None, message: str) -> None:
    if progress:
        progress(message)


def stderr_progress(message: str) -> None:
    print(f"[citation-closure] {message}", file=sys.stderr, flush=True)


def short_title(paper: PaperRecord, max_length: int = 90) -> str:
    title = paper.title or paper.paper_id or "unknown paper"
    if len(title) <= max_length:
        return title
    return f"{title[: max_length - 1]}..."


def rank_with_configured_ranker(
    papers: list[PaperRecord],
    ranking_query: str,
    *,
    ranker: str,
    embedding_client: OllamaEmbeddingClient | None,
    original_ranker: str,
    embedding_model: str,
    ollama_url: str,
    allow_fallback: bool,
    warning_context: str,
) -> tuple[list[PaperRecord], str, CrawlWarning | None]:
    try:
        return rank_papers(papers, ranking_query, method=ranker, embedding_client=embedding_client), ranker, None
    except (RuntimeError, requests.RequestException) as exc:
        if ranker == "lexical" or not allow_fallback:
            raise
        warning = CrawlWarning(
            code="ranker_fell_back_to_lexical",
            message="Embedding ranker failed; used lexical ranking instead.",
            details={
                "requested_ranker": original_ranker,
                "active_ranker": ranker,
                "embedding_model": embedding_model,
                "ollama_url": ollama_url,
                "context": warning_context,
                "error": str(exc),
            },
        )
        return rank_papers(papers, ranking_query, method="lexical"), "lexical", warning


def interleave_relation_papers(
    references: list[PaperRecord],
    citations: list[PaperRecord],
    limit: int,
) -> list[tuple[str, PaperRecord]]:
    neighbors: list[tuple[str, PaperRecord]] = []
    for index in range(max(len(references), len(citations))):
        if index < len(references):
            neighbors.append(("reference", references[index]))
            if len(neighbors) >= limit:
                break
        if index < len(citations):
            neighbors.append(("citation", citations[index]))
            if len(neighbors) >= limit:
                break
    return neighbors


def cap_for_depth(depth_caps: list[int] | None, depth_index: int, default_cap: int) -> int:
    if not depth_caps or depth_index >= len(depth_caps):
        return default_cap
    return max(0, depth_caps[depth_index])


def dedupe_edges(edges: list[CitationEdge]) -> list[CitationEdge]:
    seen: set[tuple[str, str, str]] = set()
    deduped: list[CitationEdge] = []
    for edge in edges:
        key = (edge.source, edge.target, edge.relation)
        if key not in seen:
            seen.add(key)
            deduped.append(edge)
    return deduped


def write_outputs(manifest: Manifest, out_dir: Path) -> None:
    manifest_data = manifest.to_dict()
    (out_dir / "manifest.json").write_text(json.dumps(manifest_data, indent=2, sort_keys=True), encoding="utf-8")
    graph = {
        "nodes": [
            {
                "id": display_id(paper),
                "paper_id": paper.paper_id,
                "title": paper.title,
                "doi": paper.doi,
                "selected_for_reading": paper.selected_for_reading,
                "abstract_only": paper.abstract_only,
                "pdf_path": paper.pdf_path,
                "pdf_url": paper.pdf_url,
                "pdf_candidates": paper.pdf_candidates,
                "resolution_source": paper.resolution_source,
                "year": paper.year,
                "venue": paper.venue,
                "citation_count": paper.citation_count,
                "graph_depth": paper.graph_depth,
                "parent_paper_id": paper.parent_paper_id,
                "parent_relation": paper.parent_relation,
                "expanded_for_crawl": paper.expanded_for_crawl,
                "relevance_score": paper.relevance_score,
                "lexical_score": paper.lexical_score,
                "embedding_score": paper.embedding_score,
                "ranking_score": paper.ranking_score,
            }
            for paper in manifest.papers
        ],
        "edges": [edge.to_dict() for edge in manifest.edges],
        "warnings": [warning.to_dict() for warning in manifest.warnings],
    }
    (out_dir / "graph.json").write_text(json.dumps(graph, indent=2, sort_keys=True), encoding="utf-8")
    write_graph_html(manifest, out_dir)
    write_institution_links(manifest, out_dir)
