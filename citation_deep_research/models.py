from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class PaperRecord:
    paper_id: str | None = None
    title: str | None = None
    abstract: str | None = None
    doi: str | None = None
    openalex_id: str | None = None
    arxiv_id: str | None = None
    url: str | None = None
    pdf_url: str | None = None
    year: int | None = None
    venue: str | None = None
    authors: list[str] = field(default_factory=list)
    citation_count: int | None = None
    source: str = "semantic_scholar"
    relevance_score: float = 0.0
    lexical_score: float = 0.0
    embedding_score: float | None = None
    ranking_score: float = 0.0
    graph_depth: int | None = None
    parent_paper_id: str | None = None
    parent_relation: str | None = None
    expanded_for_crawl: bool = False
    selected_for_reading: bool = False
    abstract_only: bool = True
    pdf_path: str | None = None
    resolution_source: str | None = None
    pdf_candidates: list[dict[str, Any]] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CitationEdge:
    source: str
    target: str
    relation: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(slots=True)
class CrawlWarning:
    code: str
    message: str
    paper_id: str | None = None
    relation: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Manifest:
    seed: PaperRecord
    papers: list[PaperRecord]
    edges: list[CitationEdge]
    warnings: list[CrawlWarning]
    config: dict[str, Any]
    selected_paper_ids: list[str]
    generated_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "config": self.config,
            "seed": self.seed.to_dict(),
            "selected_paper_ids": self.selected_paper_ids,
            "papers": [paper.to_dict() for paper in self.papers],
            "edges": [edge.to_dict() for edge in self.edges],
            "warnings": [warning.to_dict() for warning in self.warnings],
            "counts": {
                "papers": len(self.papers),
                "edges": len(self.edges),
                "warnings": len(self.warnings),
                "selected_for_reading": len(self.selected_paper_ids),
                "pdfs_downloaded": sum(1 for paper in self.papers if paper.pdf_path),
                "abstract_only": sum(1 for paper in self.papers if paper.abstract_only),
            },
        }
