from __future__ import annotations

from collections import OrderedDict

from .models import PaperRecord
from .ollama import OllamaEmbeddingClient
from .text import cosine_counts, cosine_vectors, normalize_doi, normalize_title, token_counts


def paper_text(paper: PaperRecord) -> str:
    return " ".join(part for part in [paper.title, paper.abstract] if part)


def canonical_key(paper: PaperRecord) -> str:
    doi = normalize_doi(paper.doi)
    if doi:
        return f"doi:{doi}"
    if paper.paper_id:
        return f"s2:{paper.paper_id}"
    if paper.openalex_id:
        return f"openalex:{paper.openalex_id}"
    title = normalize_title(paper.title)
    if title:
        return f"title:{title}"
    return f"unknown:{id(paper)}"


def display_id(paper: PaperRecord) -> str:
    return paper.paper_id or paper.openalex_id or canonical_key(paper)


def merge_papers(primary: PaperRecord, incoming: PaperRecord) -> PaperRecord:
    primary.paper_id = primary.paper_id or incoming.paper_id
    primary.title = primary.title or incoming.title
    primary.abstract = primary.abstract or incoming.abstract
    primary.doi = primary.doi or incoming.doi
    primary.openalex_id = primary.openalex_id or incoming.openalex_id
    primary.arxiv_id = primary.arxiv_id or incoming.arxiv_id
    primary.url = primary.url or incoming.url
    primary.pdf_url = primary.pdf_url or incoming.pdf_url
    primary.year = primary.year or incoming.year
    primary.venue = primary.venue or incoming.venue
    primary.citation_count = primary.citation_count or incoming.citation_count
    if not primary.authors and incoming.authors:
        primary.authors = incoming.authors
    primary.raw.update(incoming.raw)
    return primary


def dedupe_papers(papers: list[PaperRecord]) -> list[PaperRecord]:
    merged: OrderedDict[str, PaperRecord] = OrderedDict()
    alias_to_key: dict[str, str] = {}
    for paper in papers:
        aliases = candidate_aliases(paper)
        existing_key = next((alias_to_key[alias] for alias in aliases if alias in alias_to_key), None)
        key = existing_key or canonical_key(paper)
        if key in merged:
            merge_papers(merged[key], paper)
        else:
            merged[key] = paper
        for alias in candidate_aliases(merged[key]):
            alias_to_key[alias] = key
    return list(merged.values())


def candidate_aliases(paper: PaperRecord) -> set[str]:
    aliases = set()
    doi = normalize_doi(paper.doi)
    title = normalize_title(paper.title)
    if doi:
        aliases.add(f"doi:{doi}")
    if paper.paper_id:
        aliases.add(f"s2:{paper.paper_id}")
    if paper.openalex_id:
        aliases.add(f"openalex:{paper.openalex_id}")
    if title:
        aliases.add(f"title:{title}")
    return aliases


def rank_papers(
    papers: list[PaperRecord],
    query: str,
    *,
    method: str = "lexical",
    embedding_client: OllamaEmbeddingClient | None = None,
    lexical_weight: float = 0.25,
    embedding_weight: float = 0.75,
) -> list[PaperRecord]:
    query_counts = token_counts(query)
    query_embedding: list[float] | None = None
    if method in {"ollama", "hybrid"}:
        if not embedding_client:
            raise RuntimeError("Embedding ranker requested but no embedding client was provided")
        query_embedding = embedding_client.embed(query)

    for paper in papers:
        paper.lexical_score = cosine_counts(query_counts, token_counts(paper_text(paper)))
        paper.embedding_score = None
        if query_embedding is not None:
            paper.embedding_score = cosine_vectors(query_embedding, embedding_client.embed(paper_text(paper)))
        if method == "lexical":
            paper.ranking_score = paper.lexical_score
        elif method == "ollama":
            paper.ranking_score = paper.embedding_score or 0.0
        elif method == "hybrid":
            paper.ranking_score = (embedding_weight * (paper.embedding_score or 0.0)) + (
                lexical_weight * paper.lexical_score
            )
        else:
            raise ValueError(f"Unknown ranking method: {method}")
        paper.relevance_score = paper.ranking_score
    return sorted(
        papers,
        key=lambda paper: (paper.ranking_score, paper.citation_count or 0, paper.year or 0),
        reverse=True,
    )
