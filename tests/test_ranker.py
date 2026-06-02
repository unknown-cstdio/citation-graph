from citation_deep_research.models import PaperRecord
from citation_deep_research.ranker import dedupe_papers, rank_papers


def test_dedupe_merges_by_doi_and_preserves_metadata() -> None:
    first = PaperRecord(paper_id="s2-a", doi="https://doi.org/10.1234/ABC", title="Original")
    second = PaperRecord(paper_id="s2-b", doi="10.1234/abc", abstract="Later metadata")

    papers = dedupe_papers([first, second])

    assert len(papers) == 1
    assert papers[0].paper_id == "s2-a"
    assert papers[0].abstract == "Later metadata"


def test_rank_papers_scores_title_and_abstract_relevance() -> None:
    relevant = PaperRecord(title="Citation graphs", abstract="Semantic Scholar citation closure")
    irrelevant = PaperRecord(title="Marine biology", abstract="coral reefs")

    ranked = rank_papers([irrelevant, relevant], "citation graph scholar")

    assert ranked[0] is relevant
    assert ranked[0].relevance_score > ranked[1].relevance_score
    assert ranked[0].lexical_score == ranked[0].ranking_score


def test_rank_papers_can_use_hybrid_embedding_score() -> None:
    class FakeEmbeddingClient:
        def embed(self, text: str):
            if text == "query":
                return [1.0, 0.0]
            if "semantic match" in text:
                return [1.0, 0.0]
            return [0.0, 1.0]

    semantic = PaperRecord(title="semantic match")
    lexical = PaperRecord(title="query exact token")

    ranked = rank_papers(
        [lexical, semantic],
        "query",
        method="hybrid",
        embedding_client=FakeEmbeddingClient(),
        lexical_weight=0.0,
        embedding_weight=1.0,
    )

    assert ranked[0] is semantic
    assert ranked[0].embedding_score == 1.0
    assert ranked[0].ranking_score > ranked[1].ranking_score
