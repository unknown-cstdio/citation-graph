import requests

from citation_deep_research.clients import HttpRequestError
from citation_deep_research.models import PaperRecord
from citation_deep_research.pipeline import CitationClosurePipeline


class FakeSemanticScholar:
    def resolve_seed(self, seed):
        return PaperRecord(paper_id="seed", title="seed paper", abstract="networking")

    def references(self, paper_id, *, limit):
        return [PaperRecord(paper_id="ref", title="reference paper")], []

    def citations(self, paper_id, *, limit):
        return [PaperRecord(paper_id="cite", title="citation paper")], []


class FanoutSemanticScholar:
    def resolve_seed(self, seed):
        return PaperRecord(paper_id="seed", title="seed paper", abstract="networking")

    def references(self, paper_id, *, limit):
        if paper_id == "seed":
            return [PaperRecord(paper_id="ref-a", title="reference networking paper")], []
        return [
            PaperRecord(paper_id=f"{paper_id}-ref-{index}", title=f"{paper_id} reference child {index}")
            for index in range(limit)
        ], []

    def citations(self, paper_id, *, limit):
        if paper_id == "seed":
            return [PaperRecord(paper_id="cite-a", title="citation networking paper")], []
        return [
            PaperRecord(paper_id=f"{paper_id}-cite-{index}", title=f"{paper_id} citation child {index}")
            for index in range(limit)
        ], []


class PartiallyFailingSemanticScholar:
    def resolve_seed(self, seed):
        return PaperRecord(paper_id="seed", title="seed paper", abstract="networking")

    def references(self, paper_id, *, limit):
        return [PaperRecord(paper_id="ref", title="reference paper")], []

    def citations(self, paper_id, *, limit):
        raise HttpRequestError("429 Client Error; response=Too Many Requests", url="https://example.test/citations", status_code=429)


def test_pipeline_falls_back_to_lexical_when_ollama_ranker_fails(monkeypatch, tmp_path) -> None:
    class FailingEmbeddingClient:
        def __init__(self, *args, **kwargs):
            pass

        def embed(self, text):
            raise requests.ConnectionError("ollama unavailable")

    monkeypatch.setattr("citation_deep_research.pipeline.OllamaEmbeddingClient", FailingEmbeddingClient)

    pipeline = CitationClosurePipeline(semantic_scholar=FakeSemanticScholar(), openalex=None, unpaywall=None)
    manifest = pipeline.crawl(
        seed="seed",
        query="networking",
        out_dir=tmp_path,
        max_candidates=2,
        max_papers_to_read=2,
        ranker="hybrid",
        download_pdfs=False,
        use_fallbacks=False,
    )

    assert manifest.config["effective_ranker"] == "lexical"
    assert manifest.warnings[0].code == "ranker_fell_back_to_lexical"


def test_relation_request_failure_records_warning_and_continues(tmp_path) -> None:
    pipeline = CitationClosurePipeline(semantic_scholar=PartiallyFailingSemanticScholar(), openalex=None, unpaywall=None)

    manifest = pipeline.crawl(
        seed="seed",
        query="networking",
        out_dir=tmp_path,
        max_candidates=2,
        max_papers_to_read=2,
        ranker="lexical",
        download_pdfs=False,
        use_fallbacks=False,
    )

    paper_ids = {paper.paper_id for paper in manifest.papers}
    warning = manifest.warnings[0]

    assert "ref" in paper_ids
    assert warning.code == "citations_rate_limited"
    assert warning.paper_id == "seed"
    assert warning.relation == "citations"
    assert warning.details["status_code"] == 429


def test_pipeline_emits_progress_messages(tmp_path) -> None:
    messages = []
    pipeline = CitationClosurePipeline(semantic_scholar=FakeSemanticScholar(), openalex=None, unpaywall=None)

    pipeline.crawl(
        seed="seed",
        query="networking",
        out_dir=tmp_path,
        max_candidates=2,
        max_papers_to_read=2,
        ranker="lexical",
        download_pdfs=False,
        use_fallbacks=False,
        progress=messages.append,
    )

    assert any(message.startswith("Resolving seed:") for message in messages)
    assert any(message.startswith("Depth 1/1:") for message in messages)
    assert any(message.startswith("Expanding 1/1 at depth 1:") for message in messages)
    assert any(message.startswith("Ranking ") for message in messages)
    assert any(message.startswith("Done:") for message in messages)


def test_depth_two_expansion_budget_is_shared_across_frontier(tmp_path) -> None:
    pipeline = CitationClosurePipeline(semantic_scholar=FanoutSemanticScholar(), openalex=None, unpaywall=None)

    manifest = pipeline.crawl(
        seed="seed",
        query="networking",
        out_dir=tmp_path,
        depth=2,
        max_candidates=10,
        max_papers_to_read=3,
        ranker="lexical",
        download_pdfs=False,
        use_fallbacks=False,
    )

    paper_ids = {paper.paper_id for paper in manifest.papers}

    assert any(paper_id and paper_id.startswith("ref-a-") for paper_id in paper_ids)
    assert any(paper_id and paper_id.startswith("cite-a-") for paper_id in paper_ids)


def test_depth_caps_limit_nodes_per_depth(tmp_path) -> None:
    pipeline = CitationClosurePipeline(semantic_scholar=FanoutSemanticScholar(), openalex=None, unpaywall=None)

    manifest = pipeline.crawl(
        seed="seed",
        query="networking",
        out_dir=tmp_path,
        depth=2,
        max_candidates=20,
        depth_caps=[2, 2],
        max_papers_to_read=3,
        ranker="lexical",
        download_pdfs=False,
        use_fallbacks=False,
    )

    depth_counts = {}
    for paper in manifest.papers:
        depth_counts[paper.graph_depth] = depth_counts.get(paper.graph_depth, 0) + 1

    assert depth_counts[1] == 2
    assert depth_counts[2] == 2


def test_frontier_expansion_uses_configured_embedding_ranker(monkeypatch, tmp_path) -> None:
    class FakeEmbeddingClient:
        def __init__(self, *args, **kwargs):
            pass

        def embed(self, text):
            if text == "prefer citation side":
                return [1.0, 0.0]
            if "citation networking paper" in text:
                return [1.0, 0.0]
            return [0.0, 1.0]

    monkeypatch.setattr("citation_deep_research.pipeline.OllamaEmbeddingClient", FakeEmbeddingClient)

    pipeline = CitationClosurePipeline(semantic_scholar=FanoutSemanticScholar(), openalex=None, unpaywall=None)
    manifest = pipeline.crawl(
        seed="seed",
        query="prefer citation side",
        out_dir=tmp_path,
        depth=2,
        max_candidates=10,
        max_papers_to_read=1,
        ranker="ollama",
        download_pdfs=False,
        use_fallbacks=False,
    )

    paper_ids = {paper.paper_id for paper in manifest.papers}
    expanded = [paper for paper in manifest.papers if paper.expanded_for_crawl]

    assert any(paper_id and paper_id.startswith("cite-a-") for paper_id in paper_ids)
    assert not any(paper_id and paper_id.startswith("ref-a-") for paper_id in paper_ids)
    assert expanded


def test_max_papers_to_expand_decouples_expansion_from_reading_selection(tmp_path) -> None:
    pipeline = CitationClosurePipeline(semantic_scholar=FanoutSemanticScholar(), openalex=None, unpaywall=None)

    manifest = pipeline.crawl(
        seed="seed",
        query="networking",
        out_dir=tmp_path,
        depth=2,
        max_candidates=10,
        max_papers_to_read=1,
        max_papers_to_expand=2,
        ranker="lexical",
        download_pdfs=False,
        use_fallbacks=False,
    )

    expanded = [paper.paper_id for paper in manifest.papers if paper.expanded_for_crawl]

    assert "ref-a" in expanded
    assert "cite-a" in expanded
    assert manifest.to_dict()["counts"]["selected_for_reading"] == 1
    assert manifest.config["max_papers_to_expand"] == 2
