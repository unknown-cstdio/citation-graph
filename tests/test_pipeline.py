import requests

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
