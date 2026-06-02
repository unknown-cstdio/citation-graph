from citation_deep_research.models import CrawlWarning, Manifest, PaperRecord


def test_manifest_serializes_warning_counts() -> None:
    manifest = Manifest(
        seed=PaperRecord(paper_id="seed"),
        papers=[PaperRecord(paper_id="seed")],
        edges=[],
        warnings=[CrawlWarning(code="references_unavailable", message="missing", paper_id="seed", relation="references")],
        config={},
        selected_paper_ids=[],
        generated_at="now",
    )

    data = manifest.to_dict()

    assert data["counts"]["warnings"] == 1
    assert data["warnings"][0]["code"] == "references_unavailable"
