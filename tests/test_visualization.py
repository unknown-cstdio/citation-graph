from citation_deep_research.models import Manifest, PaperRecord
from citation_deep_research.visualization import build_visualization_data, paper_status, write_graph_html


def test_paper_status_labels_download_and_selection_state() -> None:
    assert paper_status(PaperRecord(pdf_path="paper.pdf", selected_for_reading=True)) == "downloaded"
    assert paper_status(PaperRecord(pdf_url="https://example.test/paper.pdf", selected_for_reading=True)) == "pdf-url-only"
    assert paper_status(PaperRecord(selected_for_reading=True)) == "abstract-only"
    assert paper_status(PaperRecord()) == "not-selected"


def test_write_graph_html_includes_visualization_data(tmp_path) -> None:
    seed = PaperRecord(
        paper_id="seed",
        title="Seed Paper",
        selected_for_reading=True,
        pdf_path="seed.pdf",
        pdf_candidates=[{"source": "semantic_scholar", "status": "downloaded", "url": "https://example.test"}],
    )
    other = PaperRecord(paper_id="other", title="Other Paper")
    manifest = Manifest(
        seed=seed,
        papers=[seed, other],
        edges=[],
        warnings=[],
        config={"effective_ranker": "hybrid", "query": "networking"},
        selected_paper_ids=["seed"],
        generated_at="now",
    )

    data = build_visualization_data(manifest)
    write_graph_html(manifest, tmp_path)

    html = (tmp_path / "graph.html").read_text(encoding="utf-8")

    assert data["nodes"][0]["status"] == "downloaded"
    assert data["nodes"][0]["graph_depth"] is None
    assert data["nodes"][0]["pdf_candidates"][0]["status"] == "downloaded"
    assert data["nodes"][1]["status"] == "not-selected"
    assert "Seed Paper" in html
    assert "__GRAPH_DATA__" not in html
