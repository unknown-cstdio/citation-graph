from pathlib import Path

from citation_deep_research.importer import discover_pdf_files, import_pdfs, pdf_match_score
from citation_deep_research.models import Manifest, PaperRecord
from citation_deep_research.pipeline import write_outputs


def test_pdf_match_score_matches_acm_doi_suffix() -> None:
    paper = PaperRecord(title="Tiny packet programs", doi="10.1145/2535771.2535780")
    path = Path("2535771.2535780.pdf")

    assert pdf_match_score(paper, path) == 1.0


def test_pdf_match_score_matches_extensionless_acm_doi_suffix() -> None:
    paper = PaperRecord(title="Tiny packet programs", doi="10.1145/2535771.2535780")
    path = Path("2535771.2535780")

    assert pdf_match_score(paper, path) == 1.0


def test_import_pdfs_updates_manifest_and_regenerates_outputs(tmp_path: Path) -> None:
    run = tmp_path / "run"
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    downloaded_pdf = downloads / "2535771.2535780.pdf"
    downloaded_pdf.write_bytes(b"%PDF fake")

    paper = PaperRecord(
        paper_id="paper",
        title="Tiny packet programs for low-latency network control and monitoring",
        doi="10.1145/2535771.2535780",
        selected_for_reading=True,
    )
    manifest = Manifest(
        seed=paper,
        papers=[paper],
        edges=[],
        warnings=[],
        config={},
        selected_paper_ids=["paper"],
        generated_at="now",
    )
    run.mkdir()
    write_outputs(manifest, run)

    result = import_pdfs(run, downloads)

    assert len(result.imported) == 1
    assert result.source_count == 1
    assert result.unresolved_count == 1
    assert (run / "manifest.json").exists()
    assert (run / "graph.html").exists()
    assert list((run / "pdfs").glob("*.pdf"))


def test_discover_pdf_files_accepts_extensionless_pdf(tmp_path: Path) -> None:
    extensionless = tmp_path / "2535771.2535780"
    extensionless.write_bytes(b"%PDF fake")
    non_pdf = tmp_path / "not-a-pdf"
    non_pdf.write_text("html", encoding="utf-8")

    files = discover_pdf_files(tmp_path)

    assert files == [extensionless]
