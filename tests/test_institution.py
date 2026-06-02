from pathlib import Path

from citation_deep_research.institution import (
    InstitutionProxyConfig,
    build_institution_link_data,
    proxy_links_for_paper,
    write_institution_links,
)
from citation_deep_research.models import Manifest, PaperRecord


def test_proxy_links_for_acm_paper() -> None:
    paper = PaperRecord(title="ACM Paper", doi="10.1145/2656877.2656890")
    config = InstitutionProxyConfig(acm_proxy_host="dl-acm-org.proxy.example.edu")

    links = proxy_links_for_paper(paper, config)

    assert links[0]["source"] == "institution_acm_pdf"
    assert links[0]["url"] == "https://dl-acm-org.proxy.example.edu/doi/pdf/10.1145/2656877.2656890"
    assert links[1]["source"] == "institution_acm_landing"


def test_institution_links_only_include_selected_unresolved_papers(tmp_path: Path) -> None:
    unresolved = PaperRecord(
        paper_id="unresolved",
        title="Needs proxy",
        doi="10.1145/1234567.1234568",
        selected_for_reading=True,
    )
    downloaded = PaperRecord(
        paper_id="downloaded",
        title="Already downloaded",
        doi="10.1145/1.2",
        selected_for_reading=True,
        pdf_path="paper.pdf",
    )
    manifest = Manifest(
        seed=unresolved,
        papers=[unresolved, downloaded],
        edges=[],
        warnings=[],
        config={},
        selected_paper_ids=["unresolved", "downloaded"],
        generated_at="now",
    )
    config = InstitutionProxyConfig(acm_proxy_host="dl-acm-org.proxy.example.edu")

    data = build_institution_link_data(manifest, config)
    write_institution_links(manifest, tmp_path, config)

    assert len(data["papers"]) == 1
    assert data["papers"][0]["title"] == "Needs proxy"
    assert "Needs proxy" in (tmp_path / "institution-links.md").read_text(encoding="utf-8")
    assert "Already downloaded" not in (tmp_path / "institution-links.md").read_text(encoding="utf-8")
    assert "https://dl-acm-org.proxy.example.edu/doi/pdf/10.1145/1234567.1234568" in (
        tmp_path / "institution-pdf-links.txt"
    ).read_text(encoding="utf-8")
    assert not (tmp_path / "institution-links.txt").exists()
    assert "https://dl-acm-org.proxy.example.edu/doi/10.1145/1234567.1234568" in (
        tmp_path / "institution-landing-links.txt"
    ).read_text(encoding="utf-8")
    assert "https://dl-acm-org.proxy.example.edu/doi/pdf/10.1145/1234567.1234568" in (
        tmp_path / "institution-all-links.txt"
    ).read_text(encoding="utf-8")
