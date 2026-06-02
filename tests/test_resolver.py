from pathlib import Path

from citation_deep_research.models import PaperRecord
from citation_deep_research.resolver import PdfResolver


class NullOpenAlex:
    def lookup(self, *, doi=None, title=None):
        return None


class NullUnpaywall:
    def lookup(self, doi):
        return None


def test_resolver_marks_abstract_only_when_no_pdf(tmp_path: Path) -> None:
    resolver = PdfResolver(openalex=NullOpenAlex(), unpaywall=NullUnpaywall())
    paper = PaperRecord(title="No PDF", abstract="Only abstract text")

    resolved = resolver.resolve(paper, tmp_path)

    assert resolved.abstract_only is True
    assert resolved.pdf_path is None
    assert resolved.pdf_candidates == [
        {
            "source": "resolver",
            "url": None,
            "status": "no_candidates",
            "reason": "no_pdf_url_found",
        }
    ]


def test_resolver_skips_fallbacks_in_semantic_scholar_only_mode(tmp_path: Path) -> None:
    class ExplodingOpenAlex:
        def lookup(self, *, doi=None, title=None):
            raise AssertionError("OpenAlex should not be called")

    class ExplodingUnpaywall:
        def lookup(self, doi):
            raise AssertionError("Unpaywall should not be called")

    resolver = PdfResolver(
        openalex=ExplodingOpenAlex(),
        unpaywall=ExplodingUnpaywall(),
        use_fallbacks=False,
        download_pdfs=False,
    )
    paper = PaperRecord(title="S2 PDF", pdf_url="https://example.edu/paper.pdf")

    resolved = resolver.resolve(paper, tmp_path)

    assert resolved.abstract_only is False
    assert resolved.resolution_source == "semantic_scholar"
    assert resolved.pdf_candidates == [
        {
            "source": "semantic_scholar",
            "url": "https://example.edu/paper.pdf",
            "status": "candidate",
            "reason": "download_disabled",
        }
    ]


def test_resolver_uses_arxiv_only_when_fallbacks_enabled(tmp_path: Path) -> None:
    resolver = PdfResolver(
        openalex=None,
        unpaywall=None,
        use_fallbacks=False,
        download_pdfs=False,
    )
    paper = PaperRecord(title="arxiv paper", arxiv_id="2401.12345")

    resolved = resolver.resolve(paper, tmp_path)

    assert resolved.abstract_only is True
    assert resolved.resolution_source is None


def test_resolver_logs_failed_pdf_download(tmp_path: Path) -> None:
    class FakeSession:
        def get(self, url, stream, timeout, allow_redirects):
            raise __import__("requests").ConnectionError("offline")

    resolver = PdfResolver(openalex=NullOpenAlex(), unpaywall=NullUnpaywall())
    resolver.session = FakeSession()
    paper = PaperRecord(title="Broken PDF", pdf_url="https://example.edu/missing.pdf")

    resolved = resolver.resolve(paper, tmp_path)

    assert resolved.abstract_only is True
    assert resolved.pdf_candidates[0]["source"] == "semantic_scholar"
    assert resolved.pdf_candidates[0]["status"] == "failed"
    assert resolved.pdf_candidates[0]["reason"] == "ConnectionError"
