from __future__ import annotations

import hashlib
import re
from pathlib import Path
from urllib.parse import quote

import requests

from .clients import OpenAlexClient, UnpaywallClient, enrich_from_openalex
from .models import PaperRecord


PDF_MAGIC = b"%PDF"


class PdfResolver:
    def __init__(
        self,
        *,
        openalex: OpenAlexClient | None,
        unpaywall: UnpaywallClient | None,
        download_pdfs: bool = True,
        use_fallbacks: bool = True,
        timeout: float = 45.0,
    ) -> None:
        self.openalex = openalex
        self.unpaywall = unpaywall
        self.download_pdfs = download_pdfs
        self.use_fallbacks = use_fallbacks
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "citation-graph-deep-research/0.1.0"})

    def resolve(self, paper: PaperRecord, pdf_dir: Path) -> PaperRecord:
        paper.pdf_candidates = []
        if self.use_fallbacks and self.openalex:
            work = self.openalex.lookup(doi=paper.doi, title=paper.title)
            enrich_from_openalex(paper, work)

        candidates = self._candidate_urls(paper)
        if not candidates:
            paper.pdf_candidates.append(
                {
                    "source": "resolver",
                    "url": None,
                    "status": "no_candidates",
                    "reason": "no_pdf_url_found",
                }
            )
            paper.abstract_only = True
            return paper
        for source, url in candidates:
            paper.pdf_url = paper.pdf_url or url
            if not self.download_pdfs:
                paper.pdf_candidates.append(
                    {
                        "source": source,
                        "url": url,
                        "status": "candidate",
                        "reason": "download_disabled",
                    }
                )
                paper.abstract_only = False
                paper.resolution_source = source
                return paper
            result = self._download_pdf(source, url, pdf_dir, paper)
            paper.pdf_candidates.append(result)
            if result["status"] == "downloaded":
                path = result["path"]
                paper.pdf_path = str(path)
                paper.abstract_only = False
                paper.resolution_source = source
                return paper

        paper.abstract_only = True
        return paper

    def _candidate_urls(self, paper: PaperRecord) -> list[tuple[str, str]]:
        candidates: list[tuple[str, str]] = []
        seen: set[str] = set()

        def add(source: str, url: str | None) -> None:
            if url and url not in seen:
                seen.add(url)
                candidates.append((source, url))

        add("semantic_scholar", paper.pdf_url)

        if self.use_fallbacks:
            openalex = paper.raw.get("openalex") or {}
            best_oa = openalex.get("best_oa_location") or {}
            primary = openalex.get("primary_location") or {}
            add("openalex", best_oa.get("pdf_url") or primary.get("pdf_url"))

            if self.unpaywall:
                unpaywall = self.unpaywall.lookup(paper.doi)
                if unpaywall:
                    best_location = unpaywall.get("best_oa_location") or {}
                    add("unpaywall", best_location.get("url_for_pdf") or best_location.get("url"))
                    paper.raw["unpaywall"] = unpaywall

        if self.use_fallbacks and paper.arxiv_id:
            add("arxiv", f"https://arxiv.org/pdf/{quote(paper.arxiv_id)}.pdf")

        return candidates

    def _download_pdf(self, source: str, url: str, pdf_dir: Path, paper: PaperRecord) -> dict:
        pdf_dir.mkdir(parents=True, exist_ok=True)
        result = {"source": source, "url": url, "status": "failed", "reason": None, "path": None}
        try:
            response = self.session.get(url, stream=True, timeout=self.timeout, allow_redirects=True)
            response.raise_for_status()
        except requests.RequestException as exc:
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            result["reason"] = f"http_{status_code}" if status_code else exc.__class__.__name__
            return result

        content_type = response.headers.get("content-type", "").lower()
        chunks: list[bytes] = []
        total = 0
        for chunk in response.iter_content(chunk_size=65536):
            if not chunk:
                continue
            chunks.append(chunk)
            total += len(chunk)
            if total > 100 * 1024 * 1024:
                result["status"] = "too_large"
                result["reason"] = "over_100mb"
                result["content_type"] = content_type
                return result

        body = b"".join(chunks)
        if not body.startswith(PDF_MAGIC) and "pdf" not in content_type:
            result["status"] = "not_pdf"
            result["reason"] = "content_type_or_magic_mismatch"
            result["content_type"] = content_type
            result["bytes"] = len(body)
            return result

        path = pdf_dir / pdf_filename(paper, body)
        path.write_bytes(body)
        result["status"] = "downloaded"
        result["reason"] = None
        result["path"] = str(path)
        result["content_type"] = content_type
        result["bytes"] = len(body)
        return result


def pdf_filename(paper: PaperRecord, body: bytes) -> str:
    title = paper.title or paper.paper_id or paper.doi or "paper"
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", title).strip("-").lower()[:80] or "paper"
    digest = hashlib.sha1(body).hexdigest()[:10]
    return f"{slug}-{digest}.pdf"
