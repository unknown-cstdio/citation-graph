from __future__ import annotations

import os
import random
import re
import time
from typing import Any, Callable
from urllib.parse import quote

import requests

from .models import CrawlWarning, PaperRecord
from .rate_limit import RateLimiter
from .text import extract_doi, normalize_doi


RETRY_STATUSES = {429, 500, 502, 503, 504}


class HttpRequestError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        url: str,
        status_code: int | None = None,
        response_text: str | None = None,
    ) -> None:
        super().__init__(message)
        self.url = url
        self.status_code = status_code
        self.response_text = response_text


class HttpClient:
    def __init__(
        self,
        *,
        base_url: str,
        headers: dict[str, str] | None = None,
        rate_limiter: RateLimiter | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_base_sleep: float = 1.0,
        retry_max_sleep: float = 60.0,
        retry_jitter: float = 0.25,
        throttle_cooldown: float = 30.0,
        retry_progress: Callable[[str], None] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.rate_limiter = rate_limiter
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_base_sleep = retry_base_sleep
        self.retry_max_sleep = retry_max_sleep
        self.retry_jitter = retry_jitter
        self.throttle_cooldown = throttle_cooldown
        self.retry_progress = retry_progress
        self.session = requests.Session()
        self.session.headers.update(headers or {})

    def get(self, path: str, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
        url = path if path.startswith("http") else f"{self.base_url}/{path.lstrip('/')}"
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            if self.rate_limiter:
                self.rate_limiter.wait()
            try:
                response = self.session.get(url, params=params, timeout=self.timeout)
                if response.status_code in RETRY_STATUSES and attempt < self.max_retries:
                    sleep_for = self.retry_sleep(response, attempt)
                    self.emit_retry_progress(url, response.status_code, attempt, sleep_for)
                    time.sleep(sleep_for)
                    continue
                response.raise_for_status()
                return response.json()
            except requests.RequestException as exc:
                if getattr(exc, "response", None) is not None:
                    response_text = exc.response.text[:500]
                    last_error = HttpRequestError(
                        f"{exc}; response={response_text}",
                        url=url,
                        status_code=exc.response.status_code,
                        response_text=response_text,
                    )
                else:
                    last_error = exc
                if attempt >= self.max_retries:
                    break
                response = getattr(exc, "response", None)
                sleep_for = self.retry_sleep(response, attempt)
                self.emit_retry_progress(url, getattr(response, "status_code", None), attempt, sleep_for)
                time.sleep(sleep_for)
        if isinstance(last_error, HttpRequestError):
            raise last_error
        raise HttpRequestError(f"GET failed for {url}: {last_error}", url=url) from last_error

    def retry_sleep(self, response: requests.Response | None, attempt: int) -> float:
        if response is not None:
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                try:
                    return max(0.0, min(float(retry_after), self.retry_max_sleep))
                except ValueError:
                    pass
        sleep_for = self.retry_base_sleep * (2**attempt)
        if response is not None and response.status_code == 429:
            sleep_for = max(sleep_for, self.throttle_cooldown)
        if self.retry_jitter > 0:
            sleep_for += random.uniform(0, self.retry_jitter)
        return max(0.0, min(sleep_for, self.retry_max_sleep))

    def emit_retry_progress(self, url: str, status_code: int | None, attempt: int, sleep_for: float) -> None:
        if not self.retry_progress:
            return
        status = status_code if status_code is not None else "network error"
        self.retry_progress(
            f"Semantic Scholar request got {status}; retry {attempt + 1}/{self.max_retries} "
            f"in {sleep_for:.1f}s: {url}"
        )


class SemanticScholarClient:
    BASE_URL = "https://api.semanticscholar.org/graph/v1"
    PAPER_FIELDS = ",".join(
        [
            "paperId",
            "title",
            "abstract",
            "year",
            "venue",
            "authors",
            "externalIds",
            "url",
            "openAccessPdf",
            "isOpenAccess",
            "citationCount",
            "referenceCount",
        ]
    )
    RELATION_FIELDS = ",".join(
        [
            "contexts",
            "intents",
            "isInfluential",
            "paperId",
            "title",
            "abstract",
            "year",
            "venue",
            "authors",
            "externalIds",
            "url",
            "openAccessPdf",
            "citationCount",
        ]
    )

    def __init__(
        self,
        api_key: str | None = None,
        requests_per_second: float = 1.0,
        *,
        max_retries: int = 8,
        throttle_cooldown: float = 30.0,
        retry_progress: Callable[[str], None] | None = None,
    ) -> None:
        api_key = api_key or os.getenv("SEMANTIC_SCHOLAR_API_KEY")
        headers = {"User-Agent": "citation-graph-deep-research/0.1.0"}
        if api_key:
            headers["x-api-key"] = api_key
        self.http = HttpClient(
            base_url=self.BASE_URL,
            headers=headers,
            rate_limiter=RateLimiter(requests_per_second),
            max_retries=max_retries,
            throttle_cooldown=throttle_cooldown,
            retry_progress=retry_progress,
        )

    def resolve_seed(self, seed: str) -> PaperRecord:
        identifier = self._seed_to_identifier(seed)
        if identifier:
            return self.get_paper(identifier)
        return self.search_title(seed)

    def get_paper(self, identifier: str) -> PaperRecord:
        data = self.http.get(f"/paper/{quote(identifier, safe=':')}", params={"fields": self.PAPER_FIELDS})
        return paper_from_s2(data)

    def search_title(self, title: str) -> PaperRecord:
        data = self.http.get(
            "/paper/search",
            params={"query": title, "limit": 1, "fields": self.PAPER_FIELDS},
        )
        papers = data.get("data") or []
        if not papers:
            raise RuntimeError(f"No Semantic Scholar paper found for title query: {title}")
        return paper_from_s2(papers[0])

    def references(self, paper_id: str, *, limit: int = 100) -> tuple[list[PaperRecord], list[CrawlWarning]]:
        return self._relation_papers(paper_id, relation="references", child_key="citedPaper", limit=limit)

    def citations(self, paper_id: str, *, limit: int = 100) -> tuple[list[PaperRecord], list[CrawlWarning]]:
        return self._relation_papers(paper_id, relation="citations", child_key="citingPaper", limit=limit)

    def _relation_papers(
        self,
        paper_id: str,
        *,
        relation: str,
        child_key: str,
        limit: int,
    ) -> tuple[list[PaperRecord], list[CrawlWarning]]:
        papers: list[PaperRecord] = []
        warnings: list[CrawlWarning] = []
        offset = 0
        page_limit = min(100, max(1, limit))
        while len(papers) < limit:
            data = self.http.get(
                f"/paper/{paper_id}/{relation}",
                params={"fields": self.RELATION_FIELDS, "offset": offset, "limit": page_limit},
            )
            if data.get("data") is None:
                warning = relation_warning(paper_id, relation, data)
                if warning:
                    warnings.append(warning)
                break
            rows = data.get("data") or []
            if not rows:
                break
            for row in rows:
                child = row.get(child_key)
                if child:
                    papers.append(paper_from_s2(child))
                    if len(papers) >= limit:
                        break
            next_offset = data.get("next")
            if next_offset is None:
                break
            offset = int(next_offset)
        return papers, warnings

    def _seed_to_identifier(self, seed: str) -> str | None:
        value = seed.strip()
        if re.fullmatch(r"[a-f0-9]{40}", value, flags=re.IGNORECASE):
            return value
        if value.lower().startswith(("doi:", "arxiv:", "corpusid:")):
            return value
        doi = extract_doi(value)
        if doi:
            return f"DOI:{doi}"
        s2_match = re.search(r"semanticscholar\.org/(?:paper/[^/]+/)?([a-f0-9]{40})", value, re.IGNORECASE)
        if s2_match:
            return s2_match.group(1)
        return None


class OpenAlexClient:
    BASE_URL = "https://api.openalex.org"

    def __init__(self, mailto: str | None = None) -> None:
        self.mailto = mailto or os.getenv("OPENALEX_MAILTO")
        self.http = HttpClient(base_url=self.BASE_URL, headers={"User-Agent": "citation-graph-deep-research/0.1.0"})

    def lookup(self, *, doi: str | None = None, title: str | None = None) -> dict[str, Any] | None:
        params: dict[str, Any] = {"per-page": 1}
        if self.mailto:
            params["mailto"] = self.mailto
        if doi:
            try:
                return self.http.get(f"/works/https://doi.org/{quote(doi, safe='')}", params=params)
            except RuntimeError:
                pass
        if title:
            params["search"] = title
            data = self.http.get("/works", params=params)
            results = data.get("results") or []
            return results[0] if results else None
        return None


class UnpaywallClient:
    BASE_URL = "https://api.unpaywall.org/v2"

    def __init__(self, email: str | None = None) -> None:
        self.email = email or os.getenv("UNPAYWALL_EMAIL")
        self.http = HttpClient(base_url=self.BASE_URL, headers={"User-Agent": "citation-graph-deep-research/0.1.0"})

    def lookup(self, doi: str | None) -> dict[str, Any] | None:
        if not doi or not self.email:
            return None
        try:
            return self.http.get(f"/{quote(doi, safe='')}", params={"email": self.email})
        except RuntimeError as exc:
            if "404 Client Error" in str(exc):
                return None
            raise


def paper_from_s2(data: dict[str, Any]) -> PaperRecord:
    external_ids = data.get("externalIds") or {}
    doi = normalize_doi(external_ids.get("DOI"))
    open_access_pdf = data.get("openAccessPdf") or {}
    return PaperRecord(
        paper_id=data.get("paperId"),
        title=data.get("title"),
        abstract=data.get("abstract"),
        doi=doi,
        arxiv_id=external_ids.get("ArXiv"),
        url=data.get("url"),
        pdf_url=open_access_pdf.get("url"),
        year=data.get("year"),
        venue=data.get("venue"),
        authors=[author.get("name") for author in data.get("authors") or [] if author.get("name")],
        citation_count=data.get("citationCount"),
        source="semantic_scholar",
        raw={"semantic_scholar": data},
    )


def relation_warning(paper_id: str, relation: str, data: dict[str, Any]) -> CrawlWarning | None:
    info_key = "citingPaperInfo" if relation == "references" else "citedPaperInfo"
    info = data.get(info_key) or {}
    disclaimer = ((info.get("openAccessPdf") or {}).get("disclaimer") or "").strip()
    if "elided by the publisher" in disclaimer:
        return CrawlWarning(
            code=f"{relation}_publisher_elided",
            message=f"Semantic Scholar did not return {relation}; publisher elided this field.",
            paper_id=paper_id,
            relation=relation,
            details={
                "title": info.get("title"),
                "disclaimer": disclaimer,
            },
        )
    return CrawlWarning(
        code=f"{relation}_unavailable",
        message=f"Semantic Scholar returned no {relation} data for this paper.",
        paper_id=paper_id,
        relation=relation,
        details={"title": info.get("title")},
    )


def openalex_abstract(work: dict[str, Any]) -> str | None:
    inverted = work.get("abstract_inverted_index")
    if not inverted:
        return None
    positions: dict[int, str] = {}
    for token, indexes in inverted.items():
        for index in indexes:
            positions[int(index)] = token
    return " ".join(positions[index] for index in sorted(positions))


def enrich_from_openalex(paper: PaperRecord, work: dict[str, Any] | None) -> None:
    if not work:
        return
    paper.openalex_id = paper.openalex_id or work.get("id")
    paper.doi = paper.doi or normalize_doi(work.get("doi"))
    paper.abstract = paper.abstract or openalex_abstract(work)
    paper.title = paper.title or work.get("display_name")
    best_oa = work.get("best_oa_location") or {}
    primary = work.get("primary_location") or {}
    best_pdf = best_oa.get("pdf_url") or primary.get("pdf_url")
    paper.pdf_url = paper.pdf_url or best_pdf
    paper.raw["openalex"] = work
