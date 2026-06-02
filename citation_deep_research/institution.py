from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

from .models import Manifest, PaperRecord


@dataclass(slots=True)
class InstitutionProxyConfig:
    acm_proxy_host: str | None = None
    doi_proxy_prefix: str | None = None

    @classmethod
    def from_env(cls) -> "InstitutionProxyConfig":
        return cls(
            acm_proxy_host=os.getenv("INSTITUTION_ACM_PROXY_HOST") or None,
            doi_proxy_prefix=os.getenv("INSTITUTION_DOI_PROXY_PREFIX") or None,
        )

    @property
    def enabled(self) -> bool:
        return bool(self.acm_proxy_host or self.doi_proxy_prefix)


def proxy_links_for_paper(paper: PaperRecord, config: InstitutionProxyConfig) -> list[dict[str, str]]:
    links: list[dict[str, str]] = []
    doi = (paper.doi or "").strip()
    if config.acm_proxy_host and doi.startswith("10.1145/"):
        links.append(
            {
                "source": "institution_acm_pdf",
                "url": f"https://{config.acm_proxy_host}/doi/pdf/{quote(doi, safe='/')}",
                "note": "ACM PDF through institution proxy",
            }
        )
        links.append(
            {
                "source": "institution_acm_landing",
                "url": f"https://{config.acm_proxy_host}/doi/{quote(doi, safe='/')}",
                "note": "ACM landing page through institution proxy",
            }
        )
    if config.doi_proxy_prefix and doi:
        links.append(
            {
                "source": "institution_doi",
                "url": f"{config.doi_proxy_prefix}{quote(f'https://doi.org/{doi}', safe=':/')}",
                "note": "DOI landing page through institution proxy",
            }
        )
    return links


def build_institution_link_data(manifest: Manifest, config: InstitutionProxyConfig | None = None) -> dict:
    config = config or InstitutionProxyConfig.from_env()
    papers = []
    if not config.enabled:
        return {"enabled": False, "papers": []}
    for paper in manifest.papers:
        if not paper.selected_for_reading or paper.pdf_path:
            continue
        links = proxy_links_for_paper(paper, config)
        if not links:
            continue
        papers.append(
            {
                "paper_id": paper.paper_id,
                "title": paper.title,
                "doi": paper.doi,
                "ranking_score": paper.ranking_score,
                "pdf_candidates": paper.pdf_candidates,
                "links": links,
            }
        )
    return {
        "enabled": True,
        "config": {
            "acm_proxy_host": config.acm_proxy_host,
            "doi_proxy_prefix": config.doi_proxy_prefix,
        },
        "papers": papers,
    }


def write_institution_links(manifest: Manifest, out_dir: Path, config: InstitutionProxyConfig | None = None) -> None:
    data = build_institution_link_data(manifest, config)
    json_path = out_dir / "institution-links.json"
    markdown_path = out_dir / "institution-links.md"
    pdf_text_path = out_dir / "institution-pdf-links.txt"
    landing_text_path = out_dir / "institution-landing-links.txt"
    all_text_path = out_dir / "institution-all-links.txt"
    if not data["enabled"]:
        json_path.write_text('{"enabled": false, "papers": []}\n', encoding="utf-8")
        markdown_path.write_text(
            "# Institution Access Links\n\n"
            "Institution proxy link generation is disabled. Set `INSTITUTION_ACM_PROXY_HOST` or "
            "`INSTITUTION_DOI_PROXY_PREFIX` in `.env`.\n",
            encoding="utf-8",
        )
        for path in [pdf_text_path, landing_text_path, all_text_path]:
            path.write_text("", encoding="utf-8")
        return

    import json

    json_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    lines = ["# Institution Access Links", ""]
    pdf_url_lines = []
    landing_url_lines = []
    all_url_lines = []
    if not data["papers"]:
        lines.append("No selected unresolved papers had institution proxy links to generate.")
    for paper in data["papers"]:
        lines.extend([f"## {paper['title']}", "", f"- DOI: `{paper['doi'] or ''}`"])
        for link in paper["links"]:
            lines.append(f"- [{link['source']}]({link['url']}) - {link['note']}")
            all_url_lines.append(link["url"])
            if link["source"].endswith("_pdf"):
                pdf_url_lines.append(link["url"])
            else:
                landing_url_lines.append(link["url"])
        lines.append("")
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    pdf_text_path.write_text(_url_text(pdf_url_lines), encoding="utf-8")
    landing_text_path.write_text(_url_text(landing_url_lines), encoding="utf-8")
    all_text_path.write_text(_url_text(all_url_lines), encoding="utf-8")


def _url_text(urls: list[str]) -> str:
    return "\n".join(urls) + ("\n" if urls else "")
