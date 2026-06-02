from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from .models import CitationEdge, CrawlWarning, Manifest, PaperRecord
from .pipeline import write_outputs
from .resolver import pdf_filename
from .text import normalize_doi, token_counts


@dataclass(slots=True)
class ImportResult:
    imported: list[dict]
    unmatched: list[str]
    unresolved_count: int
    source_count: int
    dry_run: bool = False


def import_pdfs(run_dir: Path, pdf_dir: Path, *, dry_run: bool = False) -> ImportResult:
    manifest = load_manifest(run_dir / "manifest.json")
    target_dir = run_dir / "pdfs"
    target_dir.mkdir(parents=True, exist_ok=True)

    unresolved = [paper for paper in manifest.papers if paper.selected_for_reading and not paper.pdf_path]
    used_sources: set[Path] = set()
    imported: list[dict] = []

    source_files = discover_pdf_files(pdf_dir)

    for paper in unresolved:
        match = best_pdf_match(paper, pdf_dir, used_sources)
        if not match:
            continue
        used_sources.add(match)
        destination = target_dir / imported_pdf_filename(paper, match)
        if not dry_run:
            shutil.copy2(match, destination)
            paper.pdf_path = str(destination)
            paper.abstract_only = False
            paper.resolution_source = "manual_import"
            paper.pdf_candidates.append(
                {
                    "source": "manual_import",
                    "url": str(match),
                    "status": "downloaded",
                    "reason": None,
                    "path": str(destination),
                    "bytes": destination.stat().st_size,
                }
            )
        imported.append(
            {
                "paper_id": paper.paper_id,
                "title": paper.title,
                "source_path": str(match),
                "destination_path": str(destination),
            }
        )

    all_pdfs = set(source_files)
    unmatched = sorted(str(path) for path in all_pdfs - used_sources)
    if not dry_run:
        write_outputs(manifest, run_dir)
    return ImportResult(
        imported=imported,
        unmatched=unmatched,
        unresolved_count=len(unresolved),
        source_count=len(source_files),
        dry_run=dry_run,
    )


def load_manifest(path: Path) -> Manifest:
    data = json.loads(path.read_text(encoding="utf-8"))
    return Manifest(
        seed=PaperRecord(**{key: value for key, value in data["seed"].items() if key in PaperRecord.__dataclass_fields__}),
        papers=[
            PaperRecord(**{key: value for key, value in paper.items() if key in PaperRecord.__dataclass_fields__})
            for paper in data["papers"]
        ],
        edges=[CitationEdge(**edge) for edge in data["edges"]],
        warnings=[CrawlWarning(**warning) for warning in data.get("warnings", [])],
        config=data["config"],
        selected_paper_ids=data["selected_paper_ids"],
        generated_at=data["generated_at"],
    )


def best_pdf_match(paper: PaperRecord, pdf_dir: Path, used_sources: set[Path]) -> Path | None:
    candidates = [path for path in discover_pdf_files(pdf_dir) if path not in used_sources]
    if not candidates:
        return None
    scored = [(pdf_match_score(paper, path), path) for path in candidates]
    scored.sort(reverse=True, key=lambda item: item[0])
    score, path = scored[0]
    return path if score >= 0.35 else None


def discover_pdf_files(pdf_dir: Path) -> list[Path]:
    files = []
    for path in pdf_dir.iterdir():
        if not path.is_file():
            continue
        if path.suffix.lower() == ".pdf" or looks_like_pdf(path):
            files.append(path)
    return files


def looks_like_pdf(path: Path) -> bool:
    try:
        return path.read_bytes()[:4] == b"%PDF"
    except OSError:
        return False


def pdf_match_score(paper: PaperRecord, path: Path) -> float:
    filename = (path.stem if path.suffix.lower() == ".pdf" else path.name).lower()
    doi = normalize_doi(paper.doi)
    if doi:
        doi_suffix = doi.split("/", 1)[-1].lower()
        if doi_suffix and doi_suffix in filename:
            return 1.0
        compact_suffix = doi_suffix.replace(".", "").replace("-", "")
        compact_filename = filename.replace(".", "").replace("-", "").replace("_", "")
        if compact_suffix and compact_suffix in compact_filename:
            return 0.95

    title_tokens = token_counts(paper.title)
    filename_tokens = token_counts(filename)
    if not title_tokens or not filename_tokens:
        return 0.0
    overlap = sum(min(title_tokens[token], filename_tokens[token]) for token in title_tokens)
    return overlap / max(1, sum(title_tokens.values()))


def imported_pdf_filename(paper: PaperRecord, source_path: Path) -> str:
    try:
        body = source_path.read_bytes()
    except OSError:
        body = source_path.name.encode()
    return pdf_filename(paper, body)
