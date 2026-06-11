from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .clients import OpenAlexClient, SemanticScholarClient, UnpaywallClient
from .env import load_env_file
from .importer import import_pdfs
from .pipeline import CitationClosurePipeline, stderr_progress


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="citation-closure",
        description="Crawl a small citation neighborhood and hand PDFs to PaperQA2.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    crawl = subparsers.add_parser("crawl", help="Crawl and resolve a citation neighborhood.")
    crawl.add_argument("--seed", required=True, help="Semantic Scholar paperId, DOI, Semantic Scholar URL, or title.")
    crawl.add_argument("--query", help="Research question used for relevance ranking. Defaults to seed title/abstract.")
    crawl.add_argument("--out", required=True, type=Path, help="Output run directory.")
    crawl.add_argument("--depth", type=int, default=1, help="Citation BFS depth. Default: 1.")
    crawl.add_argument("--max-candidates", type=int, default=100, help="Maximum discovered candidates. Default: 100.")
    crawl.add_argument(
        "--depth-caps",
        help="Comma-separated per-depth candidate caps, e.g. '40,30' for depth 1 and depth 2.",
    )
    crawl.add_argument("--max-papers-to-read", type=int, default=20, help="Top papers to resolve/download. Default: 20.")
    crawl.add_argument(
        "--max-papers-to-expand",
        type=int,
        help="Top ranked frontier papers to expand at each depth after the seed. Defaults to --max-papers-to-read.",
    )
    crawl.add_argument(
        "--direction",
        choices=["both", "references", "citations"],
        default="both",
        help="Citation traversal direction. Default: both.",
    )
    crawl.add_argument(
        "--no-download-pdfs",
        action="store_true",
        help="Resolve PDF URLs but do not download files.",
    )
    crawl.add_argument(
        "--semantic-scholar-only",
        action="store_true",
        help="Disable OpenAlex, Unpaywall, and arXiv fallback resolution for this run.",
    )
    crawl.add_argument(
        "--s2-rps",
        type=float,
        default=1.0,
        help="Semantic Scholar requests per second. Use <= 1.0; try 0.5-0.75 for larger crawls. Default: 1.0.",
    )
    crawl.add_argument(
        "--s2-max-retries",
        type=int,
        default=8,
        help="Maximum retries for throttled/transient Semantic Scholar requests. Default: 8.",
    )
    crawl.add_argument(
        "--s2-429-cooldown",
        type=float,
        default=30.0,
        help="Minimum seconds to wait before retrying a Semantic Scholar 429 without Retry-After. Default: 30.",
    )
    crawl.add_argument(
        "--ranker",
        choices=["lexical", "ollama", "hybrid"],
        default="lexical",
        help="Ranking method. Default: lexical.",
    )
    crawl.add_argument(
        "--embedding-model",
        default="nomic-embed-text",
        help="Ollama embedding model for --ranker ollama/hybrid. Default: nomic-embed-text.",
    )
    crawl.add_argument(
        "--ollama-url",
        default=os.getenv("OLLAMA_URL", "http://localhost:11434"),
        help="Ollama base URL for embedding rankers. Default: http://localhost:11434.",
    )
    crawl.add_argument(
        "--no-ranker-fallback",
        action="store_true",
        help="Fail instead of falling back to lexical ranking if Ollama embedding fails.",
    )
    crawl.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress crawl progress messages. Final counts are still printed.",
    )

    import_cmd = subparsers.add_parser("import-pdfs", help="Import manually downloaded PDFs into a crawl run.")
    import_cmd.add_argument("--run", required=True, type=Path, help="Run directory containing manifest.json.")
    import_cmd.add_argument("--pdf-dir", required=True, type=Path, help="Directory containing browser-downloaded PDFs.")
    import_cmd.add_argument(
        "--dry-run",
        action="store_true",
        help="Show matches without copying PDFs or updating run artifacts.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    load_env_file()
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "crawl":
        return crawl_command(args)
    if args.command == "import-pdfs":
        return import_pdfs_command(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


def crawl_command(args: argparse.Namespace) -> int:
    if args.s2_rps <= 0 or args.s2_rps > 1.0:
        print("Semantic Scholar rate must be > 0 and <= 1 request/second for this MVP.", file=sys.stderr)
        return 2
    if args.s2_max_retries < 0:
        print("Semantic Scholar max retries must be >= 0.", file=sys.stderr)
        return 2
    if args.s2_429_cooldown < 0:
        print("Semantic Scholar 429 cooldown must be >= 0.", file=sys.stderr)
        return 2
    if args.max_papers_to_expand is not None and args.max_papers_to_expand <= 0:
        print("Max papers to expand must be positive.", file=sys.stderr)
        return 2
    progress = None if args.quiet else stderr_progress
    if progress:
        auth_status = "yes" if os.getenv("SEMANTIC_SCHOLAR_API_KEY") else "no"
        progress(f"Semantic Scholar API key loaded: {auth_status}")
    pipeline = CitationClosurePipeline(
        semantic_scholar=SemanticScholarClient(
            requests_per_second=args.s2_rps,
            max_retries=args.s2_max_retries,
            throttle_cooldown=args.s2_429_cooldown,
            retry_progress=progress,
        ),
        openalex=OpenAlexClient() if os.getenv("OPENALEX_MAILTO") else None,
        unpaywall=UnpaywallClient() if os.getenv("UNPAYWALL_EMAIL") else None,
    )
    try:
        manifest = pipeline.crawl(
            seed=args.seed,
            query=args.query,
            out_dir=args.out,
            depth=args.depth,
            max_candidates=args.max_candidates,
            max_papers_to_read=args.max_papers_to_read,
            max_papers_to_expand=args.max_papers_to_expand,
            direction=args.direction,
            download_pdfs=not args.no_download_pdfs,
            use_fallbacks=not args.semantic_scholar_only,
            semantic_scholar_requests_per_second=args.s2_rps,
            semantic_scholar_max_retries=args.s2_max_retries,
            semantic_scholar_429_cooldown=args.s2_429_cooldown,
            ranker=args.ranker,
            embedding_model=args.embedding_model,
            ollama_url=args.ollama_url,
            allow_ranker_fallback=not args.no_ranker_fallback,
            depth_caps=parse_depth_caps(args.depth_caps),
            progress=progress,
        )
    except RuntimeError as exc:
        print(f"Crawl failed before outputs could be written: {exc}", file=sys.stderr)
        return 1
    data = manifest.to_dict()
    print(json.dumps(data["counts"], indent=2, sort_keys=True))
    print(f"Wrote manifest: {args.out / 'manifest.json'}")
    print(f"Wrote graph: {args.out / 'graph.json'}")
    return 0


def parse_depth_caps(value: str | None) -> list[int] | None:
    if not value:
        return None
    caps = []
    for part in value.split(","):
        stripped = part.strip()
        if not stripped:
            continue
        caps.append(int(stripped))
    return caps or None


def import_pdfs_command(args: argparse.Namespace) -> int:
    manifest_path = args.run / "manifest.json"
    if not manifest_path.exists():
        print(f"Missing manifest: {manifest_path}", file=sys.stderr)
        return 1
    if not args.pdf_dir.exists():
        print(f"Missing PDF directory: {args.pdf_dir}", file=sys.stderr)
        return 1
    result = import_pdfs(args.run, args.pdf_dir, dry_run=args.dry_run)
    payload = {
        "source_pdfs_found": result.source_count,
        "unresolved_selected_papers": result.unresolved_count,
        "imported_count": len(result.imported),
        "unmatched_count": len(result.unmatched),
        "imported": result.imported,
        "unmatched": result.unmatched,
        "dry_run": result.dry_run,
    }
    print(json.dumps(payload, indent=2))
    if result.source_count == 0:
        print("No PDF files were found. The importer accepts .pdf files and extensionless files whose contents start with %PDF.", file=sys.stderr)
    elif result.imported:
        action = "Would import" if args.dry_run else "Imported"
        print(f"{action} {len(result.imported)} PDF(s).")
    else:
        print(
            "No PDFs matched unresolved selected papers. Check filenames or try renaming downloads to include the DOI suffix.",
            file=sys.stderr,
        )
    if not args.dry_run:
        print(f"Updated manifest: {manifest_path}")
        print(f"Updated graph: {args.run / 'graph.json'}")
        print(f"Updated visualization: {args.run / 'graph.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
