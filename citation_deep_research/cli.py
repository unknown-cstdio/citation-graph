from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .clients import OpenAlexClient, SemanticScholarClient, UnpaywallClient
from .env import load_env_file
from .importer import import_pdfs
from .pipeline import CitationClosurePipeline


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
        help="Semantic Scholar requests per second. Keep at 1.0 for the requested key.",
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
    if args.s2_rps > 1.0:
        print("Semantic Scholar rate must be <= 1 request/second for this MVP.", file=sys.stderr)
        return 2
    pipeline = CitationClosurePipeline(
        semantic_scholar=SemanticScholarClient(requests_per_second=args.s2_rps),
        openalex=OpenAlexClient() if os.getenv("OPENALEX_MAILTO") else None,
        unpaywall=UnpaywallClient() if os.getenv("UNPAYWALL_EMAIL") else None,
    )
    manifest = pipeline.crawl(
        seed=args.seed,
        query=args.query,
        out_dir=args.out,
        depth=args.depth,
        max_candidates=args.max_candidates,
        max_papers_to_read=args.max_papers_to_read,
        direction=args.direction,
        download_pdfs=not args.no_download_pdfs,
        use_fallbacks=not args.semantic_scholar_only,
        semantic_scholar_requests_per_second=args.s2_rps,
        ranker=args.ranker,
        embedding_model=args.embedding_model,
        ollama_url=args.ollama_url,
        allow_ranker_fallback=not args.no_ranker_fallback,
        depth_caps=parse_depth_caps(args.depth_caps),
    )
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
