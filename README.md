# Citation Graph Deep Research

`citation-graph-deep-research` is a sidecar citation crawler for building small, research-ready paper neighborhoods before handing the resulting PDFs to PaperQA2 or another synthesis tool.

Given a seed paper, it:

- Crawls Semantic Scholar references and citations at a strict `1 request/second`.
- Uses optional OpenAlex, Unpaywall, arXiv, and institution-proxy helpers to resolve PDFs.
- Ranks papers with lexical scoring, local Ollama embeddings, or a hybrid of both.
- Writes a manifest, citation graph JSON, browser visualization, PDF-resolution logs, and institution download link lists.
- Keeps PaperQA2 separate: this project prepares the corpus, then you run PaperQA2 directly on the downloaded PDFs.

## Install

Create a virtual environment and install the project:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Create a local `.env` file for credentials:

```bash
cp .env.example .env
```

Then edit `.env` as needed:

```bash
SEMANTIC_SCHOLAR_API_KEY=your_semantic_scholar_key
UNPAYWALL_EMAIL=you@example.edu
OPENALEX_MAILTO=you@example.edu
INSTITUTION_ACM_PROXY_HOST=dl-acm-org.proxy.example.edu
INSTITUTION_DOI_PROXY_PREFIX=https://proxy.example.edu/login?url=
```

`.env` is git-ignored. Shell exports still work and take precedence over `.env` values.

Fallback behavior is credential-aware:

- Semantic Scholar is always used.
- OpenAlex is used only when `OPENALEX_MAILTO` is set.
- Unpaywall is used only when `UNPAYWALL_EMAIL` is set.
- arXiv PDF fallback is used when Semantic Scholar metadata includes an arXiv ID.
- `--semantic-scholar-only` disables OpenAlex, Unpaywall, and arXiv fallback resolution for a run.

### Optional: Ollama Ranking

Install Ollama separately, then pull the local embedding model:

```bash
ollama pull nomic-embed-text
```

Ranking modes:

- `--ranker lexical` uses exact token overlap only.
- `--ranker ollama` uses `nomic-embed-text` embeddings through Ollama.
- `--ranker hybrid` blends Ollama embedding similarity with lexical overlap.

If Ollama is unavailable, `ollama` and `hybrid` runs fall back to lexical ranking and record a manifest warning. Use `--no-ranker-fallback` if you prefer the crawl to fail instead.

### Optional: PaperQA2

Install PaperQA2 separately if you want to synthesize over downloaded PDFs:

```bash
pip install "paper-qa>=5"
```

After a crawl has PDFs in `runs/<name>/pdfs`, run PaperQA2 directly:

```bash
cd runs/<name>/pdfs
pqa -i <name> ask "Write a concise cited survey of this citation neighborhood."
```

Use `runs/<name>/manifest.json` to cross-check that cited papers came from the crawled corpus.

## Usage

### Crawl

Run `citation-closure crawl` with a seed paper ID, DOI, URL, or title:

```bash
citation-closure crawl \
  --seed "DOI:10.1145/2656877.2656890" \
  --query "programmable data planes software defined networking packet processing network architecture" \
  --out runs/p4-smoke \
  --ranker hybrid
```

Useful crawl options:

- `--depth`: citation traversal depth. Default: `1`.
- `--depth-caps`: comma-separated per-depth discovery caps, for example `40,30`.
- `--max-candidates`: maximum discovered candidates across the run. Default: `100`.
- `--max-papers-to-read`: final top-ranked papers to resolve/download. Default: `20`.
- `--direction`: `both`, `references`, or `citations`. Default: `both`.
- `--no-download-pdfs`: resolve PDF URLs but do not download files.
- `--semantic-scholar-only`: disable fallback resolvers.
- `--ranker`: `lexical`, `ollama`, or `hybrid`. Default: `lexical`.

Default crawl policy is intentionally conservative:

- `depth=1`
- `max_candidates=100`
- `max_papers_to_read=20`
- `direction=both`
- Semantic Scholar pacing is capped at `1 request/second`

For `depth > 1`, each frontier is ranked with the configured ranker and only the top `max_papers_to_read` papers are expanded into the next depth. Expanded papers that do not make the final top-ranked reading set are kept in the graph for context, but remain marked as not selected.

Each run writes:

- `runs/<name>/manifest.json`
- `runs/<name>/graph.json`
- `runs/<name>/graph.html`
- `runs/<name>/pdfs/*.pdf`, when downloads succeed

Open `runs/<name>/graph.html` in a browser to inspect ranks, depth, selected status, download status, PDF-resolution attempts, and crawl warnings.

### Institution Links And `import-pdfs`

If institution proxy settings are configured, each crawl also writes:

- `runs/<name>/institution-links.json`
- `runs/<name>/institution-links.md`
- `runs/<name>/institution-pdf-links.txt`
- `runs/<name>/institution-landing-links.txt`
- `runs/<name>/institution-all-links.txt`

For ACM-style proxy rewriting, set:

```bash
INSTITUTION_ACM_PROXY_HOST=dl-acm-org.proxy.example.edu
```

This generates links like:

```text
https://dl-acm-org.proxy.example.edu/doi/pdf/10.1145/...
```

For generic DOI proxy links, set:

```bash
INSTITUTION_DOI_PROXY_PREFIX=https://proxy.example.edu/login?url=
```

For browser-assisted bulk downloading:

1. Open your institution proxy or one of the generated links in the browser and complete SSO/MFA login.
2. Use a browser bulk downloader such as DownThemAll.
3. Give the extension `runs/<name>/institution-pdf-links.txt`.
4. Download PDFs into a folder, for example `~/Downloads/citation-pdfs`.
5. Import those PDFs back into the run:

```bash
citation-closure import-pdfs \
  --run runs/<name> \
  --pdf-dir ~/Downloads/citation-pdfs
```

Preview matches first with:

```bash
citation-closure import-pdfs \
  --run runs/<name> \
  --pdf-dir ~/Downloads/citation-pdfs \
  --dry-run
```

The importer matches unresolved selected papers by DOI suffix first, then by title tokens. Imported PDFs are copied into `runs/<name>/pdfs/`, and `manifest.json`, `graph.json`, and `graph.html` are regenerated. Extensionless files are accepted when their contents start with the PDF magic bytes `%PDF`.

## Example

This small Semantic-Scholar-only run is a good smoke test. It uses a P4/networking seed paper known to return both reference and citation records.

```bash
citation-closure crawl \
  --seed "DOI:10.1145/2656877.2656890" \
  --query "programmable data planes software defined networking packet processing network architecture" \
  --out runs/readme-example \
  --depth 1 \
  --max-candidates 8 \
  --max-papers-to-read 4 \
  --direction both \
  --ranker lexical \
  --semantic-scholar-only \
  --no-download-pdfs
```

Expected output files:

- `runs/readme-example/manifest.json`
- `runs/readme-example/graph.json`
- `runs/readme-example/graph.html`

## Development Tests

```bash
pytest
```
