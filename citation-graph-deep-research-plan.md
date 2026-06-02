# Citation-Graph-Closure Deep Research — Build Plan

## Goal
Build a tool that, given a seed paper (or research question), does a **citation-graph-closure deep research**: BFS-expand outward through the citation network, read the papers it finds, and synthesize a survey — all using a **locally-hosted LLM** (privacy + cost).

## Recommendation: Fork PaperQA2 and add citation closure on top

After searching GitHub, no existing open-source tool combines (local LLM) + (citation-graph-closure) + (deep research synthesis). The closest base is:

**Future-House/paper-qa (PaperQA2)** — 8.5k ⭐, https://github.com/Future-House/paper-qa
- RAG over scientific PDFs with agentic workflows and LLM re-ranking
- Already supports local LLMs via llama.cpp / Ollama
- Answers include citations
- **Gap:** no citation-graph traversal. Maintainers say on the README: _"we're trying to get citation traversal into this repo"_ — but it's not there.

That gap is exactly what we want to build. Forking means we get ingestion, RAG, re-ranking, and local-LLM plumbing for free, and only have to write the expansion loop.

## Why not the alternatives

- **gpt-researcher (27k ⭐), dzhng/deep-research (18.9k ⭐)** — general deep-research agents, but web-search oriented, not citation-graph oriented. Wrong abstraction.
- **Connected Papers clones** (zjlww/_papers, research-grapher) — render the graph, don't read the papers. No LLM synthesis.
- **Small Ollama + academic agents** (ScientificPaperAgent, AutoScholar, etc.) — all 0–30 stars, each does one piece (arxiv search, RAG over one PDF). Nothing to fork with momentum.
- **Commercial tools** (Elicit, Consensus, Undermind, FutureHouse Aviary, SciSpace) — closed-source, not local.

## Target stack

| Layer | Choice | Why |
|---|---|---|
| Local LLM serving | Ollama (easy) or vLLM (fast on real GPU) | OpenAI-compatible API |
| LLM model | Qwen2.5-32B-Instruct or Llama-3.3-70B | ≥32k context needed for cross-paper synthesis |
| Citation source | Semantic Scholar Graph API (primary), OpenAlex (fallback) | Free; references + citations edges; abstracts; embeddings |
| PDF access | Unpaywall → arXiv → institutional proxy → abstract-only | Legal OA coverage is better than most assume |
| PDF → text | GROBID (best for scientific structure) or marker/nougat (faster) | Structured refs extraction |
| Embeddings | nomic-embed-text or bge-large (local) | For relevance gating |
| Vector store | Qdrant or LanceDB | Local, fast |
| Graph lib | NetworkX | Fine up to ~100k nodes; igraph if bigger |
| Orchestration | PaperQA2's existing agent loop + our expansion layer | |

## Architecture: what to add on top of PaperQA2

1. **Citation-graph crawler** (new module)
   - Seed with N papers → BFS over `references` + `citations` via S2 API
   - Deduplicate by DOI → S2 paperId → normalized title
   - Relevance-gated expansion: embed each candidate, only expand if `cosine(query, paper) > τ`
   - Budget cap: max depth, max nodes, max papers-read
   - Output: a NetworkX graph + a queue of papers to ingest

2. **PDF resolution chain** (new module)
   - Try Unpaywall → arXiv preprint link from S2 → institutional proxy → fallback to abstract
   - Mark abstract-only papers so synthesis is honest about what it did/didn't read

3. **Ingest into PaperQA2** (existing)
   - Feed resolved PDFs + metadata into PaperQA2's indexer
   - Let its RAG + re-ranker do per-paper work

4. **Cross-paper synthesis** (extend PaperQA2's agent)
   - Map: per-paper structured summary (claim / method / evidence / limits)
   - Cluster by topic (embedding k-means or community detection on citation graph)
   - Reduce: LLM writes survey with inline `[paperId]` citations
   - Render the citation graph (who-cites-whom, consensus vs. disputed)

## Open questions for tomorrow

- **S2 API key**: apply at https://www.semanticscholar.org/product/api _today if possible_ — approval takes a few days. Without it, crawler is capped at ~100 req/5min.
- **Model size vs. machine**: what GPU are we targeting? Drives model choice (32B fits on 24–48GB; 70B needs quantization or multi-GPU).
- **Closure policy knobs**: what are the defaults for max depth, max nodes, relevance threshold τ? Pick something conservative (e.g., depth=2, max 300 papers, τ=0.5) and tune.
- **Institutional proxy**: does Trinity have EZproxy / OpenAthens? If so, wire it into the PDF chain.
- **Hybrid option**: local LLM for per-paper extraction, Claude API (Sonnet 4.6 / Opus 4.7) for the final synthesis call — cheaper than all-cloud, better quality than all-local. Decide whether privacy is strict or flexible.

## Tomorrow's first steps

1. `cd ~/Dropbox/work/repos/`
2. `git clone https://github.com/Future-House/paper-qa.git`
3. Read PaperQA2's agent entry points + indexer interfaces — find where to hook the citation-closure crawler in (likely upstream of the ingest step).
4. Apply for S2 API key (if not already done).
5. Write a throwaway script: given one paperId, do 1-hop BFS via S2, print the 20 most relevant neighbors by embedding similarity. That proves the crawler primitive works before we wire it into PaperQA2.
6. Decide on the stack knobs (GPU/model, closure policy) via the open questions above.

## Reference links
- PaperQA2: https://github.com/Future-House/paper-qa
- Semantic Scholar API: https://api.semanticscholar.org/
- Unpaywall: https://unpaywall.org/products/api
- GROBID: https://github.com/kermitt2/grobid
- Ollama: https://ollama.com
