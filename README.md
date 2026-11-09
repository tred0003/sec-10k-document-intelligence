# SEC 10-K Document Intelligence & RAG

Pulls financial-statement numbers out of real 10-K annual reports (PDFs of 77–510 pages) with
**Document AI + Gemini**, then answers free-form questions about the filings with page-cited RAG.
Every step is scored against an answer key the pipeline never sees: the SEC's own **XBRL** data
for the same filing.

The point of the project is the measurement, not just the pipeline: a dumb baseline for every
component, a dev set to tune on, and held-out sets scored **once** with the pipeline frozen.

## Results

**Extraction** — 9 fields (revenue, net income, EPS, total assets, equity, cash, operating
income, ...) per filing, scored against XBRL.

| Set (20 companies each) | Pipeline | Keyword baseline |
|---|---|---|
| Dev (tuned on) | 98.2% | 55.2% |
| Holdout, frozen v1, scored once | 87.0% | 56.8% |
| Third set, frozen v2, scored once | **87.8%** | 62.2% |

On the third set, non-banks scored 95.8% (137/143); banks scored 7/21. Bank filings are
effectively a different document type, and the main remaining gap.

**Question answering** — 434 questions auto-labeled from XBRL facts *outside* the 9 extracted
fields; 271 of them (holdout + third companies) are the test set. Retrieval over 25,552
page-bounded chunks.

| Retrieval mode (test) | hit@5 | MRR@10 |
|---|---|---|
| BM25 (baseline) | 39.9% | 0.292 |
| Dense (text-embedding-005, BigQuery `VECTOR_SEARCH`) | 80.8% | 0.610 |
| Dense + Vertex AI Ranking API | 83.8% | 0.710 |
| Hybrid (BM25 + dense) + ranker | 84.9% | 0.706 |
| **LlamaIndex, default chunking + ranker** | **93.7%** | **0.761** |

| Answers (test, 271 questions) | Correct | Wrong | Abstained |
|---|---|---|---|
| Hand-built dense + ranker | 87.1% | 21 | 14 |
| **LlamaIndex + ranker** | **93.0%** | 10 | 7 |

Why LlamaIndex won: `li_ours` (my chunks loaded into LlamaIndex) scores *exactly* the same as the
hand-built pipeline, so the gain is LlamaIndex's larger, roughly page-sized default chunks, not
the framework itself.

Total cloud cost for the whole project: about $10. About $0.003 per question answered.

## How it works

```
10-K HTML ──headless Chrome──▶ PDF          XBRL companyfacts ──▶ answer key (scoring only)
                                │
              route.py: find statement pages for free (pypdf text layer)
                                │  112 of 3,741 pages, ~97% less Document AI spend
              parse.py: Document AI Layout Parser on those pages only
                                │
              extract.py: Gemini copies the printed value, scale, page and row;
                          code converts units and runs the checks
                                │
              score.py: compare to XBRL, plus a keyword baseline
```

The LLM only reads and copies. Everything that can be checked deterministically is done in code:

- **Grounding** — the value must appear on the page Gemini cited.
- **Accounting identities** — assets = liabilities + equity (within 3%), EPS sign matches net income.
- **Column check** — the value's column is re-read from the PDF text layer. Layout Parser can
  merge adjacent year columns and scramble their order; Gemini then reports the year it was asked
  for while copying the wrong one. This was the biggest single error on the holdout set, and it
  passes every consistency check because all the numbers share the wrong column.
- **Scale** (thousands vs millions) is read from the page, not guessed.
- **Noncontrolling interest** is detected by line order so totals that include it are not taken
  as the parent's figure.

A failed check routes the value to a review queue instead of returning it.

**RAG** (`index.py`, `rag.py`, `llamaindex_rag.py`): chunks never cross a page, so every answer
cites exactly one page. Gemini must answer with the page number and a verbatim quote; code
verifies the quote is on that page and contains the number, or the answer is flagged.

## What the checks still miss

- **Entity errors.** For Citi, the router picked "Condensed Parent Company Only" statements: the
  wrong entity, but internally consistent numbers, so no consistency check fires.
- **Neighbouring concepts.** Most wrong RAG answers are real quotes for the *adjacent* concept
  (dividends declared vs paid, buybacks from the equity statement vs the cash-flow statement).
  The citation check verifies the quote exists, not that it is the right concept.
- Checks only catch what they were designed for. Both of these need a new check and a fresh,
  unseen set to score it on; the holdout and third sets are now spent.

## Files

| File | Step |
|---|---|
| `corpus.py` | Download 10-Ks for 60 companies, print to PDF, build the XBRL answer key |
| `route.py` | Find the financial-statement pages with no API calls |
| `parse.py` | Document AI Layout Parser on routed pages (online, ≤6 pages per request) |
| `extract.py` | Gemini structured extraction + code checks (`--recheck` reruns checks with no model calls) |
| `score.py` | Score against XBRL and the keyword baseline |
| `questions.py` | Auto-label questions from XBRL facts; relevant pages = pages printing the number |
| `index.py` | Chunk, embed, load to BigQuery |
| `rag.py`, `rag_eval.py` | Retrieval modes, cited answers, evaluation |
| `llamaindex_rag.py` | The same Q&A rebuilt in LlamaIndex, scored on the same questions |
| `app.py`, `web/` | Local web page: extracted vs official numbers, ask a question, open the cited PDF page |
| `results/` | Every score and report quoted above |

## Running it

Python 3.11+, a Google Cloud project with Document AI, Vertex AI and BigQuery enabled, and
`gcloud auth application-default login`.

```
pip install -r requirements.txt
set GCP_PROJECT=your-project-id          (PowerShell: $env:GCP_PROJECT="your-project-id")
python corpus.py          # downloads filings into data/ (about 1.6 GB, not in the repo)
python route.py
python parse.py
python extract.py
python score.py third     # dev | holdout | third
python questions.py
python index.py
python rag_eval.py retrieval
python rag_eval.py answers dense_rank
python llamaindex_rag.py build
python llamaindex_rag.py retrieval
python llamaindex_rag.py answers li_rank
uvicorn app:app --port 8080
```

`corpus.py` expects Chrome at the default Windows path; change `CHROME` at the top for other
systems. SEC requests use a generic User-Agent; set your own contact address there for heavier use.

## Tech

Python · Google Document AI (Layout Parser) · Gemini 2.5 Flash · Vertex AI text embeddings ·
Vertex AI Ranking API · BigQuery vector search · LlamaIndex · FastAPI · pypdf · rank-bm25

Built with [Claude Code](https://claude.com/claude-code) as a coding partner. MIT licensed.
