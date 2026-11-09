"""Step 5a: chunk every filing's full text, embed it, load it into BigQuery.

Chunks never cross a page, so every retrieved chunk has exactly one page to
cite. Text comes from the free pypdf layer (pages.json); Layout Parser was
only paid for on the statement pages.

Run:  python index.py          (embed whatever is not cached yet, then load BigQuery)
Writes data/<T>/chunks.json + data/<T>/embeddings.npy, table sec_docintel.chunks
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from google import genai
from google.cloud import bigquery
from google.genai import types

from corpus import HOLDOUT, THIRD, TICKERS

PROJECT = os.environ.get("GCP_PROJECT", "advisor-agent-2026")
DATASET = "sec_docintel"
TABLE = f"{PROJECT}.{DATASET}.chunks"
EMBED_MODEL = "text-embedding-005"   # same model the nfl-dfs news archive uses
CHUNK_CHARS = 1500
OVERLAP = 200
BATCH = 25                           # stays under the 20k-token request limit
DATA = Path(__file__).parent / "data"


def chunk_pages(ticker):
    pages = json.loads((DATA / ticker / "pages.json").read_text())
    meta = json.loads((DATA / ticker / "meta.json").read_text())
    chunks = []
    for page_no, text in enumerate(pages, 1):
        text = " ".join(text.split())  # pypdf line breaks carry no meaning here
        if len(text) < 80:
            continue  # page numbers, blank separator pages
        start = 0
        while start < len(text):
            piece = text[start:start + CHUNK_CHARS]
            chunks.append({
                "chunk_id": f"{ticker}-p{page_no}-{start}",
                "ticker": ticker, "page": page_no, "period_end": meta["period_end"],
                "text": piece,
            })
            if start + CHUNK_CHARS >= len(text):
                break
            start += CHUNK_CHARS - OVERLAP
    return chunks


def embed(client, texts):
    vecs = []
    for i in range(0, len(texts), BATCH):
        for attempt in range(6):
            try:
                resp = client.models.embed_content(
                    model=EMBED_MODEL, contents=texts[i:i + BATCH],
                    config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"))
                vecs += [e.values for e in resp.embeddings]
                break
            except Exception as e:  # 429s under load; back off and retry
                if attempt == 5:
                    raise
                time.sleep(2 ** attempt)
                print(f"  retry {attempt + 1}: {str(e)[:80]}")
    return np.array(vecs, dtype=np.float32)


def load_bigquery(tickers):
    bq = bigquery.Client(project=PROJECT)
    bq.create_dataset(f"{PROJECT}.{DATASET}", exists_ok=True)
    # Streamed to a newline-delimited JSON file one filing at a time, so the
    # 25k vectors are never all held in memory as Python lists.
    ndjson = DATA / "chunks_load.ndjson"
    rows = 0
    with open(ndjson, "w", encoding="utf-8") as fh:
        for t in tickers:
            chunks = json.loads((DATA / t / "chunks.json").read_text())
            vecs = np.load(DATA / t / "embeddings.npy")
            for c, v in zip(chunks, vecs):
                fh.write(json.dumps({**c, "embedding": [round(float(x), 6) for x in v]}) + "\n")
                rows += 1
    schema = [
        bigquery.SchemaField("chunk_id", "STRING"), bigquery.SchemaField("ticker", "STRING"),
        bigquery.SchemaField("page", "INT64"), bigquery.SchemaField("period_end", "DATE"),
        bigquery.SchemaField("text", "STRING"),
        bigquery.SchemaField("embedding", "FLOAT64", mode="REPEATED"),
    ]
    with open(ndjson, "rb") as fh:
        job = bq.load_table_from_file(fh, TABLE, job_config=bigquery.LoadJobConfig(
            schema=schema, write_disposition="WRITE_TRUNCATE",
            # Every search filters to one company; clustering lets BigQuery read only
            # that company's blocks (~3 MB) instead of the whole table (~190 MB).
            clustering_fields=["ticker"],
            source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON))
        job.result()
    print(f"loaded {rows} chunks into {TABLE}")


def main():
    client = genai.Client(vertexai=True, project=PROJECT, location="us-central1")
    tickers = sys.argv[1:] or TICKERS + HOLDOUT + THIRD
    total = 0
    for t in tickers:
        if (DATA / t / "embeddings.npy").exists():
            total += len(json.loads((DATA / t / "chunks.json").read_text()))
            continue
        chunks = chunk_pages(t)
        vecs = embed(client, [c["text"] for c in chunks])
        (DATA / t / "chunks.json").write_text(json.dumps(chunks))
        np.save(DATA / t / "embeddings.npy", vecs)
        total += len(chunks)
        print(f"{t}: {len(chunks)} chunks")
    print(f"{total} chunks embedded")
    load_bigquery(TICKERS + HOLDOUT + THIRD)


if __name__ == "__main__":
    main()
