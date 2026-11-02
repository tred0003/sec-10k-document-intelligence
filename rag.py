"""Step 5c: retrieve from one company's filing, answer with a page citation, verify it.

Retrieval modes (compared in rag_eval.py):
  bm25        keyword scoring over the filing's chunks (the dumb baseline)
  dense       BigQuery VECTOR_SEARCH on text-embedding-005 vectors
  dense_rank  dense top 50, re-ordered by the Vertex AI Ranking API
  hybrid_rank bm25 top 50 + dense top 50 merged (reciprocal rank fusion), then ranked

The answer step mirrors extraction: Gemini copies the printed number, the
page and a verbatim quote; code checks the quote is on that page and holds
the number, then does the unit math. A failed check is reported, not hidden.

Run:  python rag.py AAPL "How much did Apple spend on R&D in fiscal 2025?"
"""
import json
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

import google.auth
import google.auth.transport.requests
import requests
from google import genai
from google.cloud import bigquery
from google.genai import types
from pydantic import BaseModel
from rank_bm25 import BM25Okapi

from extract import SCALE, to_number
from index import EMBED_MODEL, PROJECT, TABLE

MODEL = "gemini-2.5-flash"
RANK_MODEL = "semantic-ranker-default-005"   # same ranker as the nfl-dfs news search
FETCH = 50
DATA = Path(__file__).parent / "data"

genai_client = genai.Client(vertexai=True, project=PROJECT, location="us-central1")
bq = bigquery.Client(project=PROJECT)

SEARCH_SQL = f"""
SELECT base.chunk_id, base.page, base.text, distance
FROM VECTOR_SEARCH(
  (SELECT * FROM `{TABLE}` WHERE ticker = @ticker),
  'embedding', (SELECT @q AS embedding),
  top_k => @k, distance_type => 'COSINE')
ORDER BY distance
"""


@lru_cache(maxsize=None)
def chunks(ticker):
    return json.loads((DATA / ticker / "chunks.json").read_text())


def tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower())


@lru_cache(maxsize=None)
def bm25_index(ticker):
    return BM25Okapi([tokens(c["text"]) for c in chunks(ticker)])


def bm25(question, ticker, k=FETCH):
    scores = bm25_index(ticker).get_scores(tokens(question))
    top = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
    return [chunks(ticker)[i] for i in top]


def dense(question, ticker, k=FETCH):
    vec = genai_client.models.embed_content(
        model=EMBED_MODEL, contents=[question],
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")).embeddings[0].values
    job = bq.query(SEARCH_SQL, job_config=bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter("ticker", "STRING", ticker),
        bigquery.ScalarQueryParameter("k", "INT64", k),
        bigquery.ArrayQueryParameter("q", "FLOAT64", list(vec))]))
    return [dict(r) for r in job.result(timeout=60)]


class Ranker:
    """Vertex AI Ranking API (Discovery Engine), as in nfl-dfs gameday_agent/archive.py."""

    def __init__(self):
        self.creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        self.req = google.auth.transport.requests.Request()
        self.url = (f"https://discoveryengine.googleapis.com/v1/projects/{PROJECT}/"
                    "locations/global/rankingConfigs/default_ranking_config:rank")

    def order(self, question, rows):
        if not self.creds.valid:
            self.creds.refresh(self.req)
        r = requests.post(self.url, timeout=30, json={
            "model": RANK_MODEL, "query": question, "topN": len(rows),
            "ignoreRecordDetailsInResponse": True,
            "records": [{"id": str(i), "title": f"10-K page {row['page']}", "content": row["text"]}
                        for i, row in enumerate(rows)]},
            headers={"Authorization": f"Bearer {self.creds.token}", "x-goog-user-project": PROJECT})
        r.raise_for_status()
        return [rows[int(rec["id"])] for rec in r.json()["records"]]


ranker = Ranker()


def fuse(*lists, k=60):
    """Reciprocal rank fusion: a chunk near the top of either list rises."""
    score, by_id = {}, {}
    for rows in lists:
        for rank, row in enumerate(rows):
            score[row["chunk_id"]] = score.get(row["chunk_id"], 0) + 1 / (k + rank + 1)
            by_id[row["chunk_id"]] = row
    return [by_id[c] for c in sorted(score, key=lambda c: -score[c])]


def search(question, ticker, mode="hybrid_rank"):
    if mode == "bm25":
        return bm25(question, ticker)
    if mode == "dense":
        return dense(question, ticker)
    if mode == "dense_rank":
        return ranker.order(question, dense(question, ticker))
    if mode == "hybrid_rank":
        return ranker.order(question, fuse(bm25(question, ticker), dense(question, ticker))[:FETCH])
    raise ValueError(mode)


class Answer(BaseModel):
    found: bool
    printed: Optional[str]      # the number exactly as printed in the quote
    scale: Literal["units", "thousands", "millions", "billions"]
    page: Optional[int]
    quote: Optional[str]        # verbatim text from the cited excerpt that contains the number


PROMPT = """Answer the question using ONLY the excerpts from {company}'s 10-K below.
Each excerpt starts with its page number.

Return the number exactly as printed (keep commas and parentheses, drop $), its scale
(from a nearby "in millions"/"in thousands" note or the words around it, e.g. "$4.2 billion"
-> printed "4.2", scale billions; a share count printed in full -> units), the page, and a short
verbatim quote from that excerpt containing the number. If the excerpts do not contain the
answer for the fiscal year asked about, set found=false.

Question: {question}

{excerpts}"""


def answer(question, ticker, company, mode="hybrid_rank", k=8):
    return answer_from(question, company, search(question, ticker, mode)[:k])


def answer_from(question, company, hits):
    """The answer step on chunks already retrieved (the web page shows them too)."""
    excerpts = "\n\n".join(f"[page {h['page']}]\n{h['text']}" for h in hits)
    resp = genai_client.models.generate_content(
        model=MODEL, contents=PROMPT.format(company=company, question=question, excerpts=excerpts),
        config=types.GenerateContentConfig(temperature=0, response_mime_type="application/json",
                                           response_schema=Answer))
    a = Answer.model_validate_json(resp.text).model_dump()

    checks = []
    if a["found"] and a["printed"]:
        number = to_number(a["printed"])
        a["value"] = None if number is None else number * SCALE[a["scale"]]
        page_text = " ".join(" ".join(h["text"] for h in hits if h["page"] == a["page"]).split())
        quote = " ".join((a["quote"] or "").split())
        if not quote or quote not in page_text:
            checks.append("quote not found on the cited page")
        if a["printed"].strip("()$ ") not in quote:
            checks.append("number not in the quote")
    else:
        a["value"] = None
    a["checks"] = checks
    a["retrieved_pages"] = [h["page"] for h in hits]
    return a


if __name__ == "__main__":
    t, q = sys.argv[1], sys.argv[2]
    company = json.loads((DATA / t / "facts.json").read_text())["entityName"]
    print(json.dumps(answer(q, t, company), indent=2))
