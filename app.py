"""A web page for trying the pipeline by hand: pick a company, see the nine
extracted numbers next to the official XBRL values, ask any question about the
filing and open the cited page of the PDF.

Run:  uvicorn app:app --port 8080      then open http://localhost:8080
"""
import json
import time
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

import llamaindex_rag
import rag
from corpus import HOLDOUT, THIRD, TICKERS
from extract import FIELDS
from rag_eval import correct, rounded_ok
from score import matches

DATA = Path(__file__).parent / "data"
RESULTS = Path(__file__).parent / "results"
SETS = {**{t: "dev" for t in TICKERS}, **{t: "holdout" for t in HOLDOUT}, **{t: "third" for t in THIRD}}
MODES = ["li_rank", "dense_rank", "li_default", "hybrid_rank", "dense", "bm25"]

app = FastAPI(title="SEC doc-intel")


def load(ticker, name):
    path = DATA / ticker / name
    return json.loads(path.read_text()) if path.exists() else None


@lru_cache(maxsize=None)
def company_name(ticker):
    facts = load(ticker, "facts.json")
    return facts["entityName"] if facts else ticker


@lru_cache(maxsize=1)
def questions():
    by = {}
    for q in json.loads((RESULTS / "questions.json").read_text()):
        by.setdefault(q["ticker"], []).append(q)
    return by


def check_ticker(ticker):
    if ticker not in SETS:
        raise HTTPException(404, f"unknown ticker {ticker}")


@app.get("/")
def page():
    return FileResponse(Path(__file__).parent / "web" / "index.html")


@app.get("/api/companies")
def companies():
    out = []
    for t, which in SETS.items():
        meta = load(t, "meta.json") or {}
        out.append({"ticker": t, "name": company_name(t), "set": which,
                    "period_end": meta.get("period_end"), "pages": meta.get("pages")})
    return out


@app.get("/api/company/{ticker}")
def company(ticker: str):
    check_ticker(ticker)
    meta = load(ticker, "meta.json")
    key = load(ticker, "answer_key.json") or {}
    got = (load(ticker, "extracted.json") or {}).get("fields", {})
    route = load(ticker, "route.json") or {}
    rows = []
    for f in FIELDS:
        e = got.get(f) or {}
        official = key.get(f, {}).get("value")
        value = e.get("value")
        if official is None:
            outcome = "no official value"
        elif value is None:
            outcome = "missing"
        else:
            outcome = "correct" if matches(f, value, official) else "wrong"
        rows.append({"field": f, "value": value, "printed": e.get("printed"), "scale": e.get("scale"),
                     "page": e.get("page"), "row_label": e.get("row_label"),
                     "column_header": e.get("column_header"), "flags": e.get("flags", []),
                     "official": official, "outcome": outcome})
    labeled = [{"question": q["question"], "answer": q["answer"]} for q in questions().get(ticker, [])]
    return {"ticker": ticker, "name": company_name(ticker), "set": SETS[ticker], "meta": meta,
            "routed_pages": route.get("pages", []), "fields": rows, "questions": labeled}


class Ask(BaseModel):
    ticker: str
    question: str
    mode: str = "li_rank"      # best on the dev questions (LlamaIndex, page-sized chunks + ranker)
    answer: bool = True        # false = retrieval only, no Gemini call


@app.post("/api/ask")
def ask(req: Ask):
    check_ticker(req.ticker)
    if req.mode not in MODES:
        raise HTTPException(400, f"mode must be one of {MODES}")
    t0 = time.time()
    li = req.mode.startswith("li_")
    hits = (llamaindex_rag.search if li else rag.search)(req.question, req.ticker, req.mode)[:8]
    t1 = time.time()
    out = {"hits": [{"page": h["page"], "text": h["text"]} for h in hits],
           "seconds": {"retrieve": round(t1 - t0, 2)}}
    if req.answer:
        out["answer"] = (llamaindex_rag.answer_from if li else rag.answer_from)(
            req.question, company_name(req.ticker), hits)
        out["seconds"]["answer"] = round(time.time() - t1, 2)
    # A question from the labeled set gets graded against its official value.
    known = next((q for q in questions().get(req.ticker, []) if q["question"] == req.question.strip()), None)
    if known:
        out["official"] = known["answer"]
        out["relevant_pages"] = known["relevant_pages"]
        a = out.get("answer")
        if a:  # same buckets as rag_eval.py
            out["grade"] = ("abstained" if a["value"] is None else "correct" if correct(a["value"], known["answer"])
                            else "right at printed precision" if rounded_ok(a, known["answer"]) else "wrong")
    return out


@app.get("/pdf/{ticker}")
def pdf(ticker: str):
    check_ticker(ticker)
    return FileResponse(DATA / ticker / "10k.pdf", media_type="application/pdf",
                        headers={"Content-Disposition": f'inline; filename="{ticker}-10k.pdf"'})
