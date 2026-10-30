"""Step 2a: find the financial-statement pages for free, before paying for parsing.

The Chrome-printed PDFs carry a text layer, so pypdf can read every page at no
cost. A page counts as a primary statement when a statement title sits on one
of its first few lines and the page is dense with numbers; mentions of
"consolidated balance sheets" inside the notes are mid-paragraph and fail the
first test. Only the routed pages go to Layout Parser ($10 / 1,000 pages).

Run:  python route.py
Writes data/<T>/pages.json (all page text) and data/<T>/route.json.
"""
import itertools
import json
import re
from pathlib import Path

from pypdf import PdfReader

DATA = Path(__file__).parent / "data"

# Titles vary a lot: "Consolidated Statements of Operations", "Income Statements",
# "Consolidated Financial Position", "Cash Flows Statements", "Statement of Cash Flow".
TITLES = {
    "income": r"(statements? of (consolidated )?(operations|income|earnings)|results of operations"
              r"|income statements?)",
    "balance": r"(balance sheets?|financial (condition|position))",
    "cash_flow": r"(statements? of (consolidated )?cash flows?|cash flows? statements?)",
}
TITLE_RE = {k: re.compile(v, re.I) for k, v in TITLES.items()}
# Lines that contain a title phrase but head a different table. Comprehensive income
# is excluded only as the lead ("Statement of Comprehensive Income"), not as a
# combined statement ("Statements of Operations and Comprehensive Income").
NOT_A_TITLE = re.compile(r"(\bof|^)\s*(consolidated\s+)?comprehensive|changes in|equity|analysis"
                         r"|notes to|discussion|index", re.I)
HEAD_LINES = 7     # titles sit under page headers like "PART II / Item 8 / 50"
MAX_TITLE_LEN = 110
MIN_DIGITS = 150   # statement pages are mostly numbers; narrative pages are not


def page_texts(ticker):
    cache = DATA / ticker / "pages.json"
    if cache.exists():
        return json.loads(cache.read_text())
    pages = [p.extract_text() or "" for p in PdfReader(DATA / ticker / "10k.pdf").pages]
    cache.write_text(json.dumps(pages))
    return pages


def digits(text):
    return sum(c.isdigit() for c in text)


def statement_kind(text, next_text=""):
    # A page break can leave the title alone on its page with the table on the next.
    numbers = digits(text) if digits(text) >= 50 else digits(text) + digits(next_text)
    if numbers < MIN_DIGITS:
        return None
    for line in [l.strip() for l in text.splitlines() if l.strip()][:HEAD_LINES]:
        if len(line) > MAX_TITLE_LEN or NOT_A_TITLE.search(line):
            continue
        for kind, rx in TITLE_RE.items():
            if rx.search(line):
                return kind
    return None


def tightest_cluster(candidates):
    """The three primary statements sit together in Item 8. MD&A tables and notes
    reuse the titles, so pick one page per kind with the smallest overall spread."""
    kinds = [k for k in TITLES if candidates.get(k)]
    if not kinds:
        return {}
    best = min(itertools.product(*(candidates[k] for k in kinds)),
               key=lambda combo: (max(combo) - min(combo), min(combo)))
    return dict(zip(kinds, best))


def route(ticker):
    pages = page_texts(ticker)
    candidates = {}
    for i, text in enumerate(pages):
        kind = statement_kind(text, pages[i + 1] if i + 1 < len(pages) else "")
        if kind:
            candidates.setdefault(kind, []).append(i)
    found = tightest_cluster(candidates)
    # The cluster can land on a "(Continued)" page; back up to where the statement starts.
    for kind, i in found.items():
        while i - 1 in candidates.get(kind, []):
            i -= 1
        found[kind] = i
    # Statements often run onto a second page, so send the next page too.
    selected = sorted({p for i in found.values() for p in (i, i + 1) if p < len(pages)})
    result = {"statements": {k: v + 1 for k, v in found.items()},  # 1-based for humans
              "pages": [p + 1 for p in selected], "total_pages": len(pages)}
    (DATA / ticker / "route.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    total = sent = 0
    for folder in sorted(p for p in DATA.iterdir() if (p / "10k.pdf").exists()):
        r = route(folder.name)
        total += r["total_pages"]
        sent += len(r["pages"])
        missing = [k for k in TITLES if k not in r["statements"]]
        print(f"{folder.name:5} {r['statements']}  -> {len(r['pages'])} pages"
              + (f"  MISSING {missing}" if missing else ""))
    print(f"\nrouted {sent} of {total} pages ({sent / total:.1%}), "
          f"Layout Parser cost ~${sent * 10 / 1000:.2f} vs ${total * 10 / 1000:.2f} full")


if __name__ == "__main__":
    main()
