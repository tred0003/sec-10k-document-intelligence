"""Step 4: score the pipeline against the XBRL answer key, next to a dumb baseline.

Baseline: no Document AI, no Gemini. Free pypdf text of the same routed pages,
find the row by keyword, take the first number on that line, scale from the
"(In millions)" note on the page. If the pipeline cannot beat this, it is not
worth what it costs.

Outcomes per (company, field) in the answer key:
  correct  - within rounding of the official value
  wrong    - a value that does not match
  missing  - the answer key has it, the method returned nothing

Run:  python score.py            (dev set: the 20 companies the pipeline was built on)
      python score.py holdout    (20 companies held out from v1; used to build v2)
      python score.py third      (20 companies never looked at while building v2)
Writes results/score_<set>.json
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from corpus import HOLDOUT, THIRD, TICKERS
from extract import FIELDS, to_number

DATA = Path(__file__).parent / "data"
RESULTS = Path(__file__).parent / "results"

BASELINE_ROWS = {
    "revenue": r"^(total )?(net )?(revenues?|net sales|sales)\b",
    "operating_income": r"^(total )?(operating income|income from operations|operating profit)",
    "net_income": r"^net (income|earnings)\b",
    "eps_diluted": r"^diluted\b",
    "operating_cash_flow": r"^net cash (provided by|from|used in|\(used in\)|provided by \(used in\)).*operating",
    "total_assets": r"^total assets\b",
    "total_liabilities": r"^total liabilities\s*[\d$(]",
    "stockholders_equity": r"^total (stockholders|shareholders)",
    "cash": r"^cash and cash equivalents\b",
}
NUMBER = re.compile(r"\(?-?\$?\s?\d[\d,]*(\.\d+)?\)?")


def baseline(ticker):
    route = json.loads((DATA / ticker / "route.json").read_text())
    pages = json.loads((DATA / ticker / "pages.json").read_text())
    out = {}
    for f, pattern in BASELINE_ROWS.items():
        rx = re.compile(pattern, re.I)
        for p in route["pages"]:
            text = pages[p - 1]
            low = text.lower()
            scale = 1e3 if "in thousands" in low else 1e9 if "in billions" in low else 1e6
            line = next((l.strip() for l in text.splitlines() if rx.search(l.strip())), None)
            if not line:
                continue
            m = NUMBER.search(line[len(rx.search(line).group(0)):])
            n = to_number(m.group(0)) if m else None
            if n is not None:
                out[f] = n * (1 if f == "eps_diluted" else scale)
                break
    return out


def matches(field, got, want):
    if field == "eps_diluted":
        return abs(got - want) <= 0.005
    return abs(got - want) <= max(0.001 * abs(want), 1)  # printed values are rounded to the scale


def main():
    RESULTS.mkdir(exist_ok=True)
    counts = {"pipeline": Counter(), "baseline": Counter()}
    by_field = {"pipeline": defaultdict(Counter), "baseline": defaultdict(Counter)}
    review = Counter()       # pipeline outcome x flagged / auto-accepted
    errors, unkeyed = [], []

    which = sys.argv[1] if len(sys.argv) > 1 else "dev"
    tickers = {"holdout": HOLDOUT, "third": THIRD}.get(which, TICKERS)
    for t in sorted(t for t in tickers if (DATA / t / "answer_key.json").exists()):
        folder = DATA / t
        key = json.loads((folder / "answer_key.json").read_text())
        # A filing that failed upstream (nothing routed or parsed) scores as all missing, not skipped.
        empty = {f: {"value": None, "flags": [], "row_label": None, "page": None} for f in FIELDS}
        ext_path = folder / "extracted.json"
        ext = json.loads(ext_path.read_text())["fields"] if ext_path.exists() else empty
        base = baseline(t)

        for f in FIELDS:
            if f not in key:
                if ext[f]["value"] is not None:
                    unkeyed.append({"ticker": t, "field": f, "value": ext[f]["value"],
                                    "row_label": ext[f]["row_label"], "page": ext[f]["page"]})
                continue
            want = key[f]["value"]
            for method, got in (("pipeline", ext[f]["value"]), ("baseline", base.get(f))):
                outcome = "missing" if got is None else "correct" if matches(f, got, want) else "wrong"
                counts[method][outcome] += 1
                by_field[method][f][outcome] += 1
                if method == "pipeline":
                    review[(outcome, "flagged" if ext[f]["flags"] else "auto")] += 1
                    if outcome != "correct":
                        errors.append({"ticker": t, "field": f, "outcome": outcome, "got": got,
                                       "want": want, "row_label": ext[f]["row_label"],
                                       "page": ext[f]["page"], "flags": ext[f]["flags"]})

    total = sum(counts["pipeline"].values())
    print(f"[{which}] {total} answer-key values\n")
    print(f"{'':10} {'correct':>8} {'wrong':>6} {'missing':>8}   accuracy")
    for m in ("baseline", "pipeline"):
        c = counts[m]
        print(f"{m:10} {c['correct']:8} {c['wrong']:6} {c['missing']:8}   {c['correct'] / total:.1%}")

    print(f"\n{'field':20} {'baseline':>9} {'pipeline':>9}")
    for f in FIELDS:
        b, p = by_field["baseline"][f], by_field["pipeline"][f]
        n = sum(p.values())
        print(f"{f:20} {b['correct']:>4}/{n:<4} {p['correct']:>4}/{n:<4}")

    auto = review[("correct", "auto")] + review[("wrong", "auto")]
    flagged = review[("correct", "flagged")] + review[("wrong", "flagged")]
    print(f"\nreview queue: {flagged} values flagged, {auto} auto-accepted")
    print(f"  wrong values caught by flags: {review[('wrong', 'flagged')]}"
          f" of {review[('wrong', 'flagged')] + review[('wrong', 'auto')]}")
    print(f"  auto-accepted accuracy: {review[('correct', 'auto')]}/{auto}")

    print("\npipeline misses:")
    for e in errors:
        print(f"  {e['ticker']:5} {e['field']:20} {e['outcome']:7} got={e['got']} want={e['want']}"
              f" row='{e['row_label']}' p{e['page']} {e['flags'] or ''}")
    print(f"\nextracted where the answer key has nothing (unscored): {len(unkeyed)}")
    for u in unkeyed:
        print(f"  {u['ticker']:5} {u['field']:20} {u['value']:,.0f}  row='{u['row_label']}'")

    (RESULTS / f"score_{which}.json").write_text(json.dumps({
        "counts": {m: dict(c) for m, c in counts.items()},
        "by_field": {m: {f: dict(c) for f, c in d.items()} for m, d in by_field.items()},
        "review": {f"{o}/{s}": n for (o, s), n in review.items()},
        "errors": errors, "unkeyed": unkeyed}, indent=2))


if __name__ == "__main__":
    main()
