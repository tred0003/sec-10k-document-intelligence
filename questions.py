"""Step 5b: an auto-labeled question set for Q&A over the filings.

Each question comes from an XBRL fact outside the nine extracted fields (R&D,
buybacks, dividends, capex, goodwill, cover-page shares outstanding...), so
the official value is the answer. The relevant pages are every page where
that number is printed, found by searching the page text. No hand labels.

A question is dropped when its number is printed nowhere (nothing to retrieve)
or is too short to search for safely (e.g. "45" would match everywhere).

Run:  python questions.py
Writes results/questions.json
"""
import json
import re
from pathlib import Path

from corpus import HOLDOUT, THIRD, TICKERS, get

DATA = Path(__file__).parent / "data"
RESULTS = Path(__file__).parent / "results"

# concept(s), kind, question template. {co} = company, {fy} = fiscal year end date.
CONCEPTS = [
    (["ResearchAndDevelopmentExpense"], "duration",
     "How much did {co} spend on research and development in the fiscal year ended {fy}?"),
    (["IncomeTaxExpenseBenefit"], "duration",
     "What was {co}'s provision for income taxes in the fiscal year ended {fy}?"),
    (["ShareBasedCompensation", "AllocatedShareBasedCompensationExpense"], "duration",
     "How much stock-based compensation expense did {co} record in the fiscal year ended {fy}?"),
    (["PaymentsForRepurchaseOfCommonStock"], "duration",
     "How much did {co} spend repurchasing its own common stock in the fiscal year ended {fy}?"),
    (["PaymentsOfDividends", "PaymentsOfDividendsCommonStock"], "duration",
     "How much did {co} pay in dividends in the fiscal year ended {fy}?"),
    (["PaymentsToAcquirePropertyPlantAndEquipment"], "duration",
     "What were {co}'s capital expenditures (purchases of property and equipment) in the fiscal year ended {fy}?"),
    (["DepreciationDepletionAndAmortization", "DepreciationAndAmortization",
      "DepreciationAmortizationAndAccretionNet"], "duration",
     "What was {co}'s depreciation and amortization in the fiscal year ended {fy}?"),
    (["InterestExpense", "InterestExpenseNonoperating"], "duration",
     "What was {co}'s interest expense in the fiscal year ended {fy}?"),
    (["Goodwill"], "instant",
     "How much goodwill did {co} report on its balance sheet as of {fy}?"),
    (["InventoryNet"], "instant",
     "What was the value of {co}'s inventory as of {fy}?"),
    (["LongTermDebtNoncurrent"], "instant",
     "How much long-term debt (excluding the current portion) did {co} have as of {fy}?"),
    (["dei:EntityCommonStockSharesOutstanding"], "cover",
     "How many shares of common stock did {co} report as outstanding on the cover of its 10-K?"),
]


def facts(ticker, cik):
    cache = DATA / ticker / "facts.json"
    if not cache.exists():
        cache.write_text(get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json").text)
    return json.loads(cache.read_text())


def find_value(all_facts, concepts, kind, meta):
    from datetime import date
    end = meta["period_end"]
    for name in concepts:
        ns, _, concept = name.rpartition(":")
        rows = [r for unit in all_facts["facts"].get(ns or "us-gaap", {}).get(concept, {})
                .get("units", {}).values() for r in unit if r["accn"] == meta["accn"]]
        if kind == "cover":
            values = {r["val"] for r in rows}
            if len(values) == 1:  # several share classes = no single answer
                return values.pop(), concept
            continue
        for r in rows:
            if r["end"] != end:
                continue
            if kind == "duration":
                if "start" not in r:
                    continue
                days = (date.fromisoformat(end) - date.fromisoformat(r["start"])).days
                if not 350 <= days <= 380:
                    continue
            elif "start" in r:
                continue
            return r["val"], concept
    return None, None


def printed_forms(value):
    """How the number could appear on a page: in millions, in thousands, or in full."""
    v = abs(value)
    forms = set()
    for div in (1e6, 1e3, 1):
        x = v / div
        if abs(x - round(x)) < 1e-9 and x >= 1000:  # exact at that scale, and long enough to search
            forms.add(f"{round(x):,}")
    return forms


def pages_with(forms, pages):
    hits = []
    for page_no, text in enumerate(pages, 1):
        for f in forms:
            if re.search(rf"(?<![\d,.]){re.escape(f)}(?![\d,]|\.\d)", text):
                hits.append(page_no)
                break
    return hits


def main():
    RESULTS.mkdir(exist_ok=True)
    split = {**{t: "dev" for t in TICKERS}, **{t: "test" for t in HOLDOUT + THIRD}}
    out, dropped = [], 0
    for t, which in split.items():
        meta = json.loads((DATA / t / "meta.json").read_text())
        f = facts(t, meta["cik"])
        pages = json.loads((DATA / t / "pages.json").read_text())
        company = f["entityName"]
        for concepts, kind, template in CONCEPTS:
            value, concept = find_value(f, concepts, kind, meta)
            if value is None:
                continue
            forms = printed_forms(value)
            hits = pages_with(forms, pages) if forms else []
            if not hits:
                dropped += 1
                continue
            out.append({"id": f"{t}-{concept}", "split": which, "ticker": t, "company": company,
                        "concept": concept, "question": template.format(co=company, fy=meta["period_end"]),
                        "answer": value, "printed_forms": sorted(forms), "relevant_pages": hits})
    (RESULTS / "questions.json").write_text(json.dumps(out, indent=2))
    n = {s: sum(q["split"] == s for q in out) for s in ("dev", "test")}
    print(f"{len(out)} questions ({n['dev']} dev, {n['test']} test), {dropped} dropped (number not printed)")
    by = {}
    for q in out:
        by[q["concept"]] = by.get(q["concept"], 0) + 1
    for c, k in sorted(by.items(), key=lambda x: -x[1]):
        print(f"  {k:3} {c}")
    pages_per = sorted(len(q["relevant_pages"]) for q in out)
    print(f"relevant pages per question: median {pages_per[len(pages_per) // 2]}, max {pages_per[-1]}")


if __name__ == "__main__":
    main()
