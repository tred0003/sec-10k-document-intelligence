"""Step 3: Gemini reads the Layout Parser output and fills a fixed schema.

Division of labor: Gemini only copies what is printed (the number as text,
the scale from the table header, the page, the row label). Code turns that
into a value and runs the checks. Any field that fails a check is flagged
for human review instead of being trusted.

Run:  python extract.py            (all parsed companies)
      python extract.py AAPL --force
Writes data/<T>/extracted.json
"""
import json
import re
import sys
from pathlib import Path
from typing import Literal, Optional

from google import genai
from google.genai import types
from pydantic import BaseModel

from parse import walk

PROJECT = "advisor-agent-2026"
MODEL = "gemini-2.5-flash"
DATA = Path(__file__).parent / "data"
FIELDS = ["revenue", "operating_income", "net_income", "eps_diluted", "operating_cash_flow",
          "total_assets", "total_liabilities", "stockholders_equity", "cash"]
SCALE = {"units": 1, "thousands": 1e3, "millions": 1e6, "billions": 1e9}


class Field(BaseModel):
    printed: Optional[str]          # exactly as on the page, e.g. "416,161" or "(0.01)"; null if no such line
    scale: Literal["units", "thousands", "millions", "billions"]
    page: Optional[int]             # 10-K page number from the [page N] marker
    row_label: Optional[str]        # the row's label as printed


class Statements(BaseModel):
    revenue: Field
    operating_income: Field
    net_income: Field
    eps_diluted: Field
    operating_cash_flow: Field
    total_assets: Field
    total_liabilities: Field
    stockholders_equity: Field
    cash: Field


PROMPT = """You are reading the primary financial statements from a company's annual report (10-K).
The pages below were extracted by a layout parser; tables are shown as | separated rows and each
page starts with a [page N] marker giving its page number in the original 10-K.

For each field, copy the value for the MOST RECENT fiscal year exactly as printed
(keep commas, parentheses and minus signs; drop the $ sign). Also give the table's scale from its
header (e.g. "In millions" -> millions; per-share amounts are always "units"), the page marker
number, and the row label as printed.

Field definitions:
- revenue: the total revenue line (e.g. "Total net sales", "Total revenues", "Total net revenues").
- operating_income: the total operating income / income from operations line. Income before
  income taxes (pre-tax income) is NOT operating income; many banks and some industrials have no
  operating income line, and then the answer is null.
- net_income: net income attributable to the company (not including noncontrolling interests).
  Not "net income applicable/available to common stockholders", which subtracts preferred dividends.
- eps_diluted: diluted earnings per share for the company's common stock.
- operating_cash_flow: net cash provided by (used in) operating activities.
- total_assets, total_liabilities: the totals on the balance sheet.
- stockholders_equity: total stockholders'/shareholders' equity attributable to the company.
- cash: cash and cash equivalents on the balance sheet.

Copy only a number printed on a line that is that item. Do not add, subtract or estimate.
If the statements have no such line, return printed=null, page=null, row_label=null.

{pages}"""


def cell_text(cell):
    return " ".join(b.get("textBlock", {}).get("text", "") for b in cell.get("blocks", [])).strip()


def render(doc):
    """Layout blocks -> plain text with [page N] markers that use the original 10-K page numbers."""
    source_pages = doc["source_pages"]
    lines, current = [], None
    for b in walk(doc["documentLayout"]["blocks"]):
        page = source_pages[b["pageSpan"]["pageStart"] - 1]
        if page != current:
            lines.append(f"\n[page {page}]")
            current = page
        if "tableBlock" in b:
            t = b["tableBlock"]
            for row in t.get("headerRows", []) + t.get("bodyRows", []):
                lines.append("| " + " | ".join(cell_text(c) for c in row["cells"]) + " |")
        elif "textBlock" in b and not b["textBlock"].get("blocks"):
            lines.append(b["textBlock"].get("text", ""))
    return "\n".join(lines)


def to_number(printed):
    s = printed.replace("$", "").replace(",", "").replace(" ", "").replace("—", "").strip()
    neg = s.startswith("(") and s.endswith(")") or s.startswith("-") or s.startswith("−")
    s = s.strip("()-−")
    if not re.fullmatch(r"\d+(\.\d+)?", s):
        return None
    return -float(s) if neg else float(s)


def check(fields, page_text):
    """Code-side checks. Returns {field: [reasons]} for anything that should go to review."""
    flags = {f: [] for f in FIELDS}
    v = {f: d["value"] for f, d in fields.items() if d["value"] is not None}

    for f, d in fields.items():
        if d["printed"] is None:
            continue
        if d["value"] is None:
            flags[f].append("unreadable number")
            continue
        # Grounding: the quoted number must appear on the page Gemini cited.
        digits = re.sub(r"[^\d.]", "", d["printed"])
        page = re.sub(r"[^\d.\n ]", "", page_text.get(d["page"], ""))
        if digits not in page.replace(" ", " "):
            flags[f].append(f"printed value not found on page {d['page']}")
        if f != "eps_diluted" and abs(d["value"]) and abs(d["value"]) < 1e5:
            flags[f].append("implausibly small for a public company (scale?)")

    if all(k in v for k in ("total_assets", "total_liabilities", "stockholders_equity")):
        # Allow ~3% for noncontrolling interests and temporary equity between the two.
        gap = v["total_assets"] - v["total_liabilities"] - v["stockholders_equity"]
        if abs(gap) > 0.03 * v["total_assets"]:
            for f in ("total_assets", "total_liabilities", "stockholders_equity"):
                flags[f].append(f"assets != liabilities + equity (gap {gap:,.0f})")
    label = (fields["operating_income"]["row_label"] or "").lower()
    if "before" in label and "tax" in label:
        flags["operating_income"].append("row is pre-tax income, not operating income")
    if "net_income" in v and "eps_diluted" in v and v["net_income"] * v["eps_diluted"] < 0:
        for f in ("net_income", "eps_diluted"):
            flags[f].append("net income and EPS have opposite signs")
    if "revenue" in v and "net_income" in v and abs(v["net_income"]) > abs(v["revenue"]):
        flags["net_income"].append("net income larger than revenue")
    if "cash" in v and "total_assets" in v and v["cash"] > v["total_assets"]:
        flags["cash"].append("cash larger than total assets")
    return {f: r for f, r in flags.items() if r}


def extract(ticker, client, force=False):
    out = DATA / ticker / "extracted.json"
    if out.exists() and not force:
        return json.loads(out.read_text())
    doc = json.loads((DATA / ticker / "layout.json").read_text())
    resp = client.models.generate_content(
        model=MODEL,
        contents=PROMPT.format(pages=render(doc)),
        config=types.GenerateContentConfig(
            temperature=0, response_mime_type="application/json", response_schema=Statements),
    )
    raw = Statements.model_validate_json(resp.text).model_dump()

    fields = {}
    for f in FIELDS:
        d = raw[f]
        number = to_number(d["printed"]) if d["printed"] else None
        value = None if number is None else number * (1 if f == "eps_diluted" else SCALE[d["scale"]])
        fields[f] = {**d, "value": value}

    # Grounding uses the full page text from pypdf, independent of the layout parse.
    pages = json.loads((DATA / ticker / "pages.json").read_text())
    page_text = {p: pages[p - 1] for p in doc["source_pages"]}
    flags = check(fields, page_text)
    for f in FIELDS:
        fields[f]["flags"] = flags.get(f, [])

    usage = resp.usage_metadata
    result = {"ticker": ticker, "model": MODEL, "fields": fields,
              "tokens": {"in": usage.prompt_token_count, "out": usage.candidates_token_count}}
    out.write_text(json.dumps(result, indent=2))
    return result


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    force = "--force" in sys.argv
    client = genai.Client(vertexai=True, project=PROJECT, location="us-central1")
    tickers = args or sorted(p.name for p in DATA.iterdir() if (p / "layout.json").exists())
    for t in tickers:
        r = extract(t, client, force)
        found = sum(1 for d in r["fields"].values() if d["value"] is not None)
        flagged = [f for f, d in r["fields"].items() if d["flags"]]
        print(f"{t:5} {found}/9 extracted, {r['tokens']['in']} tokens in"
              + (f"  FLAGGED {flagged}" if flagged else ""))


if __name__ == "__main__":
    main()
