"""Step 3: Gemini reads the Layout Parser output and fills a fixed schema.

Division of labor: Gemini only copies what is printed (the number as text,
the scale from the table header, the page, the row label). Code turns that
into a value and runs the checks. Any field that fails a check is flagged
for human review instead of being trusted.

Run:  python extract.py            (all parsed companies)
      python extract.py AAPL --force
      python extract.py --recheck  (re-run code checks on saved answers, no Gemini calls)
Writes data/<T>/extracted.json
"""
import json
from datetime import date
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
    column_header: Optional[str]    # header of the column the value came from, e.g. "2025"


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

The most recent fiscal year ended on {period_end}. For each field, copy the value for that fiscal
year exactly as printed (keep commas, parentheses and minus signs; drop the $ sign). Some companies
print the oldest year first, so pick the column by its header, never by its position. Also give
the table's scale from its header (e.g. "In millions" -> millions; per-share amounts are always
"units"), the page marker number, the row label as printed, and the header of the column you used
(e.g. "2025" or "December 31, 2025").

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
  If the statement shows "Net income", then a noncontrolling interests line, then "Net income
  attributable to <company>", use the last one.
- stockholders_equity: total stockholders'/shareholders' equity attributable to the company. If the
  balance sheet shows a company total, then noncontrolling interests, then a larger "Total equity",
  use the company total (the one before the noncontrolling interests line).
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


MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}


def wrong_year(header, period_end):
    """Why the column header does not match the fiscal year end, or None if it does.
    Consistency checks (A = L + E, signs) cannot see this: a whole wrong column still balances."""
    if not header:
        return "no column header given"
    end = date.fromisoformat(period_end)
    h = header.lower()
    years = {int(y) for y in re.findall(r"\b(?:19|20)\d\d\b", h)}
    if not years:
        return f"no year in column header '{header}'"
    if len(years) > 1:
        return f"column header '{header}' spans several years"
    year = years.pop()
    month = next((n for m, n in MONTHS.items() if re.search(rf"\b{m}", h)), None)
    if month:
        day = re.search(rf"\b{list(MONTHS)[month - 1]}\w*\.?\s+(\d{{1,2}})\b", h)
        printed = date(year, month, int(day.group(1)) if day else 15)
        if abs((printed - end).days) > 20:  # 52/53-week years end on varying days
            return f"column '{header}' is not the fiscal year ending {period_end}"
        return None
    # Year-only header. Fiscal years ending Jan-Mar are often labeled by the prior year
    # (Walmart's FY ending Jan 2026 is "fiscal 2025").
    if year == end.year or (end.month <= 3 and year == end.year - 1):
        return None
    return f"column '{header}' is not the fiscal year ending {period_end}"


YEAR = re.compile(r"(?<!\d)(?:19|20)\d\d(?!\d)")
VALUE = re.compile(r"\(?-?\d[\d,]*(?:\.\d+)?\)?")


def column_year(printed, text):
    """Which year column the printed value sits in, read from the PDF text layer.
    Layout Parser can merge adjacent columns and scramble their order, and Gemini then
    reports whatever header it was told to find, so neither is trusted for this.
    Returns (value's year, latest year on the page) or None if it cannot tell."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    at = line_of(printed, lines)
    if at is None:
        return None
    headers = [l for l in lines[:at] if len(YEAR.findall(l)) >= 2 and not re.search(r"\d,\d{3}(?!\d)", l)]
    if not headers:
        return None
    years = [int(y) for y in YEAR.findall(headers[-1])]
    target = printed.strip("()-$ ").replace(",", "")
    values = [v.strip("()-").replace(",", "") for v in VALUE.findall(lines[at])][-len(years):]
    if len(values) < len(years) or target not in values:
        return None
    return years[values.index(target)], max(years)


def page_scale(text):
    # First mention wins: "In millions, except shares, which are in thousands" is millions.
    head = " ".join(text.splitlines()[:15]).lower()
    m = re.search(r"(?:in |\()(thousands|millions|billions)", head)
    return m.group(1) if m else None


NCI = re.compile(r"non-?controlling|minority interest", re.I)


def line_of(printed, lines):
    """Index of the first page line that carries this printed number."""
    for i, line in enumerate(lines):
        if re.search(rf"(?<![\d,.]){re.escape(printed.strip('()-'))}(?![\d,]|\.\d)", line):
            return i
    return None


def includes_nci(field, printed, text):
    """True when the chosen line sits after a noncontrolling-interests line and a total for
    the company itself appears before it: the value includes the minority owners' share."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    at = line_of(printed, lines)
    if at is None:
        return False
    if field == "stockholders_equity":
        is_total = re.compile(r"^total\b.*\b(equity|deficit)", re.I)
        nci_before = [i for i in range(at) if NCI.search(lines[i]) and not is_total.search(lines[i])]
        return bool(nci_before) and any(
            is_total.search(lines[i]) and "liabilities" not in lines[i].lower()
            for i in range(nci_before[-1]))  # the NCI line nearest above, not a section heading
    # net income: an NCI line follows, and then a line attributable to the company
    after = range(at + 1, min(at + 6, len(lines)))
    nci_after = next((i for i in after if NCI.search(lines[i])), None)
    return nci_after is not None and any(
        re.search(r"(attributable|applicable) to", lines[i], re.I) and not NCI.search(lines[i])
        and "common" not in lines[i].lower()
        for i in range(nci_after + 1, min(nci_after + 4, len(lines))))


def check(fields, page_text, period_end):
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
        reason = wrong_year(d["column_header"], period_end)
        if reason:
            flags[f].append(reason)
        text = page_text.get(d["page"], "")
        col = column_year(d["printed"], text)
        if col and col[0] != col[1]:
            flags[f].append(f"value is in the {col[0]} column, latest is {col[1]}")
        scale = page_scale(text)
        if f != "eps_diluted" and scale and scale != d["scale"]:
            flags[f].append(f"page says {scale}, extraction used {d['scale']}")
        if f != "eps_diluted" and abs(d["value"]) > 3e12:
            flags[f].append("implausibly large for a public company (scale?)")
        if f in ("stockholders_equity", "net_income") and includes_nci(
                f, d["printed"], page_text.get(d["page"], "")):
            flags[f].append("value includes noncontrolling interests")

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


def recheck(ticker):
    """Re-run the code checks on a saved extraction, without calling Gemini again."""
    out = DATA / ticker / "extracted.json"
    result = json.loads(out.read_text())
    doc = json.loads((DATA / ticker / "layout.json").read_text())
    period_end = json.loads((DATA / ticker / "meta.json").read_text())["period_end"]
    pages = json.loads((DATA / ticker / "pages.json").read_text())
    flags = check(result["fields"], {p: pages[p - 1] for p in doc["source_pages"]}, period_end)
    for f in FIELDS:
        result["fields"][f]["flags"] = flags.get(f, [])
    out.write_text(json.dumps(result, indent=2))
    return result


def extract(ticker, client, force=False):
    out = DATA / ticker / "extracted.json"
    if out.exists() and not force:
        return json.loads(out.read_text())
    doc = json.loads((DATA / ticker / "layout.json").read_text())
    period_end = json.loads((DATA / ticker / "meta.json").read_text())["period_end"]
    resp = client.models.generate_content(
        model=MODEL,
        contents=PROMPT.format(pages=render(doc), period_end=period_end),
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
    flags = check(fields, page_text, period_end)
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
        r = recheck(t) if "--recheck" in sys.argv else extract(t, client, force)
        found = sum(1 for d in r["fields"].values() if d["value"] is not None)
        flagged = [f for f, d in r["fields"].items() if d["flags"]]
        print(f"{t:5} {found}/9 extracted, {r['tokens']['in']} tokens in"
              + (f"  FLAGGED {flagged}" if flagged else ""))


if __name__ == "__main__":
    main()
