"""Step 1: build the document set and the answer key.

For each company: download its latest 10-K from EDGAR, print it to PDF with
headless Chrome (the PDF carries no XBRL tags, so the pipeline has to read it
like any client PDF), and pull the official reported values for that same
filing from the XBRL companyfacts API. The answer key is never shown to the
pipeline; it is only used to score it.

Run:  python corpus.py            (all companies)
      python corpus.py AAPL DKNG  (just these)
"""
import json
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

import requests

# SEC asks for a descriptive User-Agent; no personal contact needed for this volume.
HEADERS = {"User-Agent": "sec-docintel research project admin@example.com"}
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
DATA = Path(__file__).parent / "data"

# Mix of sectors on purpose: banks and insurers report "revenue" differently,
# which is exactly where extraction gets hard.
TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN",   # tech
    "JPM", "GS",                      # banks
    "JNJ", "UNH", "PFE",              # health
    "XOM", "CAT", "KO",               # energy, industrial, staples
    "WMT", "COST", "DIS",             # retail, media
    "NKE", "LULU", "DKNG", "MSGS", "PLNT",  # sports, apparel, fitness
]

# Tickers whose current CIK is not the one that filed the 10-K. XOM moved to a
# new holding company; the 10-K is still under Exxon Mobil Corp.
CIK_OVERRIDES = {"XOM": "0000034088"}

# Field -> XBRL concepts to try, in order. "duration" = covers the fiscal year,
# "instant" = balance at fiscal year end.
FIELDS = {
    "revenue": ("duration", ["Revenues",
                             "RevenueFromContractWithCustomerExcludingAssessedTax",
                             "RevenueFromContractWithCustomerIncludingAssessedTax",
                             "SalesRevenueNet",
                             "RevenuesNetOfInterestExpense"]),  # banks: "total net revenues"
    "operating_income": ("duration", ["OperatingIncomeLoss"]),
    "net_income": ("duration", ["NetIncomeLoss",
                                "NetIncomeLossAvailableToCommonStockholdersBasic"]),
    "eps_diluted": ("duration", ["EarningsPerShareDiluted"]),
    "operating_cash_flow": ("duration", ["NetCashProvidedByUsedInOperatingActivities"]),
    "total_assets": ("instant", ["Assets"]),
    "total_liabilities": ("instant", ["Liabilities"]),
    # Some companies tag their printed "total equity" line with the NCI-inclusive concept.
    "stockholders_equity": ("instant", ["StockholdersEquity",
                                        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"]),
    "cash": ("instant", ["CashAndCashEquivalentsAtCarryingValue"]),
}


def get(url):
    time.sleep(0.15)  # SEC limit is 10 requests/second
    r = requests.get(url, headers=HEADERS, timeout=60)
    r.raise_for_status()
    return r


def ticker_to_cik():
    rows = get("https://www.sec.gov/files/company_tickers.json").json().values()
    return {row["ticker"]: str(row["cik_str"]).zfill(10) for row in rows}


def latest_10k(cik):
    recent = get(f"https://data.sec.gov/submissions/CIK{cik}.json").json()["filings"]["recent"]
    for i, form in enumerate(recent["form"]):
        if form == "10-K":
            return {
                "accn": recent["accessionNumber"][i],
                "filed": recent["filingDate"][i],
                "period_end": recent["reportDate"][i],
                "primary_doc": recent["primaryDocument"][i],
            }
    return None


def answer_key(cik, filing):
    facts = get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json").json()
    gaap = facts["facts"].get("us-gaap", {})
    end = filing["period_end"]
    key = {}
    for field, (kind, concepts) in FIELDS.items():
        for concept in concepts:
            if concept not in gaap:
                continue
            for unit_rows in gaap[concept]["units"].values():
                for row in unit_rows:
                    if row["accn"] != filing["accn"] or row["end"] != end:
                        continue
                    if kind == "duration":
                        # A 10-K also reports quarters and prior years; keep the full current year.
                        if "start" not in row:
                            continue
                        days = (date.fromisoformat(end) - date.fromisoformat(row["start"])).days
                        if not 350 <= days <= 380:
                            continue
                    elif "start" in row:
                        continue
                    key[field] = {"value": row["val"], "concept": concept}
                    break
                if field in key:
                    break
            if field in key:
                break
    return key


def render_pdf(html_path, pdf_path):
    subprocess.run([
        CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
        f"--print-to-pdf={pdf_path}", html_path.resolve().as_uri(),
    ], check=True, capture_output=True, timeout=300)


def page_count(pdf_path):
    # Good enough for counting: each page object has exactly one /Type /Page.
    raw = pdf_path.read_bytes()
    return raw.count(b"/Type /Page") - raw.count(b"/Type /Pages")


def build(ticker, cik):
    folder = DATA / ticker
    folder.mkdir(parents=True, exist_ok=True)
    filing = latest_10k(cik)
    if not filing:
        print(f"{ticker}: no 10-K found")
        return None

    accn_path = filing["accn"].replace("-", "")
    url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accn_path}/{filing['primary_doc']}"
    html = folder / "10k.html"
    pdf = folder / "10k.pdf"
    if not html.exists():
        html.write_bytes(get(url).content)
    if not pdf.exists():
        render_pdf(html, pdf)

    key = answer_key(cik, filing)
    meta = {"ticker": ticker, "cik": cik, "form": "10-K", "source_url": url,
            "pages": page_count(pdf), **filing}
    (folder / "meta.json").write_text(json.dumps(meta, indent=2))
    (folder / "answer_key.json").write_text(json.dumps(key, indent=2))
    print(f"{ticker}: FY ending {filing['period_end']}, {meta['pages']} pages, "
          f"{len(key)}/{len(FIELDS)} fields in key")
    return meta


def main():
    tickers = sys.argv[1:] or TICKERS
    ciks = {**ticker_to_cik(), **CIK_OVERRIDES}
    for t in tickers:
        if t not in ciks:
            print(f"{t}: ticker not found")
            continue
        build(t, ciks[t])


if __name__ == "__main__":
    main()
