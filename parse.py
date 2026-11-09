"""Step 2b: send only the routed statement pages through Document AI Layout Parser.

Routing keeps each request at <= 6 pages, under the 15-page online limit, so
this uses synchronous processing (no GCS batch job). Output per company:
data/<T>/layout.json  - the Document AI response (blocks, tables, chunks)

Run:  python parse.py            (all routed companies, skips ones already parsed)
      python parse.py AAPL
"""
import io
import json
import os
import sys
from pathlib import Path

from google.api_core.client_options import ClientOptions
from google.cloud import documentai
from pypdf import PdfReader, PdfWriter

PROJECT = os.environ.get("GCP_PROJECT", "advisor-agent-2026")
LOCATION = "us"
PROCESSOR_NAME = "sec-statements-layout"
DATA = Path(__file__).parent / "data"

client = documentai.DocumentProcessorServiceClient(
    client_options=ClientOptions(api_endpoint=f"{LOCATION}-documentai.googleapis.com"))


def get_processor():
    parent = client.common_location_path(PROJECT, LOCATION)
    for p in client.list_processors(parent=parent):
        if p.display_name == PROCESSOR_NAME:
            return p.name
    created = client.create_processor(parent=parent, processor=documentai.Processor(
        display_name=PROCESSOR_NAME, type_="LAYOUT_PARSER_PROCESSOR"))
    print(f"created processor {created.name}")
    return created.name


def routed_pdf(ticker):
    """A small PDF holding only the routed pages, in order."""
    route = json.loads((DATA / ticker / "route.json").read_text())
    reader = PdfReader(DATA / ticker / "10k.pdf")
    writer = PdfWriter()
    for page_no in route["pages"]:
        writer.add_page(reader.pages[page_no - 1])
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue(), route["pages"]


def parse(ticker, processor):
    out = DATA / ticker / "layout.json"
    if out.exists():
        return
    pdf_bytes, source_pages = routed_pdf(ticker)
    if not source_pages:
        print(f"{ticker}: no statement pages routed, skipping")
        return
    result = client.process_document(request=documentai.ProcessRequest(
        name=processor,
        raw_document=documentai.RawDocument(content=pdf_bytes, mime_type="application/pdf"),
        process_options=documentai.ProcessOptions(layout_config=documentai.ProcessOptions.LayoutConfig(
            chunking_config=documentai.ProcessOptions.LayoutConfig.ChunkingConfig(
                chunk_size=1024, include_ancestor_headings=True))),
    ))
    doc = json.loads(documentai.Document.to_json(result.document))
    # Layout Parser numbers pages 1..n within the small PDF; keep the map back to the 10-K.
    doc["source_pages"] = source_pages
    out.write_text(json.dumps(doc))
    blocks = doc.get("documentLayout", {}).get("blocks", [])
    tables = sum(1 for b in walk(blocks) if "tableBlock" in b)
    print(f"{ticker}: {len(source_pages)} pages -> {len(blocks)} top blocks, {tables} tables")


def walk(blocks):
    for b in blocks:
        yield b
        yield from walk(b.get("textBlock", {}).get("blocks", []))


def main():
    processor = get_processor()
    tickers = sys.argv[1:] or sorted(p.name for p in DATA.iterdir() if (p / "route.json").exists())
    for t in tickers:
        parse(t, processor)


if __name__ == "__main__":
    main()
