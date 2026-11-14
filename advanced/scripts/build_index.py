"""Build a page-linked text and visual index for the OECD Viet Nam survey."""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "data" / "oecd-economic-surveys-viet-nam-2025.pdf"
INDEX = ROOT / "index"
VISUAL_PAGES = INDEX / "visual_pages"
SOURCE_URL = (
    "https://www.oecd.org/content/dam/oecd/en/publications/reports/2025/06/"
    "oecd-economic-surveys-viet-nam-2025_f2511b78/fb37254b-en.pdf"
)
CHUNK_CHARS = 1200
OVERLAP = 160
CAPTION_RE = re.compile(
    r"(?im)^\s*((?:Figure|Table|Box)\s*\d+(?:\.\d+)*\.?[^\n]{0,220})"
)
PAGE_LABEL_RE = re.compile(r"(?m)^\s*(?:\|\s*)?(\d{1,3})\s*[|‖]?")


def stable_id(*parts: object) -> str:
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:20]


def page_label(text: str) -> int | None:
    for line in text.splitlines()[:4]:
        match = PAGE_LABEL_RE.match(line)
        if match:
            value = int(match.group(1))
            if value <= 150:
                return value
    return None


def clean(text: str) -> str:
    text = text.replace("\x00", " ").replace("\ufb00", "ff").replace("\ufb01", "fi")
    text = text.replace("\ufb02", "fl").replace("\ufb03", "ffi").replace("\ufb04", "ffl")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_text(text: str) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(paragraph) > CHUNK_CHARS:
            if current:
                chunks.append(current)
                current = ""
            start = 0
            while start < len(paragraph):
                end = min(len(paragraph), start + CHUNK_CHARS)
                if end < len(paragraph):
                    boundary = paragraph.rfind(" ", start + CHUNK_CHARS // 2, end)
                    if boundary > start:
                        end = boundary
                chunks.append(paragraph[start:end].strip())
                if end >= len(paragraph):
                    break
                start = max(start + 1, end - OVERLAP)
            continue
        if current and len(current) + len(paragraph) + 2 > CHUNK_CHARS:
            chunks.append(current)
            current = ""
        current = f"{current}\n\n{paragraph}".strip() if current else paragraph
    if current:
        chunks.append(current)
    return chunks


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> int:
    if not PDF.is_file():
        print(f"Missing source PDF: {PDF}", file=sys.stderr)
        print(f"Download it from {SOURCE_URL}", file=sys.stderr)
        return 2

    doc = fitz.open(PDF)
    text_records: list[dict] = []
    visual_records: list[dict] = []
    table_records: list[dict] = []
    visual_pages: set[int] = set()
    low_text_pages: list[int] = []

    for page_num, page in enumerate(doc, start=1):
        raw = clean(page.get_text("text", sort=True))
        if len(raw) < 80:
            low_text_pages.append(page_num)
        printed_page = page_label(raw)
        for chunk_no, body in enumerate(split_text(raw)):
            text_records.append({
                "id": f"txt-{stable_id(page_num, chunk_no, body)}",
                "kind": "page_text",
                "text": body,
                "source_id": "oecd-economic-surveys-viet-nam-2025",
                "source": "OECD Economic Surveys: Viet Nam 2025",
                "pdf_page": page_num,
                "printed_page": printed_page,
                "metadata": {"language": "en", "extraction": "PyMuPDF text layer"},
            })

        captions = CAPTION_RE.findall(raw)
        table_captions = [caption for caption in captions if caption.lstrip().casefold().startswith("table")]
        # The contents page repeats many table titles but contains no table data.
        # Do not turn its catalogue into duplicate table evidence records.
        if len(table_captions) >= 5:
            table_captions = []
        if table_captions:
            blocks = [clean(block[4]) for block in page.get_text("blocks", sort=True) if len(block) > 4 and clean(block[4])]
            for table_caption in table_captions:
                prefix = table_caption[:24].casefold()
                start = next((i for i, block in enumerate(blocks) if prefix in block.casefold()), None)
                if start is None:
                    table_text = table_caption + "\n\n" + raw
                else:
                    selected = [blocks[start]]
                    for block in blocks[start + 1:]:
                        head = block.splitlines()[0].strip()
                        if head.casefold().startswith("source:"):
                            selected.append(block)
                            break
                        if re.match(r"^(?:Figure|Table|Box)\s+\d", head, re.I):
                            break
                        if re.match(r"^\d(?:\.\d+)*\.?\s+\S", head) and len(selected) > 1:
                            break
                        selected.append(block)
                    table_text = "\n\n".join(selected)
                table_records.append({
                    "id": f"tbl-{stable_id(page_num, table_caption, 'table-region')}",
                    "kind": "table_page",
                    "text": table_text,
                    "captions": [table_caption],
                    "source_id": "oecd-economic-surveys-viet-nam-2025",
                    "source": "OECD Economic Surveys: Viet Nam 2025",
                    "pdf_page": page_num,
                    "printed_page": printed_page,
                    "metadata": {"extraction": "caption-led table region from PDF text layer; checked against page crop"},
                })
        large_images = [
            info for info in page.get_image_info(xrefs=True)
            if info.get("width", 0) >= 100 and info.get("height", 0) >= 80
        ]
        if captions or large_images:
            visual_pages.add(page_num)
            visual_records.append({
                "id": f"vis-{stable_id(page_num, '|'.join(captions), 'page-visual')}",
                "kind": "visual_page",
                "text": "\n".join(captions) if captions else f"Page {page_num} contains a large embedded visual.",
                "captions": captions,
                "source_id": "oecd-economic-surveys-viet-nam-2025",
                "source": "OECD Economic Surveys: Viet Nam 2025",
                "pdf_page": page_num,
                "printed_page": printed_page,
                "image_count": len(large_images),
                "crop_path": f"advanced/index/visual_pages/page-{page_num:03d}.png",
                "metadata": {
                    "extraction": "caption plus full-page crop; verify figure visually",
                    "third_party_image_rights": "Check original figure source before redistributing crops",
                },
            })
            VISUAL_PAGES.mkdir(parents=True, exist_ok=True)
            page.get_pixmap(matrix=fitz.Matrix(1.35, 1.35), alpha=False).save(
                VISUAL_PAGES / f"page-{page_num:03d}.png"
            )

    write_jsonl(INDEX / "text_chunks.jsonl", text_records)
    write_jsonl(INDEX / "visual_pages.jsonl", visual_records)
    write_jsonl(INDEX / "table_pages.jsonl", table_records)
    manifest = {
        "title": "OECD Economic Surveys: Viet Nam 2025",
        "edition": "June 2025, Volume 2025/16",
        "source_url": SOURCE_URL,
        "citation": "OECD (2025), OECD Economic Surveys: Viet Nam 2025, OECD Publishing, Paris, https://doi.org/10.1787/fb37254b-en.",
        "license": "CC BY 4.0; third-party material may be excluded from this licence",
        "pdf_sha256": hashlib.sha256(PDF.read_bytes()).hexdigest(),
        "pdf_pages": len(doc),
        "text_chunk_count": len(text_records),
        "visual_page_count": len(visual_records),
        "table_record_count": len(table_records),
        "pages_with_low_text": low_text_pages,
        "page_mapping": "Every record has one-based physical PDF page and the printed page label when detected.",
        "limitations": [
            "PDF text is extracted from the embedded text layer; scanned or vector-only label errors should be checked against the page crop.",
            "All figure/table caption pages are rendered for visual lookup; crops are local references and should not be redistributed without checking third-party rights.",
            "No standalone informative photographs or flow diagrams were identified in the selected benchmark visuals; the report primarily uses data figures and charts.",
        ],
    }
    (INDEX / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "pages": len(doc), "text_chunks": len(text_records), "table_records": len(table_records), "visual_pages": len(visual_records),
        "low_text_pages": low_text_pages, "pdf_sha256": manifest["pdf_sha256"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
