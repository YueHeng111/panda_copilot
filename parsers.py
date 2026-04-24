from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd

try:
    import fitz
except Exception:
    fitz = None

try:
    from docx import Document
except Exception:
    Document = None


@dataclass
class DocumentChunk:
    company_id: str
    source_name: str
    source_type: str
    title: str
    text: str


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def read_json_file(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    return json.dumps(data, ensure_ascii=False, indent=2)


def read_csv_file(path: Path) -> str:
    df = pd.read_csv(path)
    return df.head(50).to_markdown(index=False)


def read_pdf_file(path: Path) -> str:
    if fitz is None:
        return f"[PDF parser unavailable] {path.name}"
    doc = fitz.open(path)
    return "\n".join(doc.load_page(i).get_text() for i in range(len(doc)))


def read_docx_file(path: Path) -> str:
    if Document is None:
        return f"[DOCX parser unavailable] {path.name}"
    doc = Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def parse_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return read_text_file(path)
    if suffix == ".json":
        return read_json_file(path)
    if suffix == ".csv":
        return read_csv_file(path)
    if suffix == ".pdf":
        return read_pdf_file(path)
    if suffix == ".docx":
        return read_docx_file(path)
    return path.read_text(encoding="utf-8", errors="ignore")


def chunk_text(text: str, max_chars: int = 1200, overlap: int = 200) -> List[str]:
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks


def load_company_documents(company_dir: Path, company_id: str) -> List[DocumentChunk]:
    docs: List[DocumentChunk] = []
    if not company_dir.exists():
        return docs
    for path in sorted(company_dir.glob("*")):
        if path.is_dir():
            continue
        text = parse_file(path)
        for idx, part in enumerate(chunk_text(text)):
            docs.append(
                DocumentChunk(
                    company_id=company_id,
                    source_name=path.name,
                    source_type=path.suffix.lower().replace(".", ""),
                    title=f"{path.stem}#{idx + 1}",
                    text=part,
                )
            )
    return docs
