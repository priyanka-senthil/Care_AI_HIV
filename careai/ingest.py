"""Ingestion: download trusted pages, clean them, and split them into overlapping chunks."""

from __future__ import annotations

import io
import re
from dataclasses import asdict, dataclass

import requests
import yaml
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "CareAI-research-bot/1.0 (student project; health information)"}
DROP_TAGS = ["script", "style", "nav", "header", "footer", "aside", "form", "noscript", "svg", "button"]


@dataclass
class Document:
    url: str
    title: str
    publisher: str
    topic: str
    text: str


@dataclass
class Chunk:
    chunk_id: str
    url: str
    title: str
    publisher: str
    topic: str
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


def load_sources(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)["sources"]


def html_to_text(html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    title = (soup.title.string or "").strip() if soup.title else ""
    for tag in soup(DROP_TAGS):
        tag.decompose()
    main = soup.find("main") or soup.find("article") or soup.body or soup
    blocks = []
    for el in main.find_all(["h1", "h2", "h3", "h4", "p", "li", "td"]):
        line = " ".join(el.get_text(" ", strip=True).split())
        if len(line) > 2:
            blocks.append(line)
    text = "\n".join(dict.fromkeys(blocks))  # remove repeated lines, keep order
    return title, text


def pdf_to_text(data: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def fetch_document(source: dict, timeout: int = 30) -> Document:
    response = requests.get(source["url"], headers=HEADERS, timeout=timeout)
    response.raise_for_status()
    if source["url"].lower().endswith(".pdf") or "pdf" in response.headers.get("content-type", ""):
        title, text = source.get("title", source["url"].rsplit("/", 1)[-1]), pdf_to_text(response.content)
    else:
        title, text = html_to_text(response.text)
    return Document(
        url=source["url"],
        title=source.get("title") or title or source["url"],
        publisher=source.get("publisher", "Unknown"),
        topic=source.get("topic", "general"),
        text=text,
    )


def chunk_document(doc: Document, size: int = 220, overlap: int = 40) -> list[Chunk]:
    """Splits by paragraphs first, then packs paragraphs into word windows with overlap."""
    paragraphs = [p.strip() for p in re.split(r"\n+", doc.text) if p.strip()]
    chunks, window = [], []
    for para in paragraphs:
        words = para.split()
        while words:
            room = size - len(window)
            window.extend(words[:room])
            words = words[room:]
            if len(window) >= size:
                chunks.append(" ".join(window))
                window = window[-overlap:] if overlap else []
    if window and (not chunks or len(window) > overlap):
        chunks.append(" ".join(window))

    slug = re.sub(r"[^a-z0-9]+", "-", doc.url.lower()).strip("-")[-60:]
    return [
        Chunk(chunk_id=f"{slug}#{i}", url=doc.url, title=doc.title,
              publisher=doc.publisher, topic=doc.topic, text=text)
        for i, text in enumerate(chunks)
        if len(text.split()) >= 25  # skip tiny fragments such as menu leftovers
    ]
