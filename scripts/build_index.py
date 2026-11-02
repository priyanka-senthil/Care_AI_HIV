"""Builds the RAG index from data/sources.yaml (and any files placed in data/local_docs/).

Usage:
    python scripts/build_index.py
    python scripts/build_index.py --sources data/sources.yaml --out data/index
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from careai.config import PROJECT_ROOT, load_config  # noqa: E402
from careai.ingest import Document, chunk_document, fetch_document, load_sources, pdf_to_text  # noqa: E402
from careai.llm import get_llm  # noqa: E402
from careai.vector_store import VectorStore  # noqa: E402


def load_local_docs(folder: Path) -> list[Document]:
    """Optional: .txt, .md or .pdf files you trust, placed in data/local_docs/."""
    docs = []
    if not folder.exists():
        return docs
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() in {".txt", ".md"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
        elif path.suffix.lower() == ".pdf":
            text = pdf_to_text(path.read_bytes())
        else:
            continue
        docs.append(Document(url=f"local://{path.name}", title=path.stem, publisher="Local document",
                             topic="local", text=text))
    return docs


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", default=str(PROJECT_ROOT / "data" / "sources.yaml"))
    parser.add_argument("--out", default=str(PROJECT_ROOT / cfg["rag"]["index_dir"]))
    args = parser.parse_args()

    documents, failures = [], []
    for source in load_sources(args.sources):
        try:
            doc = fetch_document(source)
            documents.append(doc)
            print(f"  ok    {len(doc.text.split()):>6} words  {source['url']}")
        except Exception as exc:  # keep going if one page moved or is down
            failures.append({"url": source["url"], "error": str(exc)})
            print(f"  skip  {source['url']}  ({exc})")
        time.sleep(0.5)  # be polite to public servers

    documents += load_local_docs(PROJECT_ROOT / "data" / "local_docs")
    if not documents:
        sys.exit("No documents were loaded. Check your internet connection and data/sources.yaml.")

    rag = cfg["rag"]
    chunks = [c.to_dict() for d in documents
              for c in chunk_document(d, rag["chunk_size_words"], rag["chunk_overlap_words"])]
    print(f"\n{len(documents)} documents -> {len(chunks)} chunks. Creating embeddings...")

    llm = get_llm()
    vectors = llm.embed([f"{c['title']}\n{c['text']}" for c in chunks], task="RETRIEVAL_DOCUMENT")
    store = VectorStore(vectors, chunks)
    store.save(args.out)

    report = {"documents": len(documents), "chunks": len(chunks), "failed_sources": failures}
    Path(args.out, "build_report.json").write_text(json.dumps(report, indent=2))
    print(f"Index saved to {args.out}. {len(failures)} source(s) failed; see build_report.json.")


if __name__ == "__main__":
    main()
