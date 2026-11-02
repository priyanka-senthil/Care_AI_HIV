"""A small vector index stored as two files: vectors.npy and chunks.jsonl.

Cosine similarity over normalized vectors is exact and fast for a knowledge base of this size
(a few thousand chunks). The class can be swapped for Vertex AI Vector Search if the corpus grows.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class SearchHit:
    score: float
    chunk: dict


class VectorStore:
    def __init__(self, vectors: np.ndarray, chunks: list[dict]):
        if len(vectors) != len(chunks):
            raise ValueError("Vector count and chunk count do not match.")
        self.vectors = vectors.astype(np.float32)
        self.chunks = chunks

    def __len__(self) -> int:
        return len(self.chunks)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[SearchHit]:
        scores = self.vectors @ query_vector.reshape(-1)
        order = np.argsort(-scores)[:top_k]
        return [SearchHit(float(scores[i]), self.chunks[i]) for i in order]

    def keyword_scores(self, query: str) -> np.ndarray:
        """Simple term-overlap score, used for hybrid ranking with the dense score."""
        terms = {t for t in re.findall(r"[a-z0-9]+", query.lower()) if len(t) > 2}
        if not terms:
            return np.zeros(len(self.chunks), dtype=np.float32)
        scores = []
        for chunk in self.chunks:
            words = set(re.findall(r"[a-z0-9]+", chunk["text"].lower()))
            scores.append(len(terms & words) / len(terms))
        return np.array(scores, dtype=np.float32)

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self.vectors)
        with open(directory / "chunks.jsonl", "w", encoding="utf-8") as f:
            for chunk in self.chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, directory: str | Path) -> "VectorStore":
        directory = Path(directory)
        vectors = np.load(directory / "vectors.npy")
        with open(directory / "chunks.jsonl", encoding="utf-8") as f:
            chunks = [json.loads(line) for line in f if line.strip()]
        return cls(vectors, chunks)
