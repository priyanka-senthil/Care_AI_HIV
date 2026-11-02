"""Retriever: hybrid search (dense embeddings plus keyword overlap) with a score threshold."""

from __future__ import annotations

import numpy as np

from .llm import LLM
from .vector_store import SearchHit, VectorStore


class Retriever:
    def __init__(self, store: VectorStore, llm: LLM, top_k: int = 5, min_score: float = 0.35,
                 keyword_weight: float = 0.25):
        self.store = store
        self.llm = llm
        self.top_k = top_k
        self.min_score = min_score
        self.keyword_weight = keyword_weight

    def retrieve(self, query: str, topic_hint: str | None = None) -> list[SearchHit]:
        query_vector = self.llm.embed([query], task="RETRIEVAL_QUERY")[0]
        dense = self.store.vectors @ query_vector
        keyword = self.store.keyword_scores(query)
        combined = (1 - self.keyword_weight) * dense + self.keyword_weight * keyword

        if topic_hint:  # small boost for chunks from the matching topic page
            boost = np.array([0.05 if c.get("topic") == topic_hint else 0.0 for c in self.store.chunks])
            combined = combined + boost

        order = np.argsort(-combined)
        hits, seen_text = [], set()
        for i in order:
            if combined[i] < self.min_score or len(hits) >= self.top_k:
                break
            key = self.store.chunks[i]["text"][:120]
            if key in seen_text:  # drop near-duplicate chunks from overlapping windows
                continue
            seen_text.add(key)
            hits.append(SearchHit(float(combined[i]), self.store.chunks[i]))
        return hits


def format_evidence(hits: list[SearchHit]) -> str:
    """Numbered evidence block. The numbers are used as citations in the answer."""
    lines = []
    for n, hit in enumerate(hits, start=1):
        c = hit.chunk
        lines.append(f"[{n}] {c['publisher']} | {c['title']} | {c['url']}\n{c['text']}")
    return "\n\n".join(lines)
