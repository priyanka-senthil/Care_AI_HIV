"""Model access layer.

GeminiLLM talks to Gemini on Vertex AI through the google-genai SDK.
MockLLM gives deterministic answers so the whole pipeline can run offline in tests and demos.
Both share the same small interface, so the rest of the code does not care which one is used.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Protocol

import numpy as np

from .config import backend, load_config


class LLM(Protocol):
    def generate(self, prompt: str, system: str = "", model: str | None = None,
                 temperature: float = 0.2) -> str: ...

    def generate_json(self, prompt: str, schema: dict, system: str = "",
                      model: str | None = None) -> dict: ...

    def embed(self, texts: list[str], task: str = "RETRIEVAL_DOCUMENT") -> np.ndarray: ...


class GeminiLLM:
    """Gemini on Vertex AI. Requires `gcloud auth application-default login` and a project with Vertex AI enabled."""

    def __init__(self, config: dict | None = None):
        from google import genai  # imported here so mock mode works without credentials

        self.cfg = config or load_config()
        gcp = self.cfg["gcp"]
        if not gcp.get("project"):
            raise RuntimeError("GOOGLE_CLOUD_PROJECT is not set. Copy .env.example to .env and fill it in.")
        self.client = genai.Client(vertexai=True, project=gcp["project"], location=gcp["location"])
        self.models = self.cfg["models"]

    def generate(self, prompt, system="", model=None, temperature=0.2):
        from google.genai import types

        response = self.client.models.generate_content(
            model=model or self.models["generator"],
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system or None,
                temperature=temperature,
                max_output_tokens=2048,
            ),
        )
        return (response.text or "").strip()

    def generate_json(self, prompt, schema, system="", model=None):
        from google.genai import types

        response = self.client.models.generate_content(
            model=model or self.models["router"],
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system or None,
                temperature=0.0,
                response_mime_type="application/json",
                response_schema=schema,
            ),
        )
        return json.loads(response.text)

    def embed(self, texts, task="RETRIEVAL_DOCUMENT"):
        from google.genai import types

        vectors = []
        for start in range(0, len(texts), 100):  # stay under the per-request batch limit
            batch = texts[start:start + 100]
            result = self.client.models.embed_content(
                model=self.models["embedding"],
                contents=batch,
                config=types.EmbedContentConfig(task_type=task),
            )
            vectors.extend(e.values for e in result.embeddings)
        return _normalize(np.array(vectors, dtype=np.float32))


class MockLLM:
    """Offline stand-in. Uses keyword rules for JSON tasks and a hashing embedder for retrieval."""

    DIM = 512

    def generate(self, prompt, system="", model=None, temperature=0.2):
        evidence = re.findall(r"\[(\d+)\][^\n]*\n(.+?)(?=\n\[\d+\]|\nQUESTION:|\Z)", prompt, flags=re.S)
        if not evidence:
            return ("I could not find this in my trusted sources. "
                    "Please ask your HIV care provider or pharmacist.")
        number, text = evidence[0]
        sentence = text.strip().split(". ")[0].rstrip(".")
        return f"{sentence}. [{number}]\n\nPlease talk with your HIV care provider about your own situation."

    def generate_json(self, prompt, schema, system="", model=None):
        text = prompt.lower()
        props = schema.get("properties", {})
        if "intent" in props:
            question = text.split("question:")[-1]
            if any(w in question for w in ["sad", "scared", "alone", "afraid", "ashamed", "depress", "anxious", "cope"]):
                intent = "emotional_support"
            elif any(w in question for w in ["medicine", "medication", "pill", "dose", "art ", "side effect", "missed", "drug"]):
                intent = "medication"
            elif any(w in question for w in ["weather", "recipe", "football", "stock", "movie"]):
                intent = "out_of_scope"
            else:
                intent = "hiv_information"
            return {"intent": intent, "rewritten_query": question.strip(), "reason": "mock rule"}
        if "level" in props:
            return {"level": "routine", "reason": "mock rule"}
        if "grounded" in props:
            return {"grounded": True, "unsupported_claims": []}
        return {}

    def embed(self, texts, task="RETRIEVAL_DOCUMENT"):
        matrix = np.zeros((len(texts), self.DIM), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in re.findall(r"[a-z0-9]+", text.lower()):
                if len(token) < 3:
                    continue
                h = int(hashlib.md5(token.encode()).hexdigest(), 16)
                matrix[i, h % self.DIM] += 1.0
        return _normalize(matrix)


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def get_llm() -> LLM:
    return MockLLM() if backend() == "mock" else GeminiLLM()
