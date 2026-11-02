"""Orchestrator: the agentic workflow that ties every step together.

    message
      -> 1. triage            (rules + Gemini Flash)  -> emergency/crisis: fixed safety reply, stop
      -> 2. route             (Gemini Flash)          -> out_of_scope: polite redirect, stop
      -> 3. retrieve          (embeddings + keywords) -> no evidence: "ask your care team", stop
      -> 4. generate          (tuned Gemini or Gemini Pro, specialist instruction)
      -> 5. safety check      (rules + grounding check) -> on failure: regenerate once, then fall back
      -> 6. add urgent note   (if triage said urgent)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from . import agents
from .config import PROJECT_ROOT, load_config
from .llm import LLM, get_llm
from .retriever import Retriever, format_evidence
from .router import route
from .safety import grounding_check, rule_check
from .triage import (CRISIS_MESSAGE, DEFAULT_URGENT_NOTE, EMERGENCY_MESSAGE, URGENT_NOTES, triage)
from .vector_store import VectorStore


@dataclass
class Citation:
    number: int
    title: str
    publisher: str
    url: str


@dataclass
class CareAIResponse:
    answer: str
    triage_level: str
    intent: str | None = None
    citations: list[Citation] = field(default_factory=list)
    trace: list[dict] = field(default_factory=list)


class CareAI:
    def __init__(self, llm: LLM | None = None, store: VectorStore | None = None, config: dict | None = None):
        self.cfg = config or load_config()
        self.llm = llm or get_llm()
        models = self.cfg["models"]
        self.router_model = models["router"]
        self.answer_model = models.get("tuned_generator") or models["generator"]
        rag = self.cfg["rag"]
        self.store = store or VectorStore.load(PROJECT_ROOT / rag["index_dir"])
        self.retriever = Retriever(self.store, self.llm, rag["top_k"], rag["min_score"])
        self.use_grounding = self.cfg.get("safety", {}).get("grounding_check", True)

    @staticmethod
    def _history_text(history: list[dict] | None, turns: int = 6) -> str:
        if not history:
            return ""
        return "\n".join(f"{m['role']}: {m['content'][:400]}" for m in history[-turns:])

    def answer(self, message: str, history: list[dict] | None = None) -> CareAIResponse:
        trace: list[dict] = []
        hist = self._history_text(history)

        def log(step: str, start: float, **details):
            trace.append({"step": step, "ms": round((time.perf_counter() - start) * 1000), **details})

        # 1. Triage
        t = time.perf_counter()
        tri = triage(self.llm, message, hist, self.router_model)
        log("triage", t, level=tri.level, reasons=tri.reasons, source=tri.source)
        if tri.level == "crisis":
            return CareAIResponse(CRISIS_MESSAGE, tri.level, trace=trace)
        if tri.level == "emergency":
            return CareAIResponse(EMERGENCY_MESSAGE, tri.level, trace=trace)

        # 2. Route
        t = time.perf_counter()
        r = route(self.llm, message, hist, self.router_model)
        log("route", t, intent=r.intent, query=r.query)
        if r.intent == "out_of_scope":
            return CareAIResponse(agents.OUT_OF_SCOPE_MESSAGE, tri.level, r.intent, trace=trace)

        # 3. Retrieve
        t = time.perf_counter()
        hits = self.retriever.retrieve(r.query, r.topic_hint)
        log("retrieve", t, hits=[{"score": round(h.score, 3), "id": h.chunk["chunk_id"]} for h in hits])
        if not hits:
            answer = agents.NO_EVIDENCE_MESSAGE
            return CareAIResponse(self._add_urgent_note(answer, tri), tri.level, r.intent, trace=trace)

        evidence = format_evidence(hits)
        citations = [Citation(n, h.chunk["title"], h.chunk["publisher"], h.chunk["url"])
                     for n, h in enumerate(hits, start=1)]

        # 4. Generate, then 5. check safety (one retry with a stricter instruction)
        system = agents.system_instruction(r.intent)
        prompt = agents.build_answer_prompt(evidence, message)
        answer = None
        for attempt in (1, 2):
            t = time.perf_counter()
            draft = self.llm.generate(prompt, system=system, model=self.answer_model)
            log("generate", t, attempt=attempt, model=self.answer_model)

            t = time.perf_counter()
            check = rule_check(draft, len(hits))
            if check.passed and self.use_grounding:
                check = grounding_check(self.llm, draft, evidence, self.router_model)
            log("safety_check", t, attempt=attempt, passed=check.passed, issues=check.issues)
            if check.passed:
                answer = draft
                break
            system += ("\n\nIMPORTANT: Your previous answer had these problems: "
                       + "; ".join(check.issues)
                       + ". Use only statements the evidence supports and cite each one.")

        if answer is None:
            answer = agents.NO_EVIDENCE_MESSAGE
            citations = []

        # 6. Urgent note
        answer = self._add_urgent_note(answer, tri)
        used = {c.number for c in citations if f"[{c.number}]" in answer}
        return CareAIResponse(answer, tri.level, r.intent, [c for c in citations if c.number in used], trace)

    @staticmethod
    def _add_urgent_note(answer: str, tri) -> str:
        if tri.level != "urgent":
            return answer
        notes = [URGENT_NOTES[r] for r in tri.reasons if r in URGENT_NOTES] or [DEFAULT_URGENT_NOTE]
        return "**Please act soon:** " + " ".join(dict.fromkeys(notes)) + "\n\n" + answer
