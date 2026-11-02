"""Router: classifies the intent of a message and rewrites it as a standalone search query."""

from __future__ import annotations

from dataclasses import dataclass

from .llm import LLM

INTENTS = {
    "hiv_information": "General facts about HIV: how it works, viral load, CD4, transmission, U=U, testing, prevention.",
    "medication": "HIV treatment (ART): taking medicines, missed doses, side effects, interactions, adherence.",
    "emotional_support": "Feelings about a diagnosis, stigma, disclosure, coping, mental health.",
    "care_navigation": "Finding care, appointments, lab tests to expect, paying for care, support programs.",
    "out_of_scope": "Anything not related to HIV or the patient's HIV care.",
}

# Topic hints used to give a small retrieval boost to the most relevant source pages
TOPIC_HINTS = {
    "hiv_information": "basics",
    "medication": "treatment",
    "emotional_support": "mental-health",
    "care_navigation": "care",
}

ROUTER_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": list(INTENTS)},
        "rewritten_query": {"type": "string"},
        "reason": {"type": "string"},
    },
    "required": ["intent", "rewritten_query", "reason"],
}

ROUTER_PROMPT = """You route questions for an HIV patient information assistant.

Intents:
{intents}

Tasks:
1. Pick the single best intent for the latest message.
2. Rewrite the latest message as a short, standalone search query. Resolve words like "it" or
   "that" using the conversation. Keep medical terms. Do not add facts.

Conversation so far:
{history}

QUESTION: {message}"""


@dataclass
class Route:
    intent: str
    query: str
    reason: str

    @property
    def topic_hint(self) -> str | None:
        return TOPIC_HINTS.get(self.intent)


def route(llm: LLM, message: str, history: str = "", model: str | None = None) -> Route:
    intents = "\n".join(f"- {name}: {desc}" for name, desc in INTENTS.items())
    try:
        result = llm.generate_json(
            ROUTER_PROMPT.format(intents=intents, history=history or "(none)", message=message),
            ROUTER_SCHEMA, model=model)
    except Exception:  # fall back to plain search on the original message
        return Route("hiv_information", message, "router unavailable")
    intent = result.get("intent") if result.get("intent") in INTENTS else "hiv_information"
    return Route(intent, result.get("rewritten_query") or message, result.get("reason", ""))
