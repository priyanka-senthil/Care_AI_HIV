"""Triage: decides how urgent a message is BEFORE any answer is generated.

Layer 1 is a fixed rule list. It always runs, needs no model, and cannot be talked out of an emergency.
Layer 2 asks a fast Gemini model to catch urgent cases the rules miss.
The final level is the more serious of the two layers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .llm import LLM

LEVELS = ["routine", "urgent", "emergency", "crisis"]  # ordered from least to most serious

# Each rule: (level, reason, regex). Patterns are matched on lower-case text.
RULES: list[tuple[str, str, str]] = [
    ("crisis", "possible thoughts of suicide or self-harm",
     r"\b(kill myself|end my life|suicid\w*|want to die|don'?t want to (live|be here)|hurt myself|self[- ]harm)\b"),
    ("emergency", "trouble breathing",
     r"\b(can'?t|cannot|hard to|trouble|difficulty) breath\w*|short(ness)? of breath\b"),
    ("emergency", "chest pain", r"\bchest (pain|pressure|tightness)\b"),
    ("emergency", "possible severe allergic reaction",
     r"\b(throat|tongue|lips?|face) (is |are )?(swelling|swollen|closing)\b|\banaphyla\w*"),
    ("emergency", "loss of consciousness or seizure",
     r"\b(passed out|fainted|unconscious|seizure|convuls\w*|unresponsive)\b"),
    ("emergency", "possible overdose", r"\b(overdos\w*|took too many (pills|tablets))\b"),
    ("emergency", "severe headache with stiff neck or confusion",
     r"\b(stiff neck|worst headache|sudden confusion|confused and)\b"),
    ("urgent", "possible recent HIV exposure (PEP is time-sensitive)",
     r"\b(condom (broke|slipped)|needle ?stick|shared (a )?needle|exposed to hiv|possible exposure|unprotected sex|sexual assault|was raped)\b"),
    ("urgent", "new rash or fever after starting a medicine",
     r"(?=.*\b(rash|blister\w*|peeling skin|fever)\b)(?=.*\b(medicines?|medications?|pills?|drugs?|art)\b)(?=.*\b(started|starting|new|began|since)\b)"),
    ("urgent", "high or lasting fever", r"\b(high fever|fever (for|of) (\d+|several|many) days|fever over 10[3-9])\b"),
    ("urgent", "vision changes", r"\b(blurr?y vision|vision (loss|changes)|floaters|losing (my )?sight)\b"),
    ("urgent", "pregnancy-related concern", r"\b(pregnan\w*)\b.*\b(missed|stopped|bleeding|pain)\b"),
]


@dataclass
class TriageResult:
    level: str
    reasons: list[str] = field(default_factory=list)
    source: str = "rules"

    @property
    def is_blocking(self) -> bool:
        """Emergency and crisis messages get a fixed safety response instead of a generated answer."""
        return self.level in {"emergency", "crisis"}


def rule_triage(text: str) -> TriageResult:
    lowered = text.lower()
    level, reasons = "routine", []
    for rule_level, reason, pattern in RULES:
        if re.search(pattern, lowered):
            reasons.append(reason)
            if LEVELS.index(rule_level) > LEVELS.index(level):
                level = rule_level
    return TriageResult(level, reasons, "rules")


TRIAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "level": {"type": "string", "enum": ["routine", "urgent", "emergency", "crisis"]},
        "reason": {"type": "string"},
    },
    "required": ["level", "reason"],
}

TRIAGE_PROMPT = """You are the triage step of an HIV patient information assistant.
Classify the urgency of the patient's latest message.

- crisis: thoughts of suicide or self-harm.
- emergency: symptoms that need emergency care now (for example trouble breathing, chest pain,
  signs of a severe allergic reaction, seizure, fainting, severe confusion).
- urgent: needs contact with a clinician within about a day (for example a possible HIV exposure
  in the last 72 hours, a new rash or fever after starting a medicine, many missed doses,
  high fever, vision changes).
- routine: general questions about HIV, treatment, daily life, or care.

When unsure between two levels, choose the more serious one.

Conversation so far:
{history}

Latest message: {message}"""


def model_triage(llm: LLM, message: str, history: str, model: str | None) -> TriageResult:
    result = llm.generate_json(TRIAGE_PROMPT.format(history=history or "(none)", message=message),
                               TRIAGE_SCHEMA, model=model)
    level = result.get("level", "routine")
    if level not in LEVELS:
        level = "routine"
    return TriageResult(level, [result.get("reason", "")], "model")


def triage(llm: LLM, message: str, history: str = "", model: str | None = None) -> TriageResult:
    rules = rule_triage(message)
    if rules.is_blocking:  # do not wait for a model when a rule already found an emergency
        return rules
    try:
        model_result = model_triage(llm, message, history, model)
    except Exception:  # if the model call fails, the rule result still stands
        return rules
    if LEVELS.index(model_result.level) > LEVELS.index(rules.level):
        return TriageResult(model_result.level, rules.reasons + model_result.reasons, "rules+model")
    return rules


EMERGENCY_MESSAGE = (
    "**This may be a medical emergency.** Please call **911** (or your local emergency number) "
    "or go to the nearest emergency room now. I am an information tool and cannot assess "
    "symptoms. If you can, tell the emergency team that you are living with HIV and which "
    "medicines you take."
)

CRISIS_MESSAGE = (
    "I am really sorry you are feeling this way, and I am glad you said something. "
    "You do not have to go through this alone. In the U.S. you can call or text **988** "
    "(Suicide and Crisis Lifeline) at any time to talk with a trained counselor. If you are in "
    "immediate danger, please call **911**. If you can, reach out to someone you trust or your "
    "HIV care team today."
)

URGENT_NOTES = {
    "possible recent HIV exposure (PEP is time-sensitive)":
        "If this exposure happened in the last 72 hours, contact a clinician, urgent care, or an "
        "emergency room right away and ask about PEP. PEP works best when started as soon as possible.",
}
DEFAULT_URGENT_NOTE = "Please contact your HIV care provider today, or go to urgent care if you cannot reach them."
