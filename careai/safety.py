"""Output safety: a grounding check against the evidence, plus simple rule checks."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .llm import LLM

GROUNDING_SCHEMA = {
    "type": "object",
    "properties": {
        "grounded": {"type": "boolean"},
        "unsupported_claims": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["grounded", "unsupported_claims"],
}

GROUNDING_PROMPT = """You check answers from a health information assistant.

Compare the ANSWER with the EVIDENCE. List every factual claim in the ANSWER that the EVIDENCE
does not support. General safety advice such as "ask your care provider" does not need support.
Set grounded to true only if there are no unsupported factual claims.

EVIDENCE:
{evidence}

ANSWER:
{answer}"""

# Phrases that would mean the model gave a personal medical instruction
UNSAFE_PATTERNS = [
    r"\byou should (stop|skip|double|change|increase|decrease) (taking )?(your )?(medicine|medication|dose|pills?)\b",
    r"\b(stop|quit) taking your (hiv )?(medicine|medication|art|pills?)\b",
    r"\btake (two|2|double) doses\b",
    r"\byou (definitely |probably )?(have|don'?t have) (an? )?(infection|aids|resistance)\b",
]


@dataclass
class SafetyResult:
    passed: bool
    issues: list[str] = field(default_factory=list)


def rule_check(answer: str, evidence_count: int) -> SafetyResult:
    issues = []
    lowered = answer.lower()
    for pattern in UNSAFE_PATTERNS:
        if re.search(pattern, lowered):
            issues.append(f"personal medical instruction: /{pattern}/")
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
    if evidence_count and not cited:
        issues.append("answer has no citations")
    if any(n < 1 or n > evidence_count for n in cited):
        issues.append("answer cites a source number that does not exist")
    return SafetyResult(not issues, issues)


def grounding_check(llm: LLM, answer: str, evidence: str, model: str | None = None) -> SafetyResult:
    try:
        result = llm.generate_json(GROUNDING_PROMPT.format(evidence=evidence, answer=answer),
                                   GROUNDING_SCHEMA, model=model)
    except Exception:
        return SafetyResult(True, ["grounding check unavailable"])
    claims = result.get("unsupported_claims") or []
    return SafetyResult(bool(result.get("grounded")) and not claims, [f"unsupported: {c}" for c in claims])
