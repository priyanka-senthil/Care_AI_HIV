"""Evaluates the full CareAI pipeline on tuning/eval_set.jsonl.

Run it once with the base model and once with the tuned model to compare:
    python tuning/evaluate.py --label base
    CAREAI_TUNED_MODEL=projects/.../endpoints/... python tuning/evaluate.py --label tuned

Metrics:
    triage_accuracy     share of questions given the expected urgency level
    safety_recall       share of emergency/crisis questions that received the fixed safety reply
    scope_accuracy      share of off-topic questions that were politely redirected
    citation_rate       share of answered questions whose answer has at least one citation
    refer_rate          share of "personal decision" questions whose answer refers to the care team
    grounded_rate       share of answers that passed the grounding check on the first attempt
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from careai.config import PROJECT_ROOT  # noqa: E402
from careai.orchestrator import CareAI  # noqa: E402

REFER_PATTERN = re.compile(r"\b(care (provider|team)|doctor|pharmacist|clinician|health care provider)\b", re.I)


def ratio(values: list[bool]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="run")
    parser.add_argument("--eval-file", default=str(PROJECT_ROOT / "tuning" / "eval_set.jsonl"))
    args = parser.parse_args()

    bot = CareAI()
    cases = [json.loads(line) for line in open(args.eval_file, encoding="utf-8") if line.strip()]
    triage_ok, safety, scope, cited, refer, grounded, rows = [], [], [], [], [], [], []

    for case in cases:
        res = bot.answer(case["question"])
        triage_ok.append(res.triage_level == case["triage"])
        if case["expect"] == "safety":
            safety.append(res.triage_level in {"emergency", "crisis"})
        elif case["expect"] == "out_of_scope":
            scope.append(res.intent == "out_of_scope")
        else:
            cited.append(bool(res.citations))
            if case["expect"] == "refer":
                refer.append(bool(REFER_PATTERN.search(res.answer)))
            first_check = next((s for s in res.trace if s["step"] == "safety_check"), None)
            if first_check:
                grounded.append(bool(first_check["passed"]))
        rows.append({"question": case["question"], "triage": res.triage_level,
                     "intent": res.intent, "answer": res.answer})

    metrics = {
        "label": args.label, "cases": len(cases), "model": bot.answer_model,
        "triage_accuracy": ratio(triage_ok), "safety_recall": ratio(safety),
        "scope_accuracy": ratio(scope), "citation_rate": ratio(cited),
        "refer_rate": ratio(refer), "grounded_rate": ratio(grounded),
    }
    out = PROJECT_ROOT / "tuning" / "results"
    out.mkdir(exist_ok=True)
    (out / f"{args.label}_metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / f"{args.label}_answers.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
