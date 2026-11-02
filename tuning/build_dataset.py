"""Builds the supervised fine-tuning dataset for Gemini on Vertex AI.

Why fine-tune when we already use RAG?
RAG supplies the facts. Fine-tuning teaches the BEHAVIOR: answering only from evidence, citing
every fact as [n], refusing when the evidence does not cover the question, never giving personal
medication instructions, and writing in plain, person-first language.

How the data is made:
1. "Grounded" examples: for each knowledge-base chunk, Gemini Pro (the teacher) writes a realistic
   patient question and an ideal answer that follows the CareAI rules.
2. "Safety" examples: the teacher writes a question that asks for a personal medication decision,
   with an answer that gives general facts and refers to the care team.
3. "Unanswerable" examples: a question is paired with evidence from an unrelated page, and the target
   is the standard "I could not find this" reply. This teaches the model not to guess.

Output (Vertex AI supervised tuning JSONL format):
    tuning/data/train.jsonl, tuning/data/validation.jsonl, tuning/data/review.csv

IMPORTANT: this is health content. Read review.csv and remove or fix weak examples before tuning.

Usage:
    python tuning/build_dataset.py --max-chunks 300
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from careai import agents  # noqa: E402
from careai.config import PROJECT_ROOT, load_config  # noqa: E402
from careai.llm import get_llm  # noqa: E402
from careai.retriever import format_evidence  # noqa: E402
from careai.safety import rule_check  # noqa: E402
from careai.vector_store import SearchHit, VectorStore  # noqa: E402

OUT_DIR = PROJECT_ROOT / "tuning" / "data"

TEACHER_SCHEMA = {
    "type": "object",
    "properties": {
        "question": {"type": "string"},
        "answer": {"type": "string"},
        "intent": {"type": "string", "enum": ["hiv_information", "medication", "emotional_support", "care_navigation"]},
    },
    "required": ["question", "answer", "intent"],
}

TEACHER_PROMPT = """You are creating training data for CareAI, an assistant for people living with HIV.

Follow these rules for the ANSWER:
{rules}

Using ONLY the evidence below, write:
- "question": one realistic question a person living with HIV might ask, in their own words.
  {question_style}
- "answer": the ideal CareAI answer. Cite the evidence as [1].
- "intent": the best category for the question.

EVIDENCE:
{evidence}"""

QUESTION_STYLES = {
    "grounded": "The evidence must fully answer the question.",
    "safety": ("The question should ask for a personal medical decision (for example whether to stop, "
               "skip, or change a medicine, or what their own symptom or lab result means). The answer must "
               "share only general facts from the evidence and clearly refer the decision to their HIV "
               "care provider or pharmacist."),
}


def to_vertex_example(intent: str, evidence: str, question: str, answer: str) -> dict:
    return {
        "systemInstruction": {"role": "system", "parts": [{"text": agents.system_instruction(intent)}]},
        "contents": [
            {"role": "user", "parts": [{"text": agents.build_answer_prompt(evidence, question)}]},
            {"role": "model", "parts": [{"text": answer}]},
        ],
    }


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-chunks", type=int, default=300)
    parser.add_argument("--safety-share", type=float, default=0.25, help="share of chunks used for safety examples")
    parser.add_argument("--unanswerable-share", type=float, default=0.15)
    parser.add_argument("--validation-share", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    random.seed(args.seed)
    store = VectorStore.load(PROJECT_ROOT / cfg["rag"]["index_dir"])
    llm = get_llm()
    teacher = cfg["models"]["generator"]

    chunks = list(store.chunks)
    random.shuffle(chunks)
    chunks = chunks[: args.max_chunks]

    examples, review_rows = [], []
    for i, chunk in enumerate(chunks):
        kind = "safety" if random.random() < args.safety_share else "grounded"
        evidence = format_evidence([SearchHit(1.0, chunk)])
        prompt = TEACHER_PROMPT.format(rules=agents.BASE_RULES, question_style=QUESTION_STYLES[kind], evidence=evidence)
        try:
            item = llm.generate_json(prompt, TEACHER_SCHEMA, model=teacher)
        except Exception as exc:
            print(f"  skip chunk {i}: {exc}")
            continue
        if not item.get("question") or not item.get("answer"):
            continue
        if not rule_check(item["answer"], 1).passed:  # drop answers that break the rules
            continue
        intent = item.get("intent", "hiv_information")
        examples.append(to_vertex_example(intent, evidence, item["question"], item["answer"]))
        review_rows.append([kind, chunk["url"], item["question"], item["answer"]])

        # Unanswerable: same question, unrelated evidence from a different page
        if random.random() < args.unanswerable_share:
            others = [c for c in store.chunks if c["url"] != chunk["url"] and c.get("topic") != chunk.get("topic")]
            if others:
                wrong = format_evidence([SearchHit(1.0, random.choice(others))])
                examples.append(to_vertex_example(intent, wrong, item["question"], agents.NO_EVIDENCE_MESSAGE))
                review_rows.append(["unanswerable", "(unrelated evidence)", item["question"], agents.NO_EVIDENCE_MESSAGE])
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(chunks)} chunks processed, {len(examples)} examples")

    if len(examples) < 2:
        sys.exit("Too few examples were created. Check that the index exists and the Gemini connection works.")
    random.shuffle(examples)
    n_val = max(1, int(len(examples) * args.validation_share))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, rows in (("validation", examples[:n_val]), ("train", examples[n_val:])):
        with open(OUT_DIR / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(OUT_DIR / "review.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["type", "source_url", "question", "answer"])
        writer.writerows(review_rows)

    print(f"\nWrote {len(examples) - n_val} training and {n_val} validation examples to {OUT_DIR}.")
    print("Review tuning/data/review.csv before starting the tuning job.")


if __name__ == "__main__":
    main()
