"""Offline tests. They use MockLLM and a tiny test corpus, so no Google Cloud access is needed.

Run:  pytest -q
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

os.environ["CAREAI_BACKEND"] = "mock"

from careai.config import load_config  # noqa: E402
from careai.ingest import Document, chunk_document, html_to_text  # noqa: E402
from careai.llm import MockLLM  # noqa: E402
from careai.orchestrator import CareAI  # noqa: E402
from careai.safety import rule_check  # noqa: E402
from careai.triage import rule_triage  # noqa: E402
from careai.vector_store import VectorStore  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# Small test-only corpus. It is not part of the real knowledge base.
FIXTURE = [
    ("treatment", "HIV treatment",
     "HIV treatment involves taking medicine that reduces the amount of HIV in the body. The medicine is "
     "called antiretroviral therapy or ART. People with HIV should take HIV medicine every day as prescribed. "
     "Taking HIV medicine as prescribed helps people with HIV live long and healthy lives. Talk to your health "
     "care provider about any side effects you have from your HIV medicine."),
    ("basics", "Viral load",
     "Viral load is the amount of HIV in a sample of blood. An undetectable viral load means the amount of HIV "
     "in the blood is too low to be measured by a test. People with HIV who get and keep an undetectable viral "
     "load will not transmit HIV to their sexual partners. Getting and keeping an undetectable viral load is the "
     "best thing people with HIV can do to stay healthy."),
    ("mental-health", "Mental health",
     "Many people living with HIV feel stress, sadness, fear, or anxiety after a diagnosis. These feelings are "
     "common. Talking with a counselor, a support group, or your HIV care team can help. Good mental health "
     "helps people stay in care and keep taking their HIV medicine every day as prescribed."),
    ("travel", "Traveling with HIV",
     "When traveling, bring enough HIV medicine for the whole trip plus a few extra days. Keep medicine in your "
     "carry-on bag. Some countries have entry rules for people with HIV, so check before you travel and talk "
     "with your health care provider about vaccines you may need for the trip."),
]


def make_store(llm) -> VectorStore:
    chunks = []
    for topic, title, text in FIXTURE:
        doc = Document(url=f"https://example.test/{topic}", title=title, publisher="TEST", topic=topic, text=text)
        chunks += [c.to_dict() for c in chunk_document(doc, size=120, overlap=20)]
    vectors = llm.embed([c["text"] for c in chunks])
    return VectorStore(vectors, chunks)


@pytest.fixture()
def bot():
    llm = MockLLM()
    cfg = json.loads(json.dumps(load_config()))  # copy so tests do not change the cached config
    cfg["rag"]["min_score"] = 0.15  # hashing embeddings give lower scores than real ones
    return CareAI(llm=llm, store=make_store(llm), config=cfg)


# ---------- Ingestion ----------
def test_html_cleaning_removes_navigation():
    html = ("<html><head><title>Test Page</title></head><body><nav>Menu Home About</nav>"
            "<main><h1>Treatment</h1><p>Take HIV medicine every day.</p></main>"
            "<footer>Footer links</footer></body></html>")
    title, text = html_to_text(html)
    assert title == "Test Page"
    assert "Take HIV medicine every day." in text
    assert "Menu" not in text and "Footer" not in text


def test_chunking_overlap_and_ids():
    doc = Document("https://example.test/a", "A", "TEST", "basics", " ".join(f"word{i}" for i in range(500)))
    chunks = chunk_document(doc, size=200, overlap=50)
    assert len(chunks) == 3
    assert chunks[0].text.split()[-50:] == chunks[1].text.split()[:50]
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_vector_store_round_trip(tmp_path):
    llm = MockLLM()
    store = make_store(llm)
    store.save(tmp_path)
    loaded = VectorStore.load(tmp_path)
    assert len(loaded) == len(store)
    hit = loaded.search(llm.embed(["undetectable viral load"])[0], top_k=1)[0]
    assert hit.chunk["topic"] == "basics"


# ---------- Triage ----------
@pytest.mark.parametrize("text,level", [
    ("I can't breathe and my throat is swelling", "emergency"),
    ("I have chest pain since this morning", "emergency"),
    ("I want to end my life", "crisis"),
    ("I don't want to live anymore", "crisis"),
    ("The condom broke last night", "urgent"),
    ("I started a new HIV medicine and now I have a rash", "urgent"),
    ("What does undetectable mean?", "routine"),
    ("How do I travel with my pills?", "routine"),
])
def test_rule_triage(text, level):
    assert rule_triage(text).level == level


# ---------- Output safety ----------
def test_rule_check_blocks_medication_instruction():
    assert not rule_check("You should stop taking your medicine. [1]", 1).passed


def test_rule_check_requires_valid_citations():
    assert not rule_check("ART lowers viral load.", 1).passed
    assert not rule_check("ART lowers viral load. [3]", 1).passed
    assert rule_check("ART lowers viral load. [1]", 1).passed


# ---------- End-to-end workflow ----------
def test_emergency_skips_generation(bot):
    res = bot.answer("My throat is closing and I can't breathe")
    assert res.triage_level == "emergency"
    assert "911" in res.answer
    assert [s["step"] for s in res.trace] == ["triage"]


def test_crisis_message(bot):
    res = bot.answer("I want to kill myself")
    assert res.triage_level == "crisis" and "988" in res.answer


def test_out_of_scope(bot):
    res = bot.answer("Give me a recipe for chocolate cake")
    assert res.intent == "out_of_scope"


def test_grounded_answer_with_citation(bot):
    res = bot.answer("What does an undetectable viral load mean?")
    assert res.triage_level == "routine"
    assert res.intent == "hiv_information"
    assert res.citations, res.trace
    assert res.citations[0].url == "https://example.test/basics"
    steps = [s["step"] for s in res.trace]
    assert steps == ["triage", "route", "retrieve", "generate", "safety_check"]


def test_medication_route(bot):
    res = bot.answer("What should I know about taking my HIV medicine every day?")
    assert res.intent == "medication"
    assert res.citations


def test_urgent_note_is_added(bot):
    res = bot.answer("The condom broke with my partner, what do I do?")
    assert res.triage_level == "urgent"
    assert "PEP" in res.answer


def test_no_evidence_fallback(bot):
    res = bot.answer("zzqx plorb vintage synthesizer wiring")
    assert "could not find" in res.answer.lower()
    assert not res.citations


# ---------- Tuning data format ----------
def test_tuning_example_format():
    import sys
    sys.path.insert(0, str(ROOT))
    from tuning.build_dataset import to_vertex_example

    ex = to_vertex_example("medication", "[1] TEST | T | u\ntext", "Q?", "A. [1]")
    assert ex["systemInstruction"]["parts"][0]["text"]
    assert [c["role"] for c in ex["contents"]] == ["user", "model"]
    assert ex["contents"][1]["parts"][0]["text"] == "A. [1]"
