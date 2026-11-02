"""CareAI Streamlit chat app.

Run:  streamlit run app.py
Offline demo (no Google Cloud):  CAREAI_BACKEND=mock streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from careai.config import PROJECT_ROOT, backend, load_config
from careai.orchestrator import CareAI

st.set_page_config(page_title="CareAI - HIV Information", page_icon="🩺", layout="centered")

LEVEL_STYLE = {
    "routine": None,
    "urgent": ("warning", "Urgent: please contact a clinician soon."),
    "emergency": ("error", "Possible emergency"),
    "crisis": ("error", "You are not alone. Help is available now."),
}

EXAMPLES = [
    "What does undetectable mean?",
    "What should I expect at my first HIV care visit?",
    "How can I protect my partner?",
    "I feel scared about telling my family.",
]


@st.cache_resource(show_spinner="Loading CareAI...")
def load_bot() -> CareAI:
    return CareAI()


def index_exists() -> bool:
    index_dir = PROJECT_ROOT / load_config()["rag"]["index_dir"]
    return (Path(index_dir) / "vectors.npy").exists()


def render_response(res, show_trace: bool) -> None:
    style = LEVEL_STYLE.get(res.triage_level)
    if style:
        getattr(st, style[0])(style[1])
    st.markdown(res.answer)
    if res.citations:
        with st.expander("Sources"):
            for c in res.citations:
                st.markdown(f"[{c.number}] **{c.publisher}**: [{c.title}]({c.url})")
    if show_trace and res.trace:
        with st.expander("How this answer was made (agent trace)"):
            st.json(res.trace)


# ---------- Sidebar ----------
with st.sidebar:
    st.title("CareAI")
    st.caption("HIV information from trusted public health sources (CDC, NIH, HIV.gov, MedlinePlus).")
    show_trace = st.toggle("Show agent trace", value=False)
    if backend() == "mock":
        st.info("Running in offline demo mode (mock model).")
    if st.button("Start a new conversation"):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.markdown(
        "**In an emergency, call 911.**  \n"
        "**Crisis support:** call or text 988.  \n"
        "CareAI gives general information only. It does not replace your HIV care team."
    )

# ---------- Main ----------
st.header("Ask a question about HIV")

if not index_exists():
    st.error("The knowledge base has not been built yet. Run `python scripts/build_index.py` first.")
    st.stop()

bot = load_bot()
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and "response" in msg:
            render_response(msg["response"], show_trace)
        else:
            st.markdown(msg["content"])

if not st.session_state.messages:
    st.write("You can ask in your own words. For example:")
    cols = st.columns(2)
    for i, example in enumerate(EXAMPLES):
        if cols[i % 2].button(example, use_container_width=True):
            st.session_state.pending = example
            st.rerun()

question = st.chat_input("Type your question") or st.session_state.pop("pending", None)
if question:
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Checking trusted sources..."):
            try:
                response = bot.answer(question, history)
            except Exception as exc:  # show a calm message instead of a stack trace
                st.error("Something went wrong while answering. Please try again.")
                st.caption(str(exc))
                st.stop()
        render_response(response, show_trace)
    st.session_state.messages.append({"role": "assistant", "content": response.answer, "response": response})
