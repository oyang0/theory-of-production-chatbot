import hmac
import re

import streamlit as st
from openai import OpenAI

MODEL = "gpt-6-sol"
EFFORTS = ("none", "low", "medium", "high", "xhigh", "max")
PAGE_MARKER = re.compile(r"\[PAGE_NUMBER\]\s*\n-{48}")

RESEARCH_INSTRUCTIONS = """
You are a deep-research assistant working with books in the attached vector store.
Produce a comprehensive, research-analyst-level answer grounded in relevant
retrieved passages. Distinguish what the authors claim from your own analysis.
Use short, accurate quotations when useful, and cite the files supporting
substantive claims. Do not invent quotations, page numbers, or citations.

The Markdown books use page separators consisting of two newlines,
[PAGE_NUMBER], 48 hyphens, and two newlines. Treat a page number as verified
only if it appears explicitly in retrieved text and its relationship to the
quoted passage is clear. Never infer a page number from a citation annotation,
filename, or search-result position. If a page boundary makes a quotation's
page ambiguous, cite the file without a page number. If the retrieved material
does not support an answer, say so.
"""


def get_secrets():
    names = ("APP_PASSWORD", "OPENAI_API_KEY", "VECTOR_STORE_ID")
    missing = [name for name in names if not st.secrets.get(name)]
    if missing:
        st.error("Missing required secrets: " + ", ".join(missing))
        st.stop()
    return {name: st.secrets[name] for name in names}


def validate_settings(mode, effort):
    if mode == "pro":
        return (
            f"`pro` is not a supported API mode established for {MODEL}. "
            "Select `standard` to send a request."
        )
    if effort not in EFFORTS:
        return "Select a supported reasoning effort."
    return None


def extract_answer(response):
    """Keep the API's citation annotations alongside the displayed answer."""
    parts = []
    citations = []

    for item in response.output or []:
        if getattr(item, "type", None) != "message":
            continue
        for content in getattr(item, "content", []) or []:
            if getattr(content, "type", None) != "output_text":
                continue
            text = getattr(content, "text", "") or ""
            if text:
                parts.append(text)
            for annotation in getattr(content, "annotations", []) or []:
                if getattr(annotation, "type", None) == "file_citation":
                    citations.append(
                        {
                            "file_id": getattr(annotation, "file_id", None),
                            "filename": getattr(annotation, "filename", None),
                            "index": getattr(annotation, "index", None),
                        }
                    )

    # output_text is convenient, but the content blocks above are needed
    # to retain citation metadata for later Streamlit reruns.
    answer = "\n\n".join(parts).strip() or (response.output_text or "").strip()
    unique = list(dict.fromkeys(
        (c["file_id"], c["filename"], c["index"]) for c in citations
    ))
    return answer, [
        {"file_id": file_id, "filename": filename, "index": index}
        for file_id, filename, index in unique
    ]


def show_message(message):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("citations"):
            with st.expander("Cited files"):
                for citation in message["citations"]:
                    name = citation["filename"] or "Untitled file"
                    file_id = citation["file_id"]
                    st.write(f"{name} — file ID: `{file_id}`" if file_id else name)


st.title("📚 Deep-research assistant")
st.write(
    "Ask research questions about the books in the configured knowledge base. "
    "Answers use hosted file search and retain cited-file details in this session."
)

secrets = get_secrets()

# A keyed widget is cleared after successful authentication. The password is
# never added to the conversation or sent to the model.
if not st.session_state.get("authenticated", False):
    entered = st.text_input("App password", type="password", key="app_password")
    if st.button("Unlock"):
        if hmac.compare_digest(entered, secrets["APP_PASSWORD"]):
            st.session_state.authenticated = True
            del st.session_state["app_password"]
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()

mode = st.selectbox("Reasoning mode", ("standard", "pro"))
effort = st.selectbox("Reasoning effort", EFFORTS, index=2)
settings_error = validate_settings(mode, effort)
if settings_error:
    st.warning(settings_error)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "previous_response_id" not in st.session_state:
    st.session_state.previous_response_id = None

for message in st.session_state.messages:
    show_message(message)

prompt = st.chat_input("Ask about the books", disabled=bool(settings_error))
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    show_message(st.session_state.messages[-1])

    try:
        # Construct the client only after authentication.
        client = OpenAI(api_key=secrets["OPENAI_API_KEY"], timeout=900.0)
        request = {
            "model": MODEL,
            "instructions": RESEARCH_INSTRUCTIONS,
            "input": prompt,
            "reasoning": {"effort": effort},
            "text": {"verbosity": "high"},
            "temperature": 0 if effort == "none" else 1,
            "tools": [
                {
                    "type": "file_search",
                    "vector_store_ids": [secrets["VECTOR_STORE_ID"]],
                }
            ],
            "include": ["file_search_call.results"],
            "service_tier": "flex",
        }
        if st.session_state.previous_response_id:
            request["previous_response_id"] = st.session_state.previous_response_id

        with st.spinner("Researching the books…"):
            response = client.with_options(timeout=900.0).responses.create(**request)

        if getattr(response, "status", None) == "incomplete":
            st.error("The research response was incomplete. Please try again.")
        else:
            answer, citations = extract_answer(response)
            if not answer:
                st.error("The API returned no answer. Please try again.")
            else:
                assistant_message = {
                    "role": "assistant",
                    "content": answer,
                    "citations": citations,
                }
                st.session_state.messages.append(assistant_message)
                st.session_state.previous_response_id = response.id
                show_message(assistant_message)
    except Exception as exc:
        st.error(f"Research request failed: {exc}")
