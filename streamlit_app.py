import hashlib
import hmac
import re

import streamlit as st
from openai import OpenAI


MODEL = "gpt-6-sol"

SUPPORTED_SETTINGS = {
    "standard": {"none", "low", "medium", "high", "xhigh", "max"},
    "pro": {"none", "low", "medium", "high", "xhigh", "max"},
}

PAGE_MARKER_RE = re.compile(r"\n\n\[(\d+)\]-{48}\n\n")

RESEARCH_INSTRUCTIONS = """
You are a research analyst working primarily from the uploaded Markdown books.

For substantive research questions, use file_search. Give comprehensive,
carefully reasoned answers grounded in retrieved evidence. Distinguish what
the books say from your own interpretation or synthesis. Identify material
gaps, uncertainty, and disagreements between sources.

Cite the relevant retrieved files close to the claims they support. Include
short, relevant quotations when the retrieved text lets you verify their
exact wording and attribution. Never invent a quotation, page number, line
number, or source.

The books use page-boundary markers consisting of two newlines, a bracketed
page number, 48 hyphens, and two newlines. These are metadata, not prose.
A retrieved excerpt may cross a boundary. Give a page number only when the
retrieved evidence unambiguously establishes that the cited passage is on
that page. Otherwise cite the file without a page number. If the available
retrieved text cannot verify a quotation, paraphrase instead.

Use the file-search citation annotations supplied by the API. Do not invent
citation markers or claim that a citation links to a particular page.
"""


def required_secret(name: str) -> str:
    try:
        value = st.secrets[name]
    except (KeyError, FileNotFoundError):
        st.error(
            f"Missing configuration: {name}. Add it to "
            "./.streamlit/secrets.toml and restart the app."
        )
        st.stop()

    if not isinstance(value, str) or not value.strip():
        st.error(f"Configuration error: {name} must be a non-empty string.")
        st.stop()

    return value


def session_fingerprint(password: str, vector_store_id: str) -> str:
    """Identify this authenticated session without storing its password."""
    return hashlib.sha256(
        (password + "\0" + vector_store_id).encode("utf-8")
    ).hexdigest()


def response_text_and_sources(response):
    """Extract answer text and API-provided file citation annotations."""
    text_parts = []
    sources = []
    seen = set()

    for item in response.output:
        if getattr(item, "type", None) != "message":
            continue

        for content in getattr(item, "content", []):
            if getattr(content, "type", None) != "output_text":
                continue

            text_parts.append(content.text)
            for annotation in getattr(content, "annotations", []) or []:
                if getattr(annotation, "type", None) != "file_citation":
                    continue

                file_id = getattr(annotation, "file_id", None)
                filename = getattr(annotation, "filename", None)
                key = (file_id, filename)
                if key not in seen:
                    seen.add(key)
                    sources.append(
                        {
                            "file_id": file_id,
                            "filename": filename or "Untitled file",
                        }
                    )

    return "\n".join(text_parts).strip(), sources


def display_answer(text: str, sources: list[dict]):
    # File-search citation markers are API presentation tokens, not prose.
    # Streamlit Markdown does not resolve them into clickable file citations.
    # Remove only the known file-citation token shape; retain the API's
    # annotation metadata in the source list below.
    clean_text = re.sub(r"filecite[^]+", "", text)
    st.markdown(clean_text)

    if sources:
        st.caption(
            "Cited files: "
            + "; ".join(
                f"{source['filename']} ({source['file_id']})"
                for source in sources
            )
        )
        st.caption(
            "These citations identify uploaded files. This app cannot link "
            "directly to a book page from the file-search annotations."
        )


def create_research_response(
    client: OpenAI,
    messages: list[dict],
    vector_store_id: str,
    mode: str,
    effort: str,
):
    if effort not in SUPPORTED_SETTINGS.get(mode, set()):
        raise ValueError("Unsupported research mode and reasoning effort.")

    # Pro mode is deliberately blocked above. Do not substitute an effort
    # value for the API's actual reasoning-mode parameter.
    return client.responses.create(
        model=MODEL,
        instructions=RESEARCH_INSTRUCTIONS,
        input=messages,
        reasoning={"effort": effort},
        tools=[
            {
                "type": "file_search",
                "vector_store_ids": [vector_store_id],
            }
        ],
        include=["file_search_call.results"],
    )


st.title("📚 Book-Grounded Research Chat")
st.write(
    "Ask detailed questions about the uploaded books. Answers use hosted "
    "file search and identify the files supporting their claims."
)

app_password = required_secret("APP_PASSWORD")
api_key = required_secret("OPENAI_API_KEY")
vector_store_id = required_secret("VECTOR_STORE_ID")

entered_password = st.text_input("App password", type="password")

if not entered_password:
    st.info("Enter the app password to begin.")
    st.stop()

if not hmac.compare_digest(entered_password, app_password):
    st.error("Incorrect password.")
    st.stop()

mode = st.selectbox("Reasoning mode", ["standard", "pro"])
effort = st.selectbox(
    "Reasoning effort",
    ["none", "low", "medium", "high", "xhigh", "max"],
    index=2,
)

fingerprint = session_fingerprint(entered_password, vector_store_id)
settings = (MODEL, mode, effort, vector_store_id)

if (
    st.session_state.get("research_session") != fingerprint
    or st.session_state.get("research_settings") != settings
):
    st.session_state.research_session = fingerprint
    st.session_state.research_settings = settings
    st.session_state.messages = []

if effort not in SUPPORTED_SETTINGS.get(mode, set()):
    if mode == "pro":
        st.warning(
            "Pro mode is unavailable in this version of the app. Its exact "
            "Responses API parameter and supported efforts must be verified "
            "against the current SDK documentation before requests can be sent."
        )
    else:
        st.warning(
            f"{MODEL} does not support reasoning effort '{effort}' in the "
            "configured compatibility table. Choose a supported effort."
        )
    st.stop()

client = OpenAI(api_key=api_key)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            display_answer(message["content"], message.get("sources", []))
        else:
            st.markdown(message["content"])

if prompt := st.chat_input("Ask a research question about the books"):
    with st.chat_message("user"):
        st.markdown(prompt)

    request_messages = [
        {"role": message["role"], "content": message["content"]}
        for message in st.session_state.messages
    ]
    request_messages.append({"role": "user", "content": prompt})

    try:
        with st.spinner("Searching the books and preparing an answer…"):
            response = create_research_response(
                client, request_messages, vector_store_id, mode, effort
            )
            answer, sources = response_text_and_sources(response)
            if not answer:
                raise RuntimeError("The API returned no answer text.")
    except Exception:
        # Do not display exception text: it may contain request details.
        st.error(
            "The research request failed. Check the app configuration, "
            "model access, and vector-store availability in server logs."
        )
    else:
        st.session_state.messages.append(
            {"role": "user", "content": prompt}
        )
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": sources,
            }
        )
        with st.chat_message("assistant"):
            display_answer(answer, sources)
