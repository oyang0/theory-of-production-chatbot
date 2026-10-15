import hmac
import re

import streamlit as st
from openai import OpenAI

MODEL = "gpt-6-sol"
MAX_SEARCH_ROUNDS = 3
MAX_RESULTS_PER_SEARCH = 50
SCORE_THRESHOLD = 0.25
MAX_GROUNDING_CHARS = 131_072

# A page boundary is two newlines, [PAGE_NUMBER] plus 48 hyphens,
# then two newlines. The number labels the page that follows it.
PAGE_BOUNDARY = re.compile(r"\n\n(\d+)-{48}\n\n")
CITATION = re.compile(r"\[\[([A-Za-z0-9_-]+)\]\]")


def secret(name):
    try:
        value = st.secrets.get(name)
    except Exception:
        return None
    return value if isinstance(value, str) and value.strip() else None


def page_label(text):
    """Return a page only if the retrieved text establishes it unambiguously."""
    matches = list(PAGE_BOUNDARY.finditer(text))
    if len(matches) != 1:
        return "page unknown"

    # A chunk spanning a boundary contains text from more than one page.
    # Only a boundary at the start can label all substantive text in the chunk.
    if text[: matches[0].start()].strip():
        return "page unknown"
    return f"page {matches[0].group(1)}"


def result_field(result, name, default=None):
    if isinstance(result, dict):
        return result.get(name, default)
    return getattr(result, name, default)


def collect_chunks(response, chunks, seen):
    """Collect only chunks whose text is exposed in file-search results."""
    for item in response.output:
        if getattr(item, "type", None) != "file_search_call":
            continue
        for result in getattr(item, "results", None) or []:
            text = result_field(result, "text")
            if not isinstance(text, str) or not text.strip():
                continue

            filename = result_field(result, "filename") or "Unknown file"
            file_id = result_field(result, "file_id") or ""
            key = (file_id, filename, text)
            if key in seen:
                continue

            seen.add(key)
            chunks.append(
                {
                    "id": f"B{len(chunks) + 1}",
                    "filename": filename,
                    "text": text,
                    "page": page_label(text),
                }
            )


def conversation_context(messages):
    # Keep the full visible conversation, including earlier answers, so
    # follow-up questions can refer to prior turns.
    return "\n\n".join(
        f"{message['role'].upper()}: {message['content']}"
        for message in messages
    )


def search_collection(client, vector_store_id, messages):
    chunks = []
    seen = set()
    context = conversation_context(messages)

    research_instructions = (
        "You are researching a collection of books. Search relevant material "
        "thoroughly, including passages that might qualify or conflict with an "
        "initial finding. Distinguish the authors' claims from your analysis. "
        "Do not assume one search is exhaustive. Return a brief research plan "
        "or findings; the application will separately ground the final answer "
        "in the retrieved chunks."
    )

    response = client.with_options(timeout=900.0).responses.create(
        model=MODEL,
        instructions=research_instructions,
        input=context,
        reasoning={"effort": "none"},
        text={"verbosity": "high"},
        temperature=0,
        max_output_tokens=32768,
        tools=[
            {
                "type": "file_search",
                "vector_store_ids": [vector_store_id],
                "max_num_results": MAX_RESULTS_PER_SEARCH,
                "ranking_options": {"score_threshold": SCORE_THRESHOLD},
            }
        ],
        include=["file_search_call.results"],
        service_tier="flex",
    )
    collect_chunks(response, chunks, seen)

    # Follow-up searches use the question and initial findings to look for
    # corroboration, qualifications, and conflicting passages. These are
    # additional retrieval attempts, not a guarantee of exhaustive coverage.
    question = messages[-1]["content"]
    findings = response.output_text or ""
    for focus in (
        "Find additional passages directly relevant to the question, "
        "including definitions, context, and other parts of the books.",
        "Find passages that qualify, contradict, or offer alternative "
        "interpretations of the initial findings.",
    )[: MAX_SEARCH_ROUNDS - 1]:
        follow_up = client.responses.create(
            model=MODEL,
            instructions=(
                "Use file_search to investigate the specified focus. "
                "Search the collection rather than relying only on the "
                "initial findings. Keep your response brief."
            ),
            input=(
                f"Conversation:\n{context}\n\n"
                f"Initial findings:\n{findings[:8_000]}\n\n"
                f"Research focus: {focus}\n"
                f"Current question: {question}"
            ),
            reasoning={"effort": "none"},
            text={"verbosity": "high"},
            temperature=0,
            max_output_tokens=32768,
            tools=[
                {
                    "type": "file_search",
                    "vector_store_ids": [vector_store_id],
                    "max_num_results": MAX_RESULTS_PER_SEARCH,
                    "ranking_options": {"score_threshold": SCORE_THRESHOLD},
                }
            ],
            include=["file_search_call.results"],
            service_tier="flex",
        )
        collect_chunks(follow_up, chunks, seen)

    return chunks


def grounding_material(chunks):
    """Fit complete returned chunks into the explicit grounding prompt."""
    selected = []
    used = 0
    for chunk in chunks:
        entry = (
            f"ID: {chunk['id']}\n"
            f"Filename: {chunk['filename']}\n"
            f"Page: {chunk['page']}\n"
            f"Retrieved chunk text:\n{chunk['text']}\n"
        )
        if used + len(entry) > MAX_GROUNDING_CHARS:
            continue
        selected.append(chunk)
        used += len(entry)
    return selected


def validate_answer(answer, chunks):
    valid_ids = {chunk["id"] for chunk in chunks}
    cited_ids = set(CITATION.findall(answer))
    if not cited_ids.issubset(valid_ids):
        return False

    # Reject malformed citation-like tokens rather than displaying them.
    without_valid_citations = CITATION.sub("", answer)
    if "[[" in without_valid_citations or "]]" in without_valid_citations:
        return False

    return bool(cited_ids)


def grounded_answer(client, messages, chunks):
    selected = grounding_material(chunks)
    if not selected:
        return (
            "I couldn't access enough retrieved chunk text to give a "
            "verifiable answer. Please try a narrower question.",
            [],
        )

    sources = "\n\n".join(
        f"ID: {chunk['id']}\n"
        f"Filename: {chunk['filename']}\n"
        f"Page: {chunk['page']}\n"
        f"Text:\n{chunk['text']}"
        for chunk in selected
    )
    instructions = (
        "Answer the user's latest question using the supplied retrieved "
        "chunks as evidence. Conduct a careful synthesis across relevant "
        "chunks. Distinguish what an author says from your own analysis; "
        "reconcile conflicting passages where possible and explain remaining "
        "uncertainty. If the evidence is insufficient, say so. A retrieved "
        "chunk may be only part of a page: never infer missing text, a page "
        "number, a quotation, or a locator. Treat a page-boundary delimiter "
        "as a boundary, not as proof that an incomplete chunk contains a "
        "whole page.\n\n"
        "CITATION FORMAT: Put [[B1]], [[B2]], etc. immediately after the "
        "sentence or clause each chunk supports. Use only IDs supplied below. "
        "For multiple supporting chunks, write adjacent markers such as "
        "[[B1]][[B2]]. Cite substantive claims about the books. Do not cite "
        "a chunk that does not support the claim. Do not use file-level "
        "citation markers or invent IDs. If a claim lacks support, qualify "
        "or omit it. Do not put citations inside code fences."
    )
    prompt = (
        f"Conversation:\n{conversation_context(messages)}\n\n"
        f"Retrieved chunks (the only citable sources):\n{sources}"
    )

    for _ in range(2):
        response = client.responses.create(
            model=MODEL,
            instructions=instructions,
            input=prompt,
            reasoning={"effort": "none"},
            text={"verbosity": "high"},
            temperature=0,
            max_output_tokens=32768,
            service_tier="flex",
        )
        answer = (response.output_text or "").strip()
        if validate_answer(answer, selected):
            cited = set(CITATION.findall(answer))
            return answer, [chunk for chunk in selected if chunk["id"] in cited]

        prompt += (
            "\n\nYour previous answer had missing, invalid, or malformed "
            "chunk citations. Rewrite it using only the supplied IDs in "
            "the exact [[B1]] format, adjacent to supported claims."
        )

    return (
        "I found retrieved material, but couldn't produce an answer with "
        "validated block-level citations. Please try a narrower question.",
        [],
    )


def show_message(message):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            for chunk in message["sources"]:
                with st.expander(f"{chunk['id']} — {chunk['filename']} ({chunk['page']})"):
                    st.text(chunk["text"])


st.title("📚 Research chatbot")

expected_password = secret("APP_PASSWORD")
api_key = secret("OPENAI_API_KEY")
vector_store_id = secret("OPENAI_VECTOR_STORE_ID")

if not all((expected_password, api_key, vector_store_id)):
    st.error(
        "App configuration is incomplete. Set APP_PASSWORD, OPENAI_API_KEY, "
        "and OPENAI_VECTOR_STORE_ID in Streamlit secrets."
    )
    st.stop()

password = st.text_input("App password", type="password")
if not password:
    st.info("Enter the app password to continue.")
    st.stop()

if not hmac.compare_digest(password, expected_password):
    st.error("Incorrect app password.")
    st.stop()

# Authentication has succeeded; only now create the client and show the chat.
client = OpenAI(api_key=api_key, timeout=900.0)
st.caption("Ask questions about the books in the existing research collection.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    show_message(message)

if prompt := st.chat_input("Ask a research question"):
    pending_messages = st.session_state.messages + [
        {"role": "user", "content": prompt}
    ]
    show_message(pending_messages[-1])

    try:
        with st.spinner("Searching and checking sources…"):
            chunks = search_collection(client, vector_store_id, pending_messages)
            answer, sources = grounded_answer(client, pending_messages, chunks)
    except Exception:
        st.error(
            "The research request failed. Please try again. "
            "No assistant response was saved."
        )
    else:
        st.session_state.messages = pending_messages
        assistant_message = {
            "role": "assistant",
            "content": answer,
            "sources": sources,
        }
        st.session_state.messages.append(assistant_message)
        show_message(assistant_message)
