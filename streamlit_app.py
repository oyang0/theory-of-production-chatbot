import re
from pathlib import Path

import streamlit as st
from openai import OpenAI

MODEL = "gpt-6-astra"
PAGE_SEPARATOR = re.compile(r"\n\n\[PAGE_NUMBER\]-{48}\n\n")
SOURCE_DIR = Path("books")

INSTRUCTIONS = """
You are a research assistant for the uploaded books. For questions that warrant
research, search the files thoroughly, compare relevant passages, and give a
comprehensive, well-organized answer. Distinguish what the authors say from
your interpretation. Ground substantive claims about the books in retrieved
material. If the retrieved material does not support a claim, say so. Do not
claim to have searched an entire book unless the retrieval supports that claim.

Do not invent page numbers, line numbers, quotations, or citation markers.
The application will attach citations from API-provided retrieval metadata
and independently verified local Markdown passages.
"""


def required_secret(name):
    value = st.secrets.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing required Streamlit secret: {name}")
    return value.strip()


def field(obj, name, default=None):
    return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)


def source_lines(filename):
    """Return (page, line, text) records from a local Markdown source."""
    # Only use a basename supplied by the API; never follow a path from a result.
    path = SOURCE_DIR / Path(filename).name
    if not path.is_file() or path.suffix.lower() != ".md":
        return None

    text = path.read_text(encoding="utf-8")
    pages = PAGE_SEPARATOR.split(text)
    return [
        (page_number, line_number, line)
        for page_number, page in enumerate(pages, start=1)
        for line_number, line in enumerate(page.splitlines(), start=1)
    ]


def verified_location(filename, passage):
    """Locate a retrieved passage only when its text has one exact local match."""
    if not passage or not passage.strip():
        return None

    records = source_lines(filename)
    if records is None:
        return None

    # Match within a single page. Normalize whitespace, but do not use fuzzy
    # matching: a plausible match is not sufficient for a line-level citation.
    needle = " ".join(passage.split())
    matches = []
    pages = sorted({page for page, _, _ in records})
    for page in pages:
        page_records = [(line, text) for p, line, text in records if p == page]
        for start in range(len(page_records)):
            for end in range(start, len(page_records)):
                candidate = " ".join(
                    " ".join(text.split())
                    for _, text in page_records[start : end + 1]
                ).strip()
                if candidate == needle:
                    matches.append(
                        (page, page_records[start][0], page_records[end][0])
                    )
                if len(candidate) > len(needle):
                    break

    return matches[0] if len(matches) == 1 else None


def citations_from_response(response):
    """Collect API-provided file references; verify locations independently."""
    sources = {}
    for item in response.output:
        if field(item, "type") == "file_search_call":
            for result in field(item, "results", None) or []:
                file_id = field(result, "file_id")
                filename = field(result, "filename")
                passage = field(result, "text")
                if file_id and filename:
                    sources.setdefault(file_id, {"filename": filename, "passages": []})
                    if passage:
                        sources[file_id]["passages"].append(passage)

        if field(item, "type") == "message":
            for content in field(item, "content", []) or []:
                for annotation in field(content, "annotations", []) or []:
                    if field(annotation, "type") == "file_citation":
                        file_id = field(annotation, "file_id")
                        filename = field(annotation, "filename")
                        if file_id and filename:
                            sources.setdefault(
                                file_id, {"filename": filename, "passages": []}
                            )

    citations = []
    for file_id, source in sources.items():
        locations = {
            location
            for passage in source["passages"]
            if (location := verified_location(source["filename"], passage))
        }
        if locations:
            for page, first, last in sorted(locations):
                citations.append(
                    f"{source['filename']}, page {page}, lines {first}–{last}"
                )
        else:
            citations.append(f"{source['filename']} (file-level citation)")
    return citations


st.title("📚 Book Research Assistant")
st.write(
    "Ask questions about books already uploaded to the configured OpenAI vector "
    "store. Answers use file search. Page and line citations appear only when "
    "retrieved passages can be uniquely verified against local Markdown files; "
    "otherwise citations are labeled file-level."
)

try:
    app_password = required_secret("APP_PASSWORD")
    api_key = required_secret("OPENAI_API_KEY")
    vector_store_id = required_secret("OPENAI_VECTOR_STORE_ID")
except ValueError as exc:
    st.error(str(exc))
    st.stop()

password = st.text_input("App password", type="password")
if password != app_password:
    if password:
        st.error("Incorrect password.")
    st.stop()

client = OpenAI(api_key=api_key)
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        for citation in message.get("citations", []):
            st.caption(citation)

if prompt := st.chat_input("Ask about the uploaded books"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    history = [
        {"role": message["role"], "content": message["content"]}
        for message in st.session_state.messages
    ]

    try:
        with st.spinner("Searching the books…"):
            response = client.responses.create(
                model=MODEL,
                instructions=INSTRUCTIONS,
                input=history,
                tools=[
                    {
                        "type": "file_search",
                        "vector_store_ids": [vector_store_id],
                    }
                ],
                include=["file_search_call.results"],
            )
        answer = response.output_text
        citations = citations_from_response(response)
        if not answer:
            raise RuntimeError("The API returned no answer.")
    except Exception as exc:
        st.error(f"Research request failed: {exc}")
    else:
        with st.chat_message("assistant"):
            st.markdown(answer)
            for citation in citations:
                st.caption(citation)
        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "citations": citations}
        )
