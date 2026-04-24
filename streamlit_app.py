import html
import re
import streamlit as st
from openai import OpenAI

APP_PASSWORD = st.secrets["APP_PASSWORD"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]


def check_password():
    """Returns True if the user entered the correct password."""

    if "password_correct" not in st.session_state:
        st.session_state.password_correct = False

    def password_entered():
        if st.session_state.password == APP_PASSWORD:
            st.session_state.password_correct = True
            del st.session_state.password
        else:
            st.session_state.password_correct = False

    if not st.session_state.password_correct:
        st.text_input(
            "Password",
            type="password",
            key="password",
            on_change=password_entered,
        )
        st.info("Please enter the app password to continue.", icon="🔒")
        return False

    return True


# Stop here if password is wrong or not entered yet.
if not check_password():
    st.stop()


# Show title and description.
st.title("💬 Chatbot")
st.write(
    "This is a simple chatbot app. Enter the password to access it."
)

# Create an OpenAI client using the server-side secret.
client = OpenAI(api_key=OPENAI_API_KEY)

# Create a session state variable to store the chat messages.
if "messages" not in st.session_state:
    st.session_state.messages = []

# ----------------------------
# Helpers
# ----------------------------

def get_attr(obj, name, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)

def to_dict(obj):
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return obj.__dict__ if hasattr(obj, "__dict__") else {}

def normalize_annotation(annotation):
    ann = to_dict(annotation)
    file_citation = ann.get("file_citation")
    if file_citation is not None:
        ann["file_citation"] = to_dict(file_citation)
    return ann

def normalize_result(result):
    r = to_dict(result)
    return {
        "file_id": r.get("file_id"),
        "filename": r.get("filename") or "Unknown file",
        "score": r.get("score"),
        "text": r.get("text") or "",
    }

def extract_text_annotations_and_results_from_completed(response_obj):
    """
    Fallback extractor from response.completed.
    """
    full_text = ""
    annotations = []
    search_results = []

    for item in get_attr(response_obj, "output", []) or []:
        item_type = get_attr(item, "type")

        if item_type == "file_search_call":
            for result in get_attr(item, "results", []) or []:
                search_results.append(normalize_result(result))

        elif item_type == "message":
            for content_item in get_attr(item, "content", []) or []:
                if get_attr(content_item, "type") == "output_text":
                    full_text += (
                        get_attr(content_item, "text", "")
                        or get_attr(content_item, "value", "")
                        or ""
                    )
                    for ann in get_attr(content_item, "annotations", []) or []:
                        annotations.append(normalize_annotation(ann))

    return full_text, annotations, search_results

def build_result_index(search_results):
    """
    Index retrieved chunks by file_id and filename for evidence lookup.
    """
    by_file_id = {}
    by_filename = {}

    for result in search_results or []:
        r = normalize_result(result)
        if r["file_id"]:
            by_file_id.setdefault(r["file_id"], []).append(r)
        by_filename.setdefault(r["filename"], []).append(r)

    return by_file_id, by_filename

def best_result_for_annotation(annotation, by_file_id, by_filename):
    """
    Try to match an annotation to a retrieved chunk.
    """
    fc = annotation.get("file_citation", {}) or {}
    file_id = fc.get("file_id") or annotation.get("file_id")
    filename = fc.get("filename") or annotation.get("filename")

    candidates = []
    if file_id and file_id in by_file_id:
        candidates.extend(by_file_id[file_id])
    elif filename and filename in by_filename:
        candidates.extend(by_filename[filename])

    if not candidates:
        return None

    # Prefer highest score if present
    def score_key(x):
        score = x.get("score")
        return score if isinstance(score, (int, float)) else -1

    candidates = sorted(candidates, key=score_key, reverse=True)
    return candidates[0]

def safe_tooltip_text(text, max_len=180):
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) > max_len:
        text = text[: max_len - 1] + "…"
    return html.escape(text, quote=True)

def safe_html_text(text):
    return html.escape(text or "")

def render_text_with_inline_citations(text, annotations, search_results):
    """
    Render output text as HTML with inline citation superscripts:
      cited phrase<sup title="filename — snippet">[1]</sup>

    Uses annotation character ranges when available; falls back to annotation.text replacement.
    """
    if not text:
        return "", []

    anns = [normalize_annotation(a) for a in (annotations or [])]
    by_file_id, by_filename = build_result_index(search_results)

    citation_entries = []
    citation_map = {}

    def get_citation_number(annotation):
        fc = annotation.get("file_citation", {}) or {}
        file_id = fc.get("file_id") or annotation.get("file_id")
        filename = fc.get("filename") or annotation.get("filename") or "source"
        quoted_span = annotation.get("text") or ""
        matched_result = best_result_for_annotation(annotation, by_file_id, by_filename)
        snippet = quoted_span or (matched_result.get("text") if matched_result else "") or ""

        key = (file_id, filename, quoted_span)
        if key not in citation_map:
            citation_map[key] = len(citation_entries) + 1
            citation_entries.append(
                {
                    "n": citation_map[key],
                    "file_id": file_id,
                    "filename": filename,
                    "quoted_span": quoted_span,
                    "snippet": snippet,
                    "result": matched_result,
                }
            )
        return citation_map[key]

    ranged = []
    fallback = []

    for ann in anns:
        if ann.get("type") != "file_citation":
            continue

        start_index = ann.get("start_index")
        end_index = ann.get("end_index")

        if isinstance(start_index, int) and isinstance(end_index, int):
            ranged.append(ann)
        else:
            fallback.append(ann)

    ranged.sort(key=lambda a: a["start_index"])

    html_parts = []
    cursor = 0

    for ann in ranged:
        start = ann["start_index"]
        end = ann["end_index"]

        if start < cursor or end > len(text) or start >= end:
            continue

        html_parts.append(safe_html_text(text[cursor:start]))

        cited_text = text[start:end]
        n = get_citation_number(ann)

        entry = citation_entries[n - 1]
        tooltip = safe_tooltip_text(
            f'{entry["filename"]} — {entry["snippet"]}'
        )

        html_parts.append(
            f'{safe_html_text(cited_text)}'
            f'<sup title="{tooltip}" '
            f'style="color:#6b7280;font-weight:600;cursor:help;">[{n}]</sup>'
        )
        cursor = end

    html_parts.append(safe_html_text(text[cursor:]))
    rendered_html = "".join(html_parts)

    # Fallback replacement for annotations without ranges
    for ann in fallback:
        ann_text = ann.get("text")
        if not ann_text:
            continue

        n = get_citation_number(ann)
        entry = citation_entries[n - 1]
        tooltip = safe_tooltip_text(
            f'{entry["filename"]} — {entry["snippet"]}'
        )

        replacement = (
            f'{safe_html_text(ann_text)}'
            f'<sup title="{tooltip}" '
            f'style="color:#6b7280;font-weight:600;cursor:help;">[{n}]</sup>'
        )

        rendered_html = rendered_html.replace(safe_html_text(ann_text), replacement, 1)

    return rendered_html, citation_entries

def render_evidence_panel(citations):
    if not citations:
        return

    with st.expander("Evidence", expanded=False):
        for c in citations:
            st.markdown(f"### [{c['n']}] `{c['filename']}`")

            if c.get("quoted_span"):
                st.markdown("**Quoted span in answer**")
                st.caption(c["quoted_span"])

            if c.get("snippet"):
                st.markdown("**Supporting snippet**")
                st.code(c["snippet"][:1200])

            result = c.get("result")
            if result:
                if result.get("score") is not None:
                    st.caption(f"retrieval score: {result['score']}")
                if result.get("text"):
                    st.markdown("**Retrieved chunk**")
                    st.code(result["text"][:2000])

def render_sources_list(citations):
    if not citations:
        return
    st.markdown("**Sources**")
    for c in citations:
        label = f"[{c['n']}] `{c['filename']}`"
        if c.get("result") and c["result"].get("score") is not None:
            label += f" — score: {c['result']['score']}"
        st.markdown(label)

# ----------------------------
# Render prior chat
# ----------------------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and message.get("rendered_html"):
            st.markdown(message["rendered_html"], unsafe_allow_html=True)
            render_sources_list(message.get("citations", []))
            render_evidence_panel(message.get("citations", []))
        else:
            st.markdown(message["content"])

# ----------------------------
# Chat input
# ----------------------------

if prompt := st.chat_input("Ask a question about your files"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    response_input = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
        if m["role"] in {"user", "assistant"}
    ]

    with st.chat_message("assistant"):
        answer_placeholder = st.empty()
        sources_placeholder = st.container()
        evidence_placeholder = st.container()
        search_placeholder = st.container()

        accumulated_text = ""
        annotations = []
        search_results = []

        stream = client.responses.create(
            model="gpt-5.4",
            input=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
            text={"format": {"type": "text"}},
            tool_choice="required",
            temperature=0,
            max_output_tokens=32768,
            top_p=1,
            reasoning={"effort": "none"},
            tools=[{
                "type": "file_search",
                "vector_store_ids": ["vs_69eb029ecadc8191b1c321b5d58f1958"],
            }],
            include=["file_search_call.results"],
        )

        for event in stream:
            event_type = getattr(event, "type", "")

            if event_type == "response.output_text.delta":
                delta = get_attr(event, "delta", "")
                if delta:
                    accumulated_text += delta
                    answer_placeholder.markdown(accumulated_text)

            elif event_type == "response.output_text.done":
                text_obj = get_attr(event, "text", None)
                if text_obj:
                    for ann in get_attr(text_obj, "annotations", []) or []:
                        annotations.append(normalize_annotation(ann))

            elif event_type == "response.file_search_call.in_progress":
                with search_placeholder:
                    st.caption("Searching files...")

            elif event_type == "response.file_search_call.completed":
                results = get_attr(event, "results", None)
                if results:
                    search_results = [normalize_result(r) for r in results]

            elif event_type == "response.completed":
                response_obj = get_attr(event, "response", None)
                if response_obj:
                    final_text, final_annotations, final_results = (
                        extract_text_annotations_and_results_from_completed(response_obj)
                    )
                    if final_text:
                        accumulated_text = final_text
                    if final_annotations:
                        annotations = final_annotations
                    if final_results:
                        search_results = final_results

        rendered_html, citations = render_text_with_inline_citations(
            accumulated_text,
            annotations,
            search_results,
        )

        answer_placeholder.markdown(rendered_html, unsafe_allow_html=True)

        with sources_placeholder:
            render_sources_list(citations)

        with evidence_placeholder:
            render_evidence_panel(citations)

        if search_results:
            with search_placeholder:
                with st.expander("Retrieved search results", expanded=False):
                    for i, result in enumerate(search_results, start=1):
                        st.markdown(f"**{i}. {result['filename']}**")
                        if result.get("score") is not None:
                            st.caption(f"score: {result['score']}")
                        if result.get("text"):
                            st.code(result["text"][:1200])

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": accumulated_text,
            "rendered_html": rendered_html,
            "citations": citations,
        }
    )
