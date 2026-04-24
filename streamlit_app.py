import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="💬 Chatbot", page_icon="💬")

# -----------------------------
# Config
# -----------------------------
APP_PASSWORD = st.secrets["APP_PASSWORD"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
VECTOR_STORE_ID = st.secrets["VECTOR_STORE_ID"]

client = OpenAI(api_key=OPENAI_API_KEY)

# -----------------------------
# Helpers
# -----------------------------
def check_password() -> bool:
    """Simple password gate using Streamlit session state."""
    if st.session_state.get("authenticated", False):
        return True

    st.title("🔒 Protected Chat App")
    password = st.text_input("Password", type="password")

    if st.button("Enter"):
        if password == APP_PASSWORD:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password.")

    return False


def extract_text_input(messages):
    """Convert chat history into Responses API input format."""
    return [
        {
            "role": m["role"],
            "content": [{"type": "input_text", "text": m["content"]}],
        }
        for m in messages
        if m["role"] in {"user", "assistant"}
    ]


def stream_response_and_results(stream, results_placeholder):
    """
    Yield assistant text for st.write_stream, while also updating a side
    placeholder with file search results as streaming events arrive.
    """
    collected_text = []
    search_results = []
    seen_results = set()

    for event in stream:
        event_type = getattr(event, "type", None)

        # Stream assistant text
        if event_type == "response.output_text.delta":
            delta = getattr(event, "delta", "")
            if delta:
                collected_text.append(delta)
                yield delta

        # Capture file search results if present on output items
        elif event_type == "response.output_item.done":
            item = getattr(event, "item", None)
            if not item:
                continue

            if getattr(item, "type", None) == "file_search_call":
                results = getattr(item, "results", None) or []
                for r in results:
                    file_id = getattr(r, "file_id", None)
                    filename = getattr(r, "filename", None)
                    score = getattr(r, "score", None)
                    text = getattr(r, "text", None)

                    key = (file_id, filename, text)
                    if key in seen_results:
                        continue
                    seen_results.add(key)

                    result_obj = {
                        "file_id": file_id,
                        "filename": filename,
                        "score": score,
                        "text": text,
                    }
                    search_results.append(result_obj)

                if search_results:
                    md = "### Retrieved search results\n"
                    for i, r in enumerate(search_results, start=1):
                        snippet = (r["text"] or "").strip()
                        if len(snippet) > 300:
                            snippet = snippet[:300] + "..."
                        md += (
                            f"**{i}. {r['filename'] or 'Unknown file'}**"
                            f"  \nScore: {r['score']}"
                            f"  \nSnippet: {snippet}\n\n"
                        )
                    results_placeholder.markdown(md)

        elif event_type == "error":
            err = getattr(event, "error", None)
            raise RuntimeError(str(err) if err else "Streaming error")

    st.session_state.last_search_results = search_results


# -----------------------------
# Password gate
# -----------------------------
if not check_password():
    st.stop()

# -----------------------------
# App UI
# -----------------------------
st.title("💬 Chatbot")
st.write(
    "This chatbot is password-protected, uses a server-side OpenAI API key, "
    "streams responses, and shows retrieved file search results in the chat."
)

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "message_results" not in st.session_state:
    st.session_state.message_results = []

# Replay chat history
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            prior_results = (
                st.session_state.message_results[i]
                if i < len(st.session_state.message_results)
                else []
            )
            if prior_results:
                with st.expander("Retrieved search results", expanded=False):
                    for j, r in enumerate(prior_results, start=1):
                        snippet = (r.get("text") or "").strip()
                        if len(snippet) > 300:
                            snippet = snippet[:300] + "..."
                        st.markdown(
                            f"**{j}. {r.get('filename') or 'Unknown file'}**  \n"
                            f"Score: {r.get('score')}  \n"
                            f"Snippet: {snippet}"
                        )

# Chat input
if prompt := st.chat_input("Ask something about your files"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.message_results.append([])

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        results_placeholder = st.empty()

        stream = client.responses.create(
			model="gpt-5.4",
			input=extract_text_input(st.session_state.messages),
			stream=True,
			include=["file_search_call.results"],
			max_output_tokens=32768,
			reasoning={"effort": "none"},
			temperature=0,
			tool_choice="required",
			tools=[{
				"type": "file_search",
				"vector_store_ids": [VECTOR_STORE_ID],
			}],
			top_p=1,
		)

        response_text = st.write_stream(
            stream_response_and_results(stream, results_placeholder)
        )

    st.session_state.messages.append(
        {"role": "assistant", "content": response_text}
    )
    st.session_state.message_results.append(
        st.session_state.get("last_search_results", [])
    )