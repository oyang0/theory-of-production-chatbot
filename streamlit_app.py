import hmac

import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="Book research", page_icon="📚")
st.title("📚 Book research")

# Explicit, conservative capability configuration. Update only after checking
# the current model documentation for the model you deploy.
MODEL = "gpt-6-sol"
EFFORTS = ["none", "low", "medium", "high", "xhigh", "max"]

INSTRUCTIONS = """
You are a research assistant working with the uploaded books.
Use file_search to investigate the user's question. Give a comprehensive,
well-organized answer grounded in retrieved material. Cite the relevant
retrieved files for substantive claims. Distinguish what the books establish
from your interpretation, outside knowledge, and uncertainty. If retrieval
does not support an answer, say so. Never invent quotations, page numbers,
passages, or citations. Page markers in a retrieved excerpt may help locate
text, but do not claim an exact page unless the retrieved text establishes it.
"""


def secret(name: str) -> str | None:
    try:
        value = st.secrets.get(name)
    except Exception:
        return None
    return value if isinstance(value, str) and value.strip() else None


app_password = secret("APP_PASSWORD")
api_key = secret("OPENAI_API_KEY")
vector_store_id = secret("OPENAI_VECTOR_STORE_ID")

if not app_password or not api_key or not vector_store_id:
    st.error(
        "App configuration is incomplete. Ask the operator to set "
        "APP_PASSWORD, OPENAI_API_KEY, and OPENAI_VECTOR_STORE_ID in "
        ".streamlit/secrets.toml."
    )
    st.stop()

# A keyed widget is necessary for Streamlit input, but its value is never
# copied into messages or sent to OpenAI.
entered_password = st.text_input("App password", type="password", key="app_password")
if not entered_password:
    st.info("Enter the app password to continue.")
    st.stop()
if not hmac.compare_digest(entered_password, app_password):
    st.error("Incorrect app password.")
    st.stop()

# No client or chat controls exist before successful authentication.
client = OpenAI(api_key=api_key, timeout=900.0)

mode = st.selectbox("Reasoning mode", ["standard", "pro"])
effort = st.selectbox("Reasoning effort", EFFORTS, index=0)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "response_id" not in st.session_state:
    st.session_state.response_id = None

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            st.caption("Cited files: " + "; ".join(message["sources"]))

prompt = st.chat_input("Ask about the uploaded books")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        # Check readiness at query time as well as during ingestion. This
        # prevents queries against a store with pending or failed files.
        files = list(
            client.vector_stores.files.list(vector_store_id=vector_store_id)
        )
        if not files:
            st.error("The configured vector store contains no indexed books.")
            st.stop()
        unfinished = [f for f in files if f.status != "completed"]
        if unfinished:
            st.error(
                "Book indexing is not complete. Ask the operator to check "
                "the ingestion command and retry when all files are completed."
            )
            st.stop()

        request = {
            "model": MODEL,
            "instructions": INSTRUCTIONS,
            "input": prompt,
            "reasoning": {"effort": effort},
            "text": {"verbosity": "high"},
            "temperature": 0 if effort == "none" else 1,
            "tools": [
                {
                    "type": "file_search",
                    "vector_store_ids": [vector_store_id],
                    "max_num_results": 50,
                }
            ],
            "include": ["file_search_call.results"],
            "service_tier": "flex",
        }
        if st.session_state.response_id:
            request["previous_response_id"] = st.session_state.response_id

        response = client.with_options(timeout=900.0).responses.create(**request)

        search_calls = [
            item for item in response.output
            if item.type == "file_search_call"
        ]
        results = [
            result
            for call in search_calls
            for result in (call.results or [])
        ]
        if not results:
            answer = (
                "No book passages were retrieved for this question. "
                "Try a more specific query or ask the operator to check indexing."
            )
            sources = []
        else:
            answer = response.output_text or (
                "The search returned passages, but the model produced no answer."
            )
            # Annotations are the API's citation records. Do not infer pages
            # from filenames or from the model's prose.
            sources = []
            for item in response.output:
                if item.type != "message":
                    continue
                for part in item.content:
                    if part.type != "output_text":
                        continue
                    for annotation in part.annotations or []:
                        if annotation.type == "file_citation":
                            label = (
                                f"{annotation.filename} "
                                f"(OpenAI file ID: {annotation.file_id})"
                            )
                            if label not in sources:
                                sources.append(label)

        with st.chat_message("assistant"):
            st.markdown(answer)
            if sources:
                st.caption("Cited files: " + "; ".join(sources))
            elif results:
                st.warning(
                    "Passages were retrieved, but the answer has no file "
                    "citation annotations. Verify it against the books."
                )

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )
        st.session_state.response_id = response.id

    except Exception:
        # Avoid displaying exception text: SDK errors can contain request
        # details. Operators should inspect protected server logs instead.
        st.error(
            "The research request failed. Check API access, model support, "
            "vector-store configuration, and server logs."
        )
