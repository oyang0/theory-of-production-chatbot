import streamlit as st
from openai import OpenAI

# Show title and description.
st.title("💬 Chatbot")
st.write(
    "This is a simple chatbot app. "
    "Please enter the app password to continue."
)

# Initialize auth state.
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# If not authenticated, ask for password.
if not st.session_state.authenticated:
    password = st.text_input("App Password", type="password")

    if password:
        if password == st.secrets["APP_PASSWORD"]:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    else:
        st.info("Please enter the app password to continue.", icon="🔒")

else:
    # Create an OpenAI client using the server-side secret.
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "last_response_id" not in st.session_state:
        st.session_state.last_response_id = None


    def normalize_annotation(annotation):
        data = {"type": getattr(annotation, "type", "annotation")}
        for field in [
            "index",
            "file_id",
            "filename",
            "quote",
            "start_index",
            "end_index",
            "title",
            "url",
        ]:
            value = getattr(annotation, field, None)
            if value is not None:
                data[field] = value
        return data


    def normalize_search_result(result):
        return {
            "file_id": getattr(result, "file_id", None),
            "filename": getattr(result, "filename", "Unknown file"),
            "score": getattr(result, "score", None),
            "text": getattr(result, "text", None),
        }


    def render_search_results(results):
        if not results:
            return

        with st.expander("Search results", expanded=False):
            for result in results:
                filename = result.get("filename", "Unknown file")
                score = result.get("score")
                text = result.get("text")

                if score is not None:
                    st.markdown(f"**{filename}** — score: `{score:.3f}`")
                else:
                    st.markdown(f"**{filename}**")

                if text:
                    st.caption(text)


    def render_annotations(annotations):
        if not annotations:
            return

        with st.expander("Annotations", expanded=False):
            for ann in annotations:
                ann_type = ann.get("type", "annotation")

                if ann_type == "file_citation":
                    filename = ann.get("filename", "Unknown file")
                    index = ann.get("index")
                    quote = ann.get("quote")

                    if index is not None:
                        st.markdown(f"**File citation [{index}]**: `{filename}`")
                    else:
                        st.markdown(f"**File citation**: `{filename}`")

                    if quote:
                        st.caption(quote)

                elif ann_type == "url_citation":
                    title = ann.get("title", "Link")
                    url = ann.get("url", "")
                    st.markdown(f"**URL citation**: [{title}]({url})")

                else:
                    st.json(ann)


    def render_message(message):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant":
                render_search_results(message.get("search_results", []))
                render_annotations(message.get("annotations", []))


    def build_request(prompt, previous_response_id=None):
        request = {
            "model": "gpt-5.4",
            "input": prompt,
            "stream": True,
            "include": ["file_search_call.results"],
            "max_output_tokens": 32768,
            "reasoning": {"effort": "none"},
            "temperature": 0,
            "tool_choice": "required",
            "tools": [{
                "type": "file_search",
                "vector_store_ids": ["vs_69eb029ecadc8191b1c321b5d58f1958"],
            }],
        }

        if previous_response_id:
            request["previous_response_id"] = previous_response_id

        return request


    def stream_and_collect(stream):
        full_text = ""
        search_results = []
        annotations = []
        response_id = None
        seen_results = set()

        for event in stream:
            event_type = getattr(event, "type", "")

            if event_type == "response.created":
                response = getattr(event, "response", None)
                if response is not None:
                    response_id = getattr(response, "id", None)

            elif event_type == "response.output_text.delta":
                delta = getattr(event, "delta", "")
                if delta:
                    full_text += delta
                    yield delta

            elif event_type == "response.file_search_call.completed":
                results = getattr(event, "results", None) or []
                for result in results:
                    normalized = normalize_search_result(result)
                    key = (
                        normalized["file_id"],
                        normalized["filename"],
                        normalized["score"],
                        normalized["text"],
                    )
                    if key not in seen_results:
                        seen_results.add(key)
                        search_results.append(normalized)

            elif event_type == "response.output_item.done":
                item = getattr(event, "item", None)
                if not item:
                    continue

                contents = getattr(item, "content", None) or []
                for content in contents:
                    if getattr(content, "type", None) == "output_text":
                        for ann in getattr(content, "annotations", None) or []:
                            annotations.append(normalize_annotation(ann))

            elif event_type == "response.completed":
                response = getattr(event, "response", None)
                if response is not None and response_id is None:
                    response_id = getattr(response, "id", None)

        stream_and_collect.full_text = full_text
        stream_and_collect.search_results = search_results
        stream_and_collect.annotations = annotations
        stream_and_collect.response_id = response_id


    # Render existing conversation
    for message in st.session_state.messages:
        render_message(message)


    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        render_message({"role": "user", "content": prompt})

        request = build_request(
            prompt=prompt,
            previous_response_id=st.session_state.last_response_id,
        )

        with st.chat_message("assistant"):
            stream = client.responses.create(**request)
            response_text = st.write_stream(stream_and_collect(stream))

            current_search_results = getattr(stream_and_collect, "search_results", [])
            current_annotations = getattr(stream_and_collect, "annotations", [])
            current_response_id = getattr(stream_and_collect, "response_id", None)

            render_search_results(current_search_results)
            render_annotations(current_annotations)

        assistant_message = {
            "role": "assistant",
            "content": response_text,
            "search_results": current_search_results,
            "annotations": current_annotations,
            "response_id": current_response_id,
        }

        st.session_state.messages.append(assistant_message)
        st.session_state.last_response_id = current_response_id
