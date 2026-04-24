import streamlit as st
from openai import OpenAI

# Show title and description.
st.title("💬 Chatbot")
st.write(
    "This is a simple chatbot that uses OpenAI's model to generate responses. "
    "To use this app, you need an OpenAI API key configured in Streamlit secrets."
)

# Read secrets from ./.streamlit/secrets.toml
try:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
    app_password = st.secrets["APP_PASSWORD"]
    vector_store_id = st.secrets["VECTOR_STORE_ID"]
except FileNotFoundError:
    st.error(
        "Secrets file not found. Create `./.streamlit/secrets.toml` "
        "and add `OPENAI_API_KEY`, `APP_PASSWORD`, and `VECTOR_STORE_ID`."
    )
    st.stop()
except KeyError as e:
    st.error(f"Missing secret: {e}")
    st.stop()

# Ask user for the app password via st.text_input.
entered_password = st.text_input("Password", type="password")

if not entered_password:
    st.info("Please enter the app password to continue.", icon="🔐")
    st.stop()

if entered_password != app_password:
    st.error("Incorrect password.")
    st.stop()

# Create an OpenAI client.
client = OpenAI(api_key=openai_api_key)

# Create session state variables.
if "messages" not in st.session_state:
    st.session_state.messages = []

if "debug_items" not in st.session_state:
    st.session_state.debug_items = []

# Display the existing chat messages via st.chat_message.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Optionally show previous search results / annotations.
        if message["role"] == "assistant":
            if message.get("file_search_results"):
                with st.expander("Search results"):
                    st.json(message["file_search_results"])

            if message.get("annotations"):
                with st.expander("Annotations"):
                    st.json(message["annotations"])

# Create a chat input field.
if prompt := st.chat_input("What is up?"):

    # Store and display the current prompt.
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    # Containers for metadata captured during streaming.
    annotations = []
    file_search_results = []
    raw_events = []

    # Generate a response using the OpenAI API.
    stream = client.responses.create(
        model="gpt-5.5",
        input=[
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages
            if m["role"] in ("user", "assistant")
        ],
        stream=True,
        reasoning={"effort": "none"},
        temperature=0,
        max_output_tokens=32768,
        tools=[
            {
                "type": "file_search",
                "vector_store_ids": [vector_store_id],
            }
        ],
        include=["file_search_call.results"],
        tool_choice={"type": "required"},
    )

    def serialize(obj):
        """
        Best-effort helper to convert OpenAI SDK objects into JSON-serializable dicts.
        """
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        if hasattr(obj, "dict"):
            return obj.dict()
        return obj

    def write_stream():
        for event in stream:
            raw_events.append(serialize(event))

            # Stream assistant text.
            if event.type == "response.output_text.delta":
                yield event.delta

            # Capture annotations as they are added.
            elif event.type == "response.output_text.annotation.added":
                annotations.append(serialize(event.annotation))

            # Capture file search result events, if surfaced during streaming.
            elif event.type in (
                "response.file_search_call.completed",
                "response.file_search_call.in_progress",
                "response.file_search_call.searching",
            ):
                file_search_results.append(serialize(event))

    # Stream the response to the chat, then store it in session state.
    with st.chat_message("assistant"):
        response_text = st.write_stream(write_stream())

        # Write annotations to the chat.
        if annotations:
            with st.expander("Annotations"):
                st.json(annotations)

        # Write search results to the chat.
        if file_search_results:
            with st.expander("Search results"):
                st.json(file_search_results)

        # Optional: useful while developing/debugging.
        # with st.expander("Raw stream events"):
        #     st.json(raw_events)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response_text,
            "annotations": annotations,
            "file_search_results": file_search_results,
        }
    )