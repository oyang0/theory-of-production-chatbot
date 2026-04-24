import streamlit as st
from openai import OpenAI

APP_PASSWORD = st.secrets["APP_PASSWORD"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
VECTOR_STORE_ID = st.secrets["VECTOR_STORE_ID"]

# Create the OpenAI client on the server side.
client = OpenAI(api_key=OPENAI_API_KEY)

st.title("💬 Chatbot")
st.write(
    "This chatbot uses a password to access the app. "
    "OpenAI credentials are stored securely on the server."
)

# Session state
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "messages" not in st.session_state:
    st.session_state.messages = []

def check_password():
    password = st.text_input("App password", type="password")
    if not password:
        st.info("Please enter the app password to continue.", icon="🔐")
        return False
    if password != APP_PASSWORD:
        st.error("Incorrect password.")
        return False
    return True

st.session_state.authenticated = check_password()

if st.session_state.authenticated:
    # Display prior chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    def stream_assistant_and_search():
        """
        Yields text chunks for st.write_stream while also surfacing
        file search results in the same assistant chat message.
        """
        stream = client.responses.create(
            model="gpt-5.4",
            input=[
				{"role": m["role"], "content": m["content"]}
				for m in st.session_state.messages
			],
            stream=True,
			include=["file_search_call.results"],
			max_output_tokens=32768,
			reasoning={"effort": "none"},
			temperature=0,
			tool_choice="required",
			tools=[{
				"type": "file_search",
				"vector_store_ids": ["vs_69eb029ecadc8191b1c321b5d58f1958"],
			}],
			top_p=1,
        )

        search_placeholder = st.empty()
        search_lines = []

        for event in stream:
            event_type = getattr(event, "type", "")

            # Stream assistant text output
            if event_type == "response.output_text.delta":
                delta = getattr(event, "delta", "")
                if delta:
                    yield delta

            # Surface file search results when included in the stream
            elif event_type == "response.file_search_call.completed":
                results = getattr(event, "results", None)
                if results:
                    search_lines.append("**Search results:**")
                    for i, result in enumerate(results, start=1):
                        filename = getattr(result, "filename", "Untitled")
                        score = getattr(result, "score", None)
                        text = getattr(result, "text", "") or ""

                        preview = text.strip()
                        line = f"{i}. **{filename}**"
                        if score is not None:
                            line += f" (score: {score:.3f})"
                        if preview:
                            line += f"\n   - {preview}"

                        search_lines.append(line)

                    search_placeholder.markdown("\n\n".join(search_lines))

            # Some SDK versions may emit included results on output items instead
            elif event_type == "response.output_item.done":
                item = getattr(event, "item", None)
                if item and getattr(item, "type", "") == "file_search_call":
                    results = getattr(item, "results", None)
                    if results:
                        search_lines.append("**Search results:**")
                        for i, result in enumerate(results, start=1):
                            filename = getattr(result, "filename", "Untitled")
                            score = getattr(result, "score", None)
                            text = getattr(result, "text", "") or ""

                            preview = text.strip()
                            line = f"{i}. **{filename}**"
                            if score is not None:
                                line += f" (score: {score:.3f})"
                            if preview:
                                line += f"\n   - {preview}"

                            search_lines.append(line)

                        search_placeholder.markdown("\n\n".join(search_lines))

    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response_text = st.write_stream(stream_assistant_and_search())

        st.session_state.messages.append(
            {"role": "assistant", "content": response_text}
        )