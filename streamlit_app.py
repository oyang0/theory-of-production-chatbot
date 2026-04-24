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

# Display existing chat messages.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

 # Chat input.
if prompt := st.chat_input("What is up?"):
    # Store and display the user message.
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Convert chat history into Responses API input items.
    response_input = [
        {
            "role": message["role"],
            "content": [{"type": "input_text", "text": message["content"]}],
        }
        for message in st.session_state.messages
    ]

    # Stream the assistant response.
    with st.chat_message("assistant"):
        text_placeholder = st.empty()
        sources_placeholder = st.empty()

        full_response = ""
        file_search_results = []

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
            if event.type == "response.output_text.delta":
                full_response += event.delta
                text_placeholder.markdown(full_response)

            elif event.type == "response.output_item.done":
                item = event.item

                if item.type == "file_search_call":
                    results = getattr(item, "results", None) or []
                    file_search_results.extend(results)

        text_placeholder.markdown(full_response)

        # Render retrieved chunks/documents underneath
        if file_search_results:
            with sources_placeholder.container():
                with st.expander("Sources from file search"):
                    for i, result in enumerate(file_search_results, start=1):
                        filename = getattr(result, "filename", "Unknown file")
                        score = getattr(result, "score", None)
                        text = getattr(result, "text", "")

                        st.markdown(f"**{i}. {filename}**")
                        if score is not None:
                            st.caption(f"Score: {score}")

                        if text:
                            st.write(text)

                        st.divider()

    # Store assistant response in session state.
    st.session_state.messages.append(
        {"role": "assistant", "content": full_response}
    )
