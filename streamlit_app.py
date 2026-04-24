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
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    stream = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages
        ],
        stream=True,
    )

    with st.chat_message("assistant"):
        response = st.write_stream(stream)

    st.session_state.messages.append({"role": "assistant", "content": response})
