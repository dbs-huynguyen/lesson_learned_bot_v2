import streamlit as st
import requests
import uuid

API_URL = "http://localhost:2024/runs/stream"

st.title("🚀 LangGraph Streaming Chat")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# show history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# input
if prompt := st.chat_input("Ask something..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_text = ""

        try:
            with requests.post(
                API_URL,
                json={
                    "assistant_id": "chat",
                    "input": {
                        "messages": st.session_state.messages,
                    },
                    "thread_id": st.session_state.thread_id,
                },
                headers={"Content-Type": "application/json"},
                stream=True,
            ) as r:
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:
                        token = chunk.decode("utf-8")
                        full_text += token
                        placeholder.markdown(full_text + "▌")

            placeholder.markdown(full_text)

        except Exception as e:
            full_text = f"❌ Error: {e}"
            placeholder.markdown(full_text)

    st.session_state.messages.append({"role": "assistant", "content": full_text})
