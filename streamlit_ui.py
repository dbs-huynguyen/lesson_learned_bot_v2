import re
import uuid
from typing import Any, Generator

import streamlit as st
from langgraph_sdk import get_sync_client


API_URL = "http://localhost:2024"

client = get_sync_client(url=API_URL)

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
    client.threads.create(
        thread_id=st.session_state.thread_id,
        graph_id="chat",
        if_exists="do_nothing",
        ttl={"strategy": "delete", "ttl": 1},
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_sources" not in st.session_state:
    st.session_state.selected_sources = {}

if "sidebar_state" not in st.session_state:
    st.session_state.sidebar_state = "collapsed"

if "selected_source_idx" not in st.session_state:
    st.session_state.selected_source_idx = None

st.set_page_config(
    page_title="InsightBot – Bài Học & Kinh Nghiệm",
    page_icon="🤖",
    initial_sidebar_state=st.session_state.sidebar_state,
)

st.title("🤖 Kho Kinh Nghiệm Kỹ Thuật")


def toggle_citations(idx, msg) -> None:
    if st.session_state.selected_source_idx != idx:
        st.session_state.selected_source_idx = idx
        st.session_state.selected_sources = msg["documents"]
        st.session_state.sidebar_state = "expanded"
        return

    if st.session_state.selected_source_idx == idx:
        # Open
        if st.session_state.sidebar_state == "collapsed":
            st.session_state.sidebar_state = "expanded"
            st.session_state.selected_sources = msg["documents"]
        # Close
        else:
            st.session_state.sidebar_state = "collapsed"
            st.session_state.selected_sources = {}
        st.session_state.selected_source_idx = idx
        return


for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "ai" and msg["documents"]:
            if st.button("📚 Xem nguồn trích dẫn", key=f"src_{i}", on_click=toggle_citations, args=(i, msg)):
                st.rerun()


def reset_thread() -> None:
    st.session_state.messages.clear()
    st.session_state.selected_sources = {}
    client.threads.delete(thread_id=st.session_state.thread_id)
    st.session_state.thread_id = str(uuid.uuid4())
    client.threads.create(
        thread_id=st.session_state.thread_id,
        graph_id="chat",
        if_exists="do_nothing",
        ttl={"strategy": "delete", "ttl": 1},
    )

def stream_data(prompt) -> Generator[Any, Any, None]:
    for chunk in client.runs.stream(
        thread_id=st.session_state.thread_id,
        assistant_id="chat",
        input={"messages": [{"role": "human", "content": prompt}]},
        stream_mode="messages-tuple",
    ):
        if chunk.event == "messages":
            content = chunk.data[0].get("content", "")
            yield content

if prompt := st.chat_input("Nhập câu hỏi..."):
    st.session_state.messages.append({"role": "human", "content": prompt})
    with st.chat_message("human"):
        st.markdown(prompt)

    try:
        with st.chat_message("ai"):
            answer = st.write_stream(stream_data(prompt), cursor="|")

        matches = re.findall(r'\[(\d+)\].*?filename:\s*.*?\]', answer)
        matches = [int(id) for id in matches]

        state = client.threads.get_state(thread_id=st.session_state.thread_id)
        documents = {i: item for i, item in enumerate(state["values"].get("documents", []), 1)}
        documents = {k: v for k, v in documents.items() if k in matches}

    except Exception as e:
        answer = f"❌ Error: {e}"
        documents = {}

    st.session_state.messages.append({"role": "ai", "content": answer, "documents": documents})
    st.rerun()

with st.sidebar:
    st.button(
        "Reset thread",
        on_click=reset_thread,
        help="Xóa toàn bộ lịch sử hội thoại và bắt đầu lại.",
        type="secondary",
        icon="🔄️",
        width="stretch",
    )

    st.title("📖 Nguồn trích dẫn")
    if st.session_state.selected_sources:
        for i, src in st.session_state.selected_sources.items():
            metadata = src.get("metadata", {})
            page_content = src.get("page_content", "")
            st.title(f"[{i}] {metadata.get('source', 'Unknown')}")
            st.markdown(f"{page_content}")
            st.divider()
    else:
        st.markdown("> Chọn 'Xem nguồn trích dẫn' để hiển thị")
        st.divider()
