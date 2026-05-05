import re
import uuid
from typing import Any, Generator

from langgraph_sdk import get_sync_client
import streamlit as st
from st_checkbox_tree import checkbox_tree


API_URL = "http://localhost:2024"

client = get_sync_client(url=API_URL, timeout=30)


if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
    client.threads.create(
        thread_id=st.session_state.thread_id,
        graph_id="chat",
        if_exists="do_nothing",
        ttl={"strategy": "delete", "ttl": 1},
    )

if "document_types" not in st.session_state:
    st.session_state.document_types = []
    for chunk in client.runs.stream(
        thread_id=None,
        assistant_id="list_files",
        input={},
        stream_mode=["values"],
    ):
        if chunk.event == "values":
            st.session_state.document_types = chunk.data.get("nodes")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_sources" not in st.session_state:
    st.session_state.selected_sources = []

if "sidebar_state" not in st.session_state:
    st.session_state.sidebar_state = "collapsed"

if "selected_source_idx" not in st.session_state:
    st.session_state.selected_source_idx = None

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

if "is_streaming" not in st.session_state:
    st.session_state.is_streaming = False

if "metadata_filter" not in st.session_state:
    st.session_state.metadata_filter = None

if "task" not in st.session_state:
    st.session_state.task = None

st.set_page_config(
    page_title="InsightBot – Bài Học & Kinh Nghiệm",
    page_icon="🤖",
    initial_sidebar_state=st.session_state.sidebar_state,
)


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
            st.session_state.selected_sources = []
        st.session_state.selected_source_idx = idx
        return


def reset_thread() -> None:
    st.session_state.messages.clear()
    st.session_state.selected_sources = []
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
        input={
            "messages": [{"role": "human", "content": prompt}],
            "metadata_filter": st.session_state.metadata_filter,
            "task": st.session_state.task,
        },
        stream_mode=["messages-tuple"],
        stream_subgraphs=True,
    ):
        if re.search(r"^messages\|(?:answer_with_rag|answer)", chunk.event):
            message, metadata = chunk.data
            import pprint
            if "langgraph_node" in metadata and metadata["langgraph_node"] == "model":
                # pprint.pprint(metadata)

                if message.get("content") is None:
                    continue

                yield message["content"]


def repl_link(match, urls: dict[str, list[str]]) -> str:
    link_mask = match.group(0)
    if link := urls.get(link_mask):
        return "[{title}]({url})".format(title=link[0], url=link[1])
    return link_mask


def get_documents(state, answer: str) -> list[dict[str, Any]]:
    matches = re.findall(r'\[(.*#page=\d+)\]', answer)
    documents = state["values"].get("documents") or []
    doc_map = {f"{doc['metadata']['source']}#page={doc['metadata']['page_number']}": doc for doc in documents}
    return [doc_map[src] for src in matches if src in doc_map]


st.title("🤖 Kho Kinh Nghiệm Kỹ Thuật")

for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "ai" and msg["documents"]:
            st.button(
                "Xem nguồn trích dẫn",
                key=f"src_{i}",
                on_click=toggle_citations,
                type="secondary",
                icon=":material/document_search:",
                args=(i, msg),
            )

prompt = st.chat_input("Nhập câu hỏi...", disabled=st.session_state.is_streaming)

if prompt:
    st.session_state.is_streaming = True
    st.session_state.pending_prompt = prompt
    st.rerun()

if st.session_state.get("is_streaming") and st.session_state.get("pending_prompt"):
    prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None

    st.session_state.selected_sources = []
    st.session_state.messages.append({"role": "human", "content": prompt})

    with st.chat_message("human"):
        st.markdown(prompt)

    try:
        with st.chat_message("ai"):
            answer = st.write_stream(stream_data(prompt), cursor="|")
            st.session_state.is_streaming = False

            state = client.threads.get_state(thread_id=st.session_state.thread_id)
            documents = get_documents(state, answer)

            # urls: dict[str, list[str]] = {
            #     k: v
            #     for doc in documents.values()
            #     for k, v in (doc["metadata"].get("urls") or {}).items()
            # }
            # answer = re.sub(r"#[a-zA-Z0-9_]+", lambda m: repl_link(m, urls), answer)

    except Exception as e:
        answer = f"❌ Error: {e}"
        documents = {}

    st.session_state.messages.append(
        {
            "role": "ai",
            "content": answer,
            "documents": documents,
        }
    )
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

    st.title("📖 Chọn loại tài liệu Q&A")
    st.session_state.task = st.radio(
        "Chọn loại tài liệu",
        options=["BHKN", "ISO"],
        horizontal=True,
        label_visibility="collapsed",
    )
    if st.session_state.task == "ISO":
        nodes = list(
            filter(lambda x: x["value"] == "ISO", st.session_state.document_types)
        )
        checked = checkbox_tree(
            nodes,
            check_model="leaf",
            show_tree_lines=True,
            show_expand_all=True,
        )
        st.write("Đã chọn ISO: ", checked["checked"])
        st.session_state.metadata_filter = {"source": checked["checked"]}
    else:
        st.write("Đã chọn BHKN (tất cả)")
        st.session_state.metadata_filter = {"doc_type": "BHKN"}

    st.divider()

    st.title("📖 Nguồn trích dẫn")
    if st.session_state.selected_sources:
        for item in st.session_state.selected_sources:
            st.subheader(f"{item['metadata']['source']}#page={item['metadata']['page_number']}")
            st.subheader(item['metadata'].get('section'))
            st.markdown(item["page_content"])
            st.divider()
    else:
        st.markdown("> Chọn 'Xem nguồn trích dẫn' để hiển thị")
        st.divider()
