import re
from uuid import uuid4
from typing import Any
from html import escape

import streamlit as st
from streamlit import components
from st_checkbox_tree import checkbox_tree
from langgraph_sdk import get_sync_client

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="InsightBot – Bài Học & Kinh Nghiệm",
    page_icon="🤖",
    initial_sidebar_state=st.session_state.get("sidebar_state", "auto"),
    layout="centered",
)


# ── Langgraph client ──────────────────────────────────────────────────────────
@st.cache_resource
def get_client():
    return get_sync_client(url="http://localhost:2024", timeout=120)


client = get_client()


@st.cache_data
def get_document_types() -> list[dict[str, Any]]:
    final_state_of_run = client.runs.wait(
        thread_id=None,
        assistant_id="list_files",
        input={},
    )
    return final_state_of_run.get("nodes") or []


if "document_types" not in st.session_state:
    st.session_state.document_types = get_document_types()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "documents" not in st.session_state:
    st.session_state.documents = {}

if "selected_sources" not in st.session_state:
    st.session_state.selected_sources = []

if "sidebar_state" not in st.session_state:
    st.session_state.sidebar_state = "auto"

if "selected_source_idx" not in st.session_state:
    st.session_state.selected_source_idx = None

if "partial_response" not in st.session_state:
    st.session_state.partial_response = ""

if "is_streaming" not in st.session_state:
    st.session_state.is_streaming = False

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid4())

if "run_id" not in st.session_state:
    st.session_state.run_id = None


def toggle_citations(msg) -> None:
    if st.session_state.selected_source_idx != msg["id"]:
        st.session_state.selected_source_idx = msg["id"]
        st.session_state.selected_sources = [
            st.session_state.documents[src]
            for src in msg["sources"]
            if src in st.session_state.documents
        ]
        st.session_state.sidebar_state = "expanded"
        return

    if st.session_state.selected_source_idx == msg["id"]:
        # Open
        if st.session_state.sidebar_state == "collapsed":
            st.session_state.sidebar_state = "expanded"
            st.session_state.selected_sources = [
                st.session_state.documents[src]
                for src in msg["sources"]
                if src in st.session_state.documents
            ]
        # Close
        else:
            st.session_state.sidebar_state = "collapsed"
            st.session_state.selected_sources = []
        st.session_state.selected_source_idx = msg["id"]
        return


def repl_citation(match, sources: list[str]) -> str:
    for i, src in enumerate(sources, 1):
        if match.group(2) == src:
            return f'<a href="#" data-link="{src}">[{i}]</a>'
            # return f"[[{i}]]({src})"
    return f'<a href="#" data-link="{src}">[?]</a>'
    # return f"[[?]]({src})"


st.title("ChatDBS", text_alignment="center")


inline_links = st.components.v2.component(
    "inline_links",
    js="""
    export default function(component) {
        const { setTriggerValue } = component;
        
        // Use MutationObserver to watch for dynamically added links
        const observer = new MutationObserver(() => {
            const links = document.querySelectorAll('a[href="#"][data-link]');
            links.forEach((link) => {
                if (!link.dataset.listenerAttached) {
                    link.dataset.listenerAttached = 'true';
                    link.onclick = (e) => {
                        e.preventDefault();
                        const source = link.dataset.link;
                        setTriggerValue('clicked', source);
                    };
                }
            });
        });
        
        observer.observe(document.body, { childList: true, subtree: true });
        
        // Initial scan
        const links = document.querySelectorAll('a[href="#"][data-link]');
        links.forEach((link) => {
            link.dataset.listenerAttached = 'true';
            link.onclick = (e) => {
                e.preventDefault();
                const source = link.dataset.link;
                setTriggerValue('clicked', source);
            };
        });
    }
    """,
)

# Handle citation link clicks
citation_result = inline_links(on_clicked_change=lambda: None)
if citation_result.clicked:
    # Find the message containing this source
    clicked_source = citation_result.clicked
    for msg in reversed(st.session_state.messages):
        if msg["role"] == "assistant" and clicked_source in msg.get("sources", []):
            st.session_state.selected_source_idx = msg["id"]
            st.session_state.selected_sources = [
                st.session_state.documents[src]
                for src in msg["sources"]
                if src in st.session_state.documents
            ]
            st.session_state.sidebar_state = "expanded"
            break
    st.rerun()


with st.sidebar:
    st.markdown(
        """
        <div style="padding: 1.5rem; border: 1px solid #333; border-radius: 8px; background-color: #1a1a1a;">
            <h2 style="font-size: 1.2rem;">Hướng dẫn sử dụng</h2>
            <ol style="padding-left: 0.5rem; color: #8e8ea0; font-size: 0.9rem;">
                <li>Nhập câu hỏi hoặc yêu cầu của bạn vào ô chat bên dưới.</li>
                <li>ChatDBS sẽ trả lời dựa trên kiến thức được cung cấp.</li>
                <li>Bạn có thể hỏi về chủ đề kinh nghiệm lập trình hoặc tài liệu ISO.</li>
                <li>Để có câu trả lời tốt nhất, hãy cố gắng đặt câu hỏi rõ ràng và cụ thể.</li>
                <li>Nếu muốn dừng phản hồi đang được tạo ra, hãy nhấn nút "Dừng tạo nội dung".</li>
            </ol>
            <small style="padding-left: 0.5rem; color: #8e8ea0; font-size: 0.9rem;"><i>Lưu ý: ChatDBS có thể mắc lỗi. Hãy kiểm tra thông tin quan trọng.</i></small>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.header("📄 Chọn loại tài liệu Q&A", divider="blue")
    st.radio(
        "Chọn loại tài liệu",
        key="task",
        options=["BHKN", "ISO"],
        captions=["Bài học kinh nghiệm", "Tài liệu IMS"],
        horizontal=True,
        label_visibility="collapsed",
    )
    if st.session_state.task == "ISO":
        checked = checkbox_tree(
            nodes=list(
                filter(lambda x: x["value"] == "ISO", st.session_state.document_types)
            ),
            check_model="leaf",
            show_tree_lines=True,
            tree_line_color="green",
            show_expand_all=True,
        )
    else:
        pass
    st.header("📌 Nguồn trích dẫn", divider="blue")
    for i, item in enumerate(st.session_state.selected_sources, 1):
        with st.expander(f"Tài liệu {i}", icon="🔥"):
            st.pills(
                "info",
                options=[
                    f"{item['metadata']['source']}",
                    f"{item['metadata']['section']}",
                    f"{item['metadata']['occurred_at']}",
                ],
                disabled=True,
                label_visibility="collapsed",
            )
            st.caption(item["page_content"])


# ── Greeting (shown when no messages) ────────────────────────────────────────
if not st.session_state.messages:
    st.markdown(
        """
        <div class="greeting" style="margin-top: 5rem; color: #8e8ea0;">
            <p style="font-size: 1.2rem; margin: 0;">Tôi có thể giúp gì cho bạn?</p>
            <small>Hỏi bất cứ điều gì – tôi luôn sẵn sàng lắng nghe.</small>
        </div>
        """,
        unsafe_allow_html=True,
        text_alignment="center",
    )


# ── Render conversation ───────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(
            msg["content"],
            unsafe_allow_html=True if msg["role"] == "assistant" else False,
        )
        if msg["role"] == "assistant" and len(msg.get("sources") or []) > 0:
            st.button(
                "_Xem nguồn trích dẫn_",
                key=msg["id"],
                type="secondary",
                icon=":material/document_search:",
                on_click=toggle_citations,
                args=(msg,),
            )


# ── Stop button (shown only while streaming) ──────────────────────────────────
stop_slot = st.empty()
if st.session_state.is_streaming:
    with stop_slot.container():
        c1, c2, c3 = st.columns([4, 3, 4])
        with c2:
            if st.button(
                "Dừng tạo nội dung",
                icon=":material/stop_circle:",
                use_container_width=True,
            ):
                client.runs.cancel(
                    thread_id=st.session_state.thread_id,
                    run_id=st.session_state.run_id,
                    wait=True,
                )
                st.session_state.is_streaming = False
                if st.session_state.partial_response:
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": f"{st.session_state.partial_response} *(đã dừng)*",
                        }
                    )
                    st.session_state.partial_response = ""
                st.rerun()


# ── Chat input ────────────────────────────────────────────────────────────────
prompt = st.chat_input(
    "Nhắn tin cho ChatGPT...", disabled=st.session_state.is_streaming
)

if prompt and not st.session_state.is_streaming:
    # Save user message and kick off streaming
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.is_streaming = True
    st.session_state.partial_response = ""
    st.rerun()


def typing_indicator(text: str, **kwargs):
    safe_text = escape(text)
    spinner = st.components.v2.component(
        name="loading_spinner",
        html=f"""
        <div class="wrapper">
            <div class="spinner"></div>
            <div class="label">{safe_text}</div>
        </div>
        """,
        css="""
        .wrapper {
            display: flex;
            align-items: center;
            gap: 10px;

            color: var(--st-text-color);
            font-family: var(--st-font);
            font-size: 1rem;
        }

        .spinner {
            width: 18px;
            height: 18px;

            border-radius: 50%;

            border: 2px solid var(--st-border-color);
            border-top-color: var(--st-text-color);

            animation: spin 0.8s linear infinite;
        }

        @keyframes spin {
            to {
                transform: rotate(360deg);
            }
        }
        """,
    )
    spinner(**kwargs)


# ── Streaming logic ───────────────────────────────────────────────────────────
if (
    st.session_state.is_streaming
    and st.session_state.messages
    and st.session_state.messages[-1]["role"] == "user"
):
    with st.chat_message("assistant"):
        loading = st.empty()
        placeholder = st.empty()
        msg_id = None
        full_text = ""

        with loading:
            typing_indicator("ChatDBS đang suy nghĩ...")

        try:
            stream = client.runs.stream(
                thread_id=st.session_state.thread_id,
                if_not_exists="create",
                assistant_id="chat",
                stream_mode=["messages-tuple"],
                stream_subgraphs=True,
                input={"messages": [st.session_state.messages[-1]]},
            )

            for chunk in stream:
                if chunk.event == "metadata":
                    print("Run ID:", chunk.data)
                    st.session_state.run_id = chunk.data["run_id"]
                    continue  # skip metadata events

                msg, metadata = chunk.data

                # Skip summarization messages
                if msg.get("additional_kwargs", {}).get("lc_source") == "summarization":
                    continue

                msg_id = msg["id"]
                full_text += msg["content"]
                st.session_state.partial_response = full_text
                loading.empty()
                placeholder.markdown(full_text, unsafe_allow_html=True)
        except Exception as e:
            if not full_text:
                full_text = f"⚠️ Lỗi: {e}"

        sources = list(
            dict.fromkeys(
                re.findall(r"\[([A-Za-z0-9_]+\.[A-Za-z]+#page=\d+)\]", full_text)
            )
        )

        full_text = re.sub(
            r"(\[)([A-Za-z0-9_]+\.[A-Za-z]+#page=\d+)(\])",
            lambda m: repl_citation(m, sources),
            full_text,
        )

        placeholder.markdown(full_text, unsafe_allow_html=True)
        state = client.threads.get_state(thread_id=st.session_state.thread_id)
        documents = state["values"].get("documents") or []
        st.session_state.documents = documents

        st.session_state.messages.append(
            {
                "id": msg_id,
                "role": "assistant",
                "content": full_text,
                "sources": sources,
            }
        )

    st.session_state.is_streaming = False
    st.session_state.partial_response = ""
    stop_slot.empty()
    st.rerun()

