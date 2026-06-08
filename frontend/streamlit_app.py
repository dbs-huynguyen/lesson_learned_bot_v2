import re
from uuid import uuid4
from typing import Any

import streamlit as st
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


def repl_citation(match, sources: list[str], msg_id: str = None) -> str:
    button_style = "background: none; border: none; padding: 0; margin: 0; cursor: pointer; color: #1f77b4; text-decoration: none; font-weight: 500; font-size: inherit; font-family: inherit; display: inline;"
    msg_id = msg_id if msg_id else ""
    for i, src in enumerate(sources, 1):
        if match.group(2) == src:
            return f'<button data-link="{src}" data-msg-id="{msg_id}" style="{button_style}"><small>[{i}]</small></button>'
    return ""


st.title("ChatDBS", text_alignment="center")


inline_links = st.components.v2.component(
    "inline_links",
    js="""
    export default function(component) {
        const { setTriggerValue } = component;
        
        // Use event delegation on document to catch all clicks
        document.addEventListener('click', function(e) {
            // Find the closest button with data-link attribute
            const button = e.target.closest('button[data-link]');
            if (button) {
                e.preventDefault();
                e.stopPropagation();
                e.stopImmediatePropagation();

                const source = button.dataset.link;
                const msgId = button.dataset.msgId;

                // Trigger multiple values
                setTriggerValue('clicked', source);
                setTriggerValue('message_id', msgId);

                return false;
            }
        }, true); // Use capture phase

        // Observe for new buttons and add hover effect
        const observer = new MutationObserver(() => {
            const buttons = document.querySelectorAll('button[data-link]');
            buttons.forEach((button) => {
                if (!button.dataset.processed) {
                    button.dataset.processed = 'true';
                    // Add hover effect
                    button.addEventListener('mouseenter', () => {
                        button.style.textDecoration = 'underline';
                    });
                    button.addEventListener('mouseleave', () => {
                        button.style.textDecoration = 'none';
                    });
                }
            });
        });

        observer.observe(document.body, { childList: true, subtree: true });

        // Process existing buttons
        const buttons = document.querySelectorAll('button[data-link]');
        buttons.forEach((button) => {
            button.dataset.processed = 'true';
            button.addEventListener('mouseenter', () => {
                button.style.textDecoration = 'underline';
            });
            button.addEventListener('mouseleave', () => {
                button.style.textDecoration = 'none';
            });
        });
    }
    """,
)

# Handle citation link clicks
citation_result = inline_links(
    on_clicked_change=lambda: None,
    on_message_id_change=lambda: None,
)

if citation_result.clicked:
    clicked_source = citation_result.clicked
    message_id = citation_result.message_id

    # If we have message_id, directly use it (faster)
    if message_id:
        st.session_state.selected_source_idx = message_id
        if clicked_source in st.session_state.documents:
            st.session_state.selected_sources = [
                st.session_state.documents[clicked_source]
            ]
        st.session_state.sidebar_state = "expanded"

    st.rerun()


with st.sidebar:
    # st.markdown(
    #     """
    #     <div style="padding: 1.5rem; border: 1px solid #333; border-radius: 8px; background-color: #1a1a1a;">
    #         <h2 style="font-size: 1.2rem;">Hướng dẫn sử dụng</h2>
    #         <ol style="padding-left: 0.5rem; color: #8e8ea0; font-size: 0.9rem;">
    #             <li>Nhập câu hỏi hoặc yêu cầu của bạn vào ô chat bên dưới.</li>
    #             <li>ChatDBS sẽ trả lời dựa trên kiến thức được cung cấp.</li>
    #             <li>Bạn có thể hỏi về chủ đề kinh nghiệm lập trình hoặc tài liệu ISO.</li>
    #             <li>Để có câu trả lời tốt nhất, hãy cố gắng đặt câu hỏi rõ ràng và cụ thể.</li>
    #             <li>Nếu muốn dừng phản hồi đang được tạo ra, hãy nhấn nút "Dừng tạo nội dung".</li>
    #         </ol>
    #         <small style="padding-left: 0.5rem; color: #8e8ea0; font-size: 0.9rem;"><i>Lưu ý: ChatDBS có thể mắc lỗi. Hãy kiểm tra thông tin quan trọng.</i></small>
    #     </div>
    #     """,
    #     unsafe_allow_html=True,
    # )
    # st.header("📄 Chọn loại tài liệu Q&A", divider="blue")
    # st.radio(
    #     "Chọn loại tài liệu",
    #     key="task",
    #     options=["BHKN", "ISO"],
    #     captions=["Bài học kinh nghiệm", "Tài liệu IMS"],
    #     horizontal=True,
    #     label_visibility="collapsed",
    # )
    # if st.session_state.task == "ISO":
    #     checked = checkbox_tree(
    #         nodes=list(
    #             filter(lambda x: x["value"] == "ISO", st.session_state.document_types)
    #         ),
    #         check_model="leaf",
    #         show_tree_lines=True,
    #         tree_line_color="green",
    #         show_expand_all=True,
    #     )
    # else:
    #     pass
    st.header("📌 Nguồn trích dẫn", divider="blue")
    for i, item in enumerate(st.session_state.selected_sources, 1):
        # with st.expander(f"Tài liệu {i}", icon="🔥"):
        st.pills(
            "info",
            options=[
                f"{item['metadata']['source']}",
                f"{item['metadata']['occurred_at']}",
            ],
            disabled=True,
            label_visibility="collapsed",
        )
        st.caption(item["page_content"], unsafe_allow_html=True, text_alignment="justify")


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
        if msg["role"] == "assistant" and msg.get("reasoning_flow"):
            with st.status("Các bước suy luận", state="complete") as status:
                for reasoning in msg.get("reasoning_flow", []):
                    status.caption(reasoning.get("message", ""))

        st.markdown(
            msg["content"],
            unsafe_allow_html=True if msg["role"] == "assistant" else False,
        )
        # if msg["role"] == "assistant" and len(msg.get("sources") or []) > 0:
        #     st.button(
        #         "_Xem nguồn trích dẫn_",
        #         key=msg["id"],
        #         type="secondary",
        #         icon=":material/document_search:",
        #         on_click=toggle_citations,
        #         args=(msg,),
        #     )


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
                if st.session_state.run_id:
                    client.runs.cancel(
                        thread_id=st.session_state.thread_id,
                        run_id=st.session_state.run_id,
                        wait=True,
                    )
                    st.session_state.is_streaming = False
                    response = ""
                    if st.session_state.partial_response:
                        response = st.session_state.partial_response
                        st.session_state.partial_response = ""
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": f"{response} *(đã dừng)*",
                        }
                    )
                    st.rerun()


# ── Chat input ────────────────────────────────────────────────────────────────
prompt = st.chat_input(
    "Nhắn tin cho ChatGPT...", disabled=st.session_state.is_streaming
)

if prompt and not st.session_state.is_streaming:
    # Save user message and kick off streaming
    st.session_state.messages.append({"role": "user", "content": prompt.strip()})
    st.session_state.is_streaming = True
    st.session_state.partial_response = ""
    st.rerun()


# ── Streaming logic ───────────────────────────────────────────────────────────
if (
    st.session_state.is_streaming
    and st.session_state.messages
    and st.session_state.messages[-1]["role"] == "user"
):
    with st.chat_message("assistant"):
        status_slot = st.empty()
        loading = st.empty()
        placeholder = st.empty()
        is_first_msg_chunk = False
        is_first_custom_chunk = False
        reasoning_flow = []
        msg_id = None
        full_text = ""

        # Khởi tạo status ban đầu
        with status_slot:
            status = st.status("ChatDBS đang suy nghĩ...", state="running", expanded=True)

        try:
            stream = client.runs.stream(
                thread_id=st.session_state.thread_id,
                if_not_exists="create",
                assistant_id="chat",
                stream_mode=["messages-tuple", "custom"],
                stream_subgraphs=True,
                input={"messages": [st.session_state.messages[-1]]},
                on_run_created=lambda v: setattr(st.session_state, "run_id", v["run_id"]),
            )

            for chunk in stream:
                if chunk.event == "metadata":
                    continue  # skip metadata events

                if chunk.event.split("|")[0] == "custom":
                    if chunk.data.get("type") == "reasoning":
                        if not is_first_custom_chunk:
                            status.update(expanded=True)
                            is_first_custom_chunk = True

                        status.caption(chunk.data.get("message", ""))
                        reasoning_flow.append(chunk.data)
                    continue  # skip custom events

                if chunk.event.split("|")[0] == "messages":
                    if not is_first_msg_chunk:
                        status.update(label="Các bước suy luận", state="complete", expanded=True)
                        is_first_msg_chunk = True

                    msg, metadata = chunk.data

                    # Skip summarization messages
                    if msg.get("additional_kwargs", {}).get("lc_source") == "summarization":
                        continue

                    msg_id = msg["id"]
                    full_text += msg["content"]
                    st.session_state.partial_response = full_text
                    placeholder.markdown(full_text, unsafe_allow_html=True)
        except Exception as e:
            if not full_text:
                full_text = f"⚠️ Lỗi: {e}"

        sources = list(
            dict.fromkeys(
                re.findall(r"\[source=([A-Za-z0-9_]+\.[A-Za-z]+#page=\d+)\]", full_text)
            )
        )

        full_text = re.sub(
            r"`*(\[)source=([A-Za-z0-9_]+\.[A-Za-z]+#page=\d+)(\])`*",
            lambda m: repl_citation(m, sources, msg_id),
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
                "reasoning_flow": reasoning_flow,
            }
        )

    st.session_state.is_streaming = False
    st.session_state.partial_response = ""
    st.rerun()
