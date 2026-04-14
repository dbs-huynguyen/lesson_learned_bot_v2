import json
import requests
from typing import Any, Literal, TypedDict

import bm25s
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.messages import AnyMessage, SystemMessage, HumanMessage
from langchain_community.vectorstores import FAISS, DistanceStrategy
from langchain_ollama.chat_models import ChatOllama
from langchain_ollama.embeddings import OllamaEmbeddings
from langgraph.graph import MessagesState, StateGraph
from langmem.short_term.summarization import SummarizationNode


INITIAL_SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("placeholder", "{messages}"),
        ("user", "Summarize the above conversation in Vietnamese:"),
    ]
)


EXISTING_SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("placeholder", "{messages}"),
        (
            "user",
            "This is summary of the conversation so far: {existing_summary}\n\n"
            "Use Vietnamese to update and expand this summary based on the new messages above:",
        ),
    ]
)


FINAL_SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        # if exists
        ("placeholder", "{system_message}"),
        ("system", "{summary}"),
        ("placeholder", "{messages}"),
    ]
)


ROUTE_QUERY_PROMPT = ChatPromptTemplate.from_template(
    """You are an intent classifier.

Your task is to analyze the user's query and classify it into one of two labels:
- "DIRECT": general conversation, greetings, casual questions, or questions that can be answered without external knowledge sources
- "RAG": questions that require domain knowledge related to technology, software, systems, or internal documentation

[INSTRUCTION]
- Carefully read the user query.
- Infer the underlying intent, even if it is implicit, vague, or indirectly expressed.
- Consider context clues, technical terms, and problem-solving intent.
- If the query involves troubleshooting, system behavior, code, internal processes, or technical concepts → choose "RAG".
- If the query is conversational, generic knowledge, or does not require specialized/internal knowledge → choose "DIRECT".
- When in doubt, prioritize the deeper intent over surface wording.

[OUTPUT REQUIREMENT]
- Return ONLY a valid JSON object.
- The JSON must contain a single key "label".
- The value must be either "DIRECT" or "RAG".
- Do NOT include any explanation, comments, or extra text.

[EXAMPLE]
User: "Chào bạn"
Output: {{"label": "DIRECT"}}

User: "API bị lỗi 500 là do đâu?"
Output: {{"label": "RAG"}}

User: "Sao hệ thống chạy chậm vậy?"
Output: {{"label": "RAG"}}

User: "Hôm nay ăn gì?"
Output: {{"label": "DIRECT"}}

Query: {query}"""
)


ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful and intelligent assistant.

[INSTRUCTION]
- Always respond in Vietnamese
- Answer clearly, naturally, and easy to understand
- Be concise but still provide enough useful information
- Maintain a friendly and professional tone
- If the question is unclear, ask a follow-up question to clarify before answering
- If you don't know the answer, honestly say you don't know instead of guessing
- Avoid overly long explanations unless the user explicitly asks for details
- Avoid emojis and keep a formal tone

[BEHAVIOR]
- For simple questions: give direct answers
- For complex questions: break down the answer into clear parts
- For opinion-based questions: provide balanced and neutral perspectives
- For casual conversation: respond naturally like a human

[FORMAT]
- Use bullet points when listing information
- Use short paragraphs for readability
- Avoid unnecessary technical jargon unless required"""
)


ANSWER_WITH_RAG_PROMPT = ChatPromptTemplate.from_template(
    """You are an assistant for a private knowledge base focused on information technology, particularly software development.

[ROLE]
- You are a retrieval-based QA assistant.
- Your answers must be strictly grounded in the provided CONTEXT.

[INSTRUCTION]
- Always respond in Vietnamese.
- Only use information explicitly present in the CONTEXT to answer the question.
- Do NOT use prior knowledge, external sources, or assumptions.
- Do NOT infer or guess missing information, even if it seems obvious.
- Ignore chat history for factual answering. It is for conversational flow only and must NOT be used as a knowledge source.

[REASONING RULES]
- Carefully analyze the question and map it to relevant parts of the CONTEXT.
- Prefer exact matches and explicitly stated facts over interpretations.
- If multiple pieces of information originate from the same filename, you must merge them into a single, coherent answer.
- The merged content must strictly preserve the original meaning and must not introduce any new information.
- If there are conflicting details in the CONTEXT, present them clearly without resolving the conflict yourself.

[STRICT CONSTRAINTS]
- Do NOT fabricate, expand, or generalize beyond the CONTEXT.
- Do NOT include any knowledge not directly supported by the CONTEXT.
- Do NOT rephrase content in a way that changes its meaning.
- Do NOT answer if the supporting evidence is missing.

[LINK HANDLING RULES]
- The CONTEXT may contain strings in the format "#link_title".
- If such strings appear in the used content:
  - You MUST preserve them exactly as-is (character-by-character).
  - Do NOT translate, modify, or replace them.
  - Do NOT convert them into real URLs.
  - Any modification will make the answer invalid.

[CITATION RULES]
- You MUST include citations for all used sources.
- Citations must appear at the end of the answer.
- Use the exact format below:  
**Citations:**  
_[1] [filename:name_of_the_cited_document]_  
_[2] [filename:name_of_the_cited_document]_  

- Each cited document must correspond to information actually used in the answer.
- Do NOT cite documents that are not used.

[STYLE]
- Keep the answer concise, clear, and structured.
- Use bullet points if helpful.
- Avoid unnecessary explanations or repetition.

[OUTPUT]
- Return ONLY the final answer in Vietnamese, followed by the citations section.
- Do NOT include any meta-commentary, reasoning steps, or explanations."""
)


class StateSchema(MessagesState):
    summarized_messages: list[AnyMessage]
    documents: list[Document]
    current_query: str


class ClassificationOutput(TypedDict):
    label: Literal["DIRECT", "RAG"]


llm = ChatOllama(
    model="qwen3.5:9b",
    validate_model_on_init=True,
    base_url="http://192.168.88.179:11434",
    temperature=0,
    reasoning=False,
    num_predict=-2,
    client_kwargs={"timeout": 30},
)

embeddings = OllamaEmbeddings(
    model="qwen3-embedding:8b",
    validate_model_on_init=True,
    base_url="http://192.168.88.179:11434",
    client_kwargs={"timeout": 30},
)


def _vector_search(query: str) -> list[Document]:
    faiss_retriever = FAISS.load_local(
        "faiss_index",
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
        relevance_score_fn=None,
        distance_strategy=DistanceStrategy.EUCLIDEAN,
    ).as_retriever(k=10)

    docs = faiss_retriever.invoke(query)

    return docs


def _keyword_search(query: str) -> list[Document]:
    index_dir = "bm25_index"
    bm25_retriever = bm25s.BM25.load(index_dir, load_corpus=True, mmap=True)
    query_tokens = bm25s.tokenize(query, return_ids=False, stopwords="en")
    results = bm25_retriever.retrieve(query_tokens, k=10, return_as="documents")

    with open(f"{index_dir}/id_map.json") as f:
        id_map: list[dict[str, Any]] = json.load(f)

    docs: list[Document] = []
    for i in range(results.shape[1]):
        docs.append(
            Document(
                id=id_map[i]["id"],
                metadata=id_map[i]["metadata"],
                page_content=results[0][i]["text"],
            )
        )

    return docs


def summarize_conversation(state: StateSchema) -> dict[str, Any]:
    summarizer = SummarizationNode(
        model=llm.bind(num_predict=128).with_config({"stream": False}),
        max_tokens=1024,
        initial_summary_prompt=INITIAL_SUMMARY_PROMPT,
        existing_summary_prompt=EXISTING_SUMMARY_PROMPT,
        final_prompt=FINAL_SUMMARY_PROMPT,
    )

def prepare_thread(state: StateSchema) -> dict[str, Any]:
    return {"documents": []}


def route_query(state: StateSchema) -> Literal["answer", "hybrid_search"]:
    prompt = ROUTE_QUERY_PROMPT.format(query=state["messages"][-1].content)
    resp = (
        llm.with_structured_output(ClassificationOutput)
        .with_config({"temperature": 0})
        .invoke(prompt, config={"tags": ["router"]})
    )
    if resp["label"] == "DIRECT":
        return "answer"
    return "hybrid_search"


def hybrid_search(state: StateSchema) -> dict[str, Any]:
    query = state["messages"][-1].content
    vector_docs = _vector_search(query)
    keyword_docs = _keyword_search(query)
    docs = vector_docs + keyword_docs
    docs = list({doc.id: doc for doc in docs}.values())

    return {"documents": docs}


def rerank_documents(state: StateSchema) -> dict[str, Any]:
    query = state["messages"][-1].content
    docs = state["documents"]

    url = "http://192.168.88.179:2025/score"
    headers = {"accept": "application/json", "Content-Type": "application/json"}
    data = {
        "model": "AITeamVN/Vietnamese_Reranker",
        "encoding_format": "float",
        "queries": query,
        "documents": [doc.page_content for doc in docs],
    }
    scores = [0.0] * len(docs)

    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)

        if response.status_code == 200:
            results = response.json()["data"]
            scores = [result["score"] for result in results]
            reranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        else:
            print(f"Request failed with status code: {response.status_code}")
            reranked = list(zip(scores, docs))
    except Exception as e:
        print(f"An error occurred: {e}")
        reranked = list(zip(scores, docs))

    top_docs = [doc for _, doc in reranked[:5]]

    return {"documents": top_docs}


def answer(state: StateSchema) -> dict[str, Any]:
    messages = [SystemMessage(content=ANSWER_PROMPT.format())] + state["messages"]
    return {"messages": [llm.invoke(messages)]}


def answer_with_rag(state: StateSchema) -> dict[str, Any]:
    user_prompt = """
CONTEXT:
{retrieved_docs}

QUESTION:
{query}
""".strip().format(
        retrieved_docs="\n\n".join(
            [
                f"""--- Document {i} ---\nfilename: {doc.metadata["source"]}\ncontent:\n{doc.page_content}"""
                for i, doc in enumerate(state["documents"], 1)
            ]
        ),
        query=state["messages"][-1].content,
    )

    messages = (
        [SystemMessage(content=ANSWER_WITH_RAG_PROMPT.format())]
        + list(state["messages"][:-1])
        + [HumanMessage(content=user_prompt)]
    )
    ai_response = llm.invoke(messages)

    return {"messages": [ai_response]}


# Define the graph
graph = (
    StateGraph(state_schema=StateSchema)
    # define nodes
    .add_node("prepare_thread", prepare_thread)
    .add_node("hybrid_search", hybrid_search)
    .add_node("rerank_documents", rerank_documents)
    .add_node("answer", answer)
    .add_node("answer_with_rag", answer_with_rag)
    # define workflow
    .set_entry_point("prepare_thread")
    .add_conditional_edges("prepare_thread", route_query)
    .set_finish_point("answer")
    .add_edge("hybrid_search", "rerank_documents")
    .add_edge("rerank_documents", "answer_with_rag")
    .set_finish_point("answer_with_rag")
    # compile the graph
    .compile()
)
