import json
import pprint
import requests
from typing import Any, Literal, TypedDict

import bm25s
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.messages import AnyMessage, SystemMessage
from langchain.embeddings import init_embeddings
from langchain_community.tools import tool
from langchain_community.vectorstores import FAISS, DistanceStrategy
from langchain_ollama.chat_models import ChatOllama
from langgraph.graph import END, MessagesState, StateGraph
from langmem.short_term import SummarizationNode, RunningSummary


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

class StateSchema(MessagesState):
    # summarized_messages: list[AnyMessage]
    # context: dict[str, RunningSummary]

    documents: list[Document]
    current_query: str


class ContextSchema(TypedDict):
    pass


class OutputSchema(MessagesState):
    pass


class GradeOutput(TypedDict):
    score: str


llm = ChatOllama(
    model="qwen3.5:9b",
    base_url="http://192.168.88.179:11434",
    temperature=0,
    reasoning=False,
    num_predict=-2,
    client_kwargs={"timeout": 30},
)

embeddings = init_embeddings(
    "qwen3-embedding:8b",
    provider="ollama",
    base_url="http://192.168.88.179:11434",
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


@tool
def _hybrid_search(query: str) -> str:
    """Retrieve relevant lesson learned documents

    Args:
        query (str): The user's query

    Returns:
        str: A string containing the search results
    """
    return query


llm_with_tools = llm.bind_tools([_hybrid_search])

summarization_node = SummarizationNode(
    model=llm,
    max_tokens=256,
    max_tokens_before_summary=128,
    max_summary_tokens=128,
    initial_summary_prompt=INITIAL_SUMMARY_PROMPT,
    existing_summary_prompt=EXISTING_SUMMARY_PROMPT,
    final_prompt=FINAL_SUMMARY_PROMPT,
)


def route_query(state: StateSchema) -> dict[str, Any]:
    system_prompt = f"""
You are an assistant for a private knowledge base focused on information technology, particularly software development.

**Chat history:**
{{summary}}

**Instructions:**
- You must use Vietnamese as the primary language in all responses.
- If the request is a general conversation, respond directly. Otherwise, you must use the available search tool to retrieve relevant information before answering.
- You must ensure the answer is concise, accurate, and strictly aligned with the provided context.
- You must not fabricate, assume, or infer any information beyond what is explicitly supported by the provided context.
- You must not use any external or web-based information.
- The answer must not include any information that is not explicitly supported by the provided context, even if it is commonly known.
- The chat history above is for context only. You must not use it to answer questions about the referenced documents or rely on it in any way.
- You should always include citations for the sources used at the end of your answer in bulleted list format.
```example
**Citations:**
* [1] [filename:name_of_the_cited_document]
* [2] [filename:name_of_the_cited_document]
```
""".strip()
    if not isinstance(state["summarized_messages"][0], SystemMessage):
        messages = [SystemMessage(content=system_prompt.format(summary=""))] + state["summarized_messages"]
    else:
        summary = state["summarized_messages"][0].content
        pprint.pprint(system_prompt.format(summary=summary))
        messages = [SystemMessage(content=system_prompt.format(summary=summary))] + state["summarized_messages"][1:]
    # pprint.pprint(messages)
    ai_response = llm_with_tools.invoke(messages)
    if ai_response.tool_calls:
        return {"current_query": state["messages"][-1].content}

    return {"messages": [ai_response], "documents": []}


def search_condition(state: StateSchema) -> Literal["hybrid_search", "__end__"]:
    return "hybrid_search" if "current_query" in state and state["current_query"] else END


def hybrid_search(state: StateSchema) -> dict[str, Any]:
    query = state["current_query"]
    vector_docs = _vector_search(query)
    keyword_docs = _keyword_search(query)
    docs = vector_docs + keyword_docs
    docs = list({doc.id: doc for doc in docs}.values())

    return {"documents": docs}


def rerank_documents(state: StateSchema) -> dict[str, Any]:
    query = state["current_query"]
    docs = state["documents"]

    url = "http://192.168.88.179:2025/score"
    headers = {"accept": "application/json", "Content-Type": "application/json"}
    data = {
        "model": "BAAI/bge-reranker-v2-m3",
        "encoding_format": "float",
        "queries": query,
        "documents": [doc.page_content for doc in docs],
    }
    scores = [0.0] * len(docs)

    try:
        response = requests.post(url, headers=headers, json=data, timeout=10)

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


def generate_answer(state: StateSchema) -> dict[str, Any]:
    system_prompt = f"""
You are an assistant for a private knowledge base focused on information technology, particularly software development.

**Chat history:**
{{summary}}

**Context:**
{"\n\n".join([f"""[{i}] [filename: {doc.metadata.get('source', 'Unknown')}]\n{doc.page_content}""" for i, doc in enumerate(state["documents"], 1)])}

**Instructions:**
- You must use Vietnamese as the primary language in all responses.
- You must use only the provided context to answer the question.
- You must ensure the answer is concise, accurate, and strictly aligned with the provided context.
- You must not fabricate, assume, or infer any information beyond what is explicitly supported by the provided context.
- You must not use any external or web-based information.
- If no relevant information is found in the provided context, you must respond with: "Nội dung này không có trong cơ sở kiến thức của tôi."
- The answer must not include any information that is not explicitly supported by the provided context, even if it is commonly known.
- The chat history above is for context only. You must not use it to answer questions about the referenced documents or rely on it in any way.
- You should always include citations for the sources used at the end of your answer in bulleted list format.
```example
**Citations:**
* [1] [filename:name_of_the_cited_document]
* [2] [filename:name_of_the_cited_document]
```
""".strip()

    if not isinstance(state["summarized_messages"][0], SystemMessage):
        messages = [SystemMessage(content=system_prompt.format(summary=""))] + state["summarized_messages"]
    else:
        summary = state["summarized_messages"][0].content
        pprint.pprint(system_prompt.format(summary=summary))
        messages = [SystemMessage(content=system_prompt.format(summary=summary))] + state["summarized_messages"][1:]
    # pprint.pprint(messages)
    ai_response = llm.invoke(messages)

    return {"messages": [ai_response], "current_query": None}


# Define the graph
graph = (
    StateGraph(
        state_schema=StateSchema,
        context_schema=ContextSchema,
        # output_schema=OutputSchema,
    )
    # define nodes
    .add_node("summarize", summarization_node)
    .add_node("route_query", route_query)
    .add_node("hybrid_search", hybrid_search)
    .add_node("rerank_documents", rerank_documents)
    .add_node("generate_answer", generate_answer)
    # define workflow
    # .set_entry_point("summarize")
    # .add_edge("summarize", "route_query")
    .set_entry_point("route_query")
    .add_conditional_edges("route_query", search_condition)
    .add_edge("hybrid_search", "rerank_documents")
    .add_edge("rerank_documents", "generate_answer")
    .set_finish_point("generate_answer")
    # compile the graph
    .compile()
)
