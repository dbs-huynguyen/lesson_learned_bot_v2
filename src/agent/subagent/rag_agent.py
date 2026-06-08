from operator import itemgetter, add
from typing import TypedDict, Optional, Annotated, Union

from qdrant_client.http.models import Filter, FieldCondition, MatchAny, MatchValue
from langchain_core.documents import Document
from langgraph.config import get_stream_writer
from langgraph.graph import StateGraph

from src.agent.common import (
    ExtractionDate,
    ExtractionKeyword,
    MetadataFilter,
    KeywordFilter,
    DateFilter,
    get_qdrant_store,
    create_reranker,
    create_keyword_extraction_agent,
    create_date_extraction_agent,
    build_qdrant_filter,
)


class StateSchema(TypedDict):
    query: str
    query_retrieval: str
    top_k: int
    score_threshold: float
    collection_name: str
    use_reranking: bool

    keyword_filter: Optional[KeywordFilter]
    date_filter: Optional[DateFilter]
    metadata_filter: Optional[Filter]
    retrieved_docs: list[Document]


class OutputSchema(TypedDict):
    final_docs: Annotated[list[Document], add]


def extract_keyword(state: StateSchema) -> dict:
    try:
        keyword_extraction_agent = create_keyword_extraction_agent()
        result = keyword_extraction_agent.invoke(state["query"])
        print(result)
        keyword_filter = MetadataFilter[ExtractionKeyword](must=[result])
    except Exception as e:
        print(str(e))
        keyword_filter = None

    updated = dict(keyword_filter=keyword_filter)

    return updated


def extract_date(state: StateSchema) -> dict:
    try:
        date_extraction_agent = create_date_extraction_agent()
        result = date_extraction_agent.invoke(state["query"])
        print(result)
        date_filter = MetadataFilter[ExtractionDate](must=[result])
    except Exception as e:
        print(str(e))
        date_filter = None

    updated = dict(date_filter=date_filter)

    return updated


def build_metadata_filter(state: StateSchema) -> dict:
    if state["date_filter"] and state["keyword_filter"]:
        metadata_filter = MetadataFilter[Union[ExtractionKeyword, ExtractionDate]](
            must=state["keyword_filter"].must + state["date_filter"].must
        )
    elif state["date_filter"]:
        metadata_filter = state["date_filter"]
    elif state["keyword_filter"]:
        metadata_filter = state["keyword_filter"]
    else:
        metadata_filter = None

    if metadata_filter is not None:
        metadata_filter = build_qdrant_filter(metadata_filter)

    updated = dict(metadata_filter=metadata_filter)

    return updated


def hybrid_search(state: StateSchema) -> dict:
    print(f"Query for retrieval: {state['query_retrieval']}")
    print("Original Filter: " + state["metadata_filter"].model_dump_json(indent=2))

    print("-" * 10 + " Phase 1: Similarity Search Results " + "-" * 10)
    phase_one_filter = state["metadata_filter"].model_copy(deep=True)
    if isinstance(phase_one_filter.must, list):
        for item in phase_one_filter.must:
            if isinstance(item, FieldCondition) and item.key == "metadata.chunk_type":
                item.match = MatchValue(value="mo_ta")
    print("Phase 1 Filter: " + phase_one_filter.model_dump_json(indent=2))
    results = get_qdrant_store(state["collection_name"]).similarity_search_with_score(
        query=state["query_retrieval"],
        filter=phase_one_filter,
        k=state["top_k"],
        score_threshold=state["score_threshold"],
    )
    print(
        *(
            f"{doc.metadata['source']}#{doc.metadata['page_number']}: {score}"
            for doc, score in results
        ),
        sep="\n",
    )

    print("-" * 10 + " Phase 2: Similarity Search Results " + "-" * 10)
    phase_two_filter = state["metadata_filter"].model_copy(deep=True)
    if isinstance(phase_two_filter.must, list):
        phase_two_filter.must.append(
            FieldCondition(
                key="metadata.source",
                match=MatchAny(
                    any=list(
                        map(
                            lambda x: getattr(x, "metadata")["source"],
                            map(itemgetter(0), results),
                        )
                    )
                ),
            )
        )
    print("Filter with source: " + phase_two_filter.model_dump_json(indent=2))
    results = get_qdrant_store(state["collection_name"]).client.query_points(
        collection_name=state["collection_name"],
        query_filter=phase_two_filter,
        limit=state["top_k"],
    )
    print(results.points)
    # print(
    #     *(
    #         f"{doc.metadata['source']}#{doc.metadata['page_number']}: {score}"
    #         for doc, score in results
    #     ),
    #     sep="\n",
    # )

    # from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse
    # from langchain_classic.retrievers import EnsembleRetriever

    # dense_retriever = QdrantVectorStore.from_existing_collection(
    #     embedding=get_embeddings(),
    #     collection_name=state["collection_name"],
    #     vector_name="dense",
    #     retrieval_mode=RetrievalMode.DENSE,
    #     url="http://192.168.88.179:6333",
    # ).as_retriever(
    #     search_kwargs={
    #         "k": state["top_k"],
    #         # "filter": state["metadata_filter"],
    #         "filter": Filter(
    #             must=[
    #                 FieldCondition(
    #                     key="metadata.chunk_type",
    #                     match=MatchAny(any=["mo_ta"]),
    #                 ),
    #                 FieldCondition(
    #                     key="metadata.department",
    #                     match=MatchAny(any=["BP Phát triển phần mềm"]),
    #                 ),
    #             ]
    #         ),
    #         "score_threshold": state["score_threshold"],
    #     }
    # )

    # sparse_retriever = QdrantVectorStore.from_existing_collection(
    #     sparse_embedding=FastEmbedSparse(),
    #     collection_name=state["collection_name"],
    #     sparse_vector_name="sparse",
    #     retrieval_mode=RetrievalMode.SPARSE,
    #     url="http://192.168.88.179:6333",
    # ).as_retriever(
    #     search_kwargs={
    #         "k": state["top_k"],
    #         # "filter": state["metadata_filter"],
    #         "filter": Filter(
    #             must=[
    #                 FieldCondition(
    #                     key="metadata.chunk_type",
    #                     match=MatchAny(any=["mo_ta"]),
    #                 ),
    #                 FieldCondition(
    #                     key="metadata.department",
    #                     match=MatchAny(any=["BP Phát triển phần mềm"]),
    #                 ),
    #             ]
    #         ),
    #         # "score_threshold": 6.0,
    #     }
    # )

    # hybrid_retriever = EnsembleRetriever(
    #     retrievers=[dense_retriever, sparse_retriever],
    #     weights=[0.8, 0.2],
    #     id_key="_id",
    # )
    # docs = hybrid_retriever.invoke(state["query_retrieval"])
    # return {"retrieved_docs": docs}

    # if state["collection_name"] == "questions":
    #     print(
    #         *(f"{doc.metadata['doc_id']}: {score}" for doc, score in results), sep="\n"
    #     )
    # else:
    #     print(
    #         *(
    #             f"{doc.metadata['source']}#{doc.metadata['page_number']}: {score}"
    #             for doc, score in results
    #         ),
    #         sep="\n",
    #     )

    updated = dict(retrieved_docs=list(map(itemgetter(0), results)))

    return updated


def rerank_docs(state: StateSchema) -> dict:
    if state["collection_name"] == "lessons_learned":
        retrieved_docs = state["retrieved_docs"]
    else:
        qdrant_store = get_qdrant_store("lessons_learned")
        retrieved_docs = qdrant_store.get_by_ids(
            list(
                dict.fromkeys(
                    doc.metadata["lesson_learned_id"] for doc in state["retrieved_docs"]
                )
            )
        )
        print("========== Mapped Results ==========")
        print(
            *(
                f"{doc.metadata['source']}#{doc.metadata['page_number']}"
                for doc in retrieved_docs
            ),
            sep="\n",
        )

    if state["use_reranking"]:
        reranker = create_reranker()
        results = reranker.compress_documents_with_score(
            retrieved_docs, state["query_retrieval"]
        )
        print("========== Reranked Results ==========")
        print(
            *(
                f"{doc.metadata['source']}#{doc.metadata['page_number']}: {score}"
                for doc, score in results
            ),
            sep="\n",
        )
        docs = list(map(itemgetter(0), results))
    else:
        docs = retrieved_docs

    writer = get_stream_writer()
    writer(
        dict(
            type="reasoning",
            message="Sắp xếp {} tài liệu theo mức độ liên quan.".format(len(docs)),
        )
    )

    updated = dict(final_docs=docs)

    return updated


# Define the graph
graph = (
    StateGraph(state_schema=StateSchema, output_schema=OutputSchema)
    # define nodes
    .add_node("extract_keyword", extract_keyword)
    .add_node("extract_date", extract_date)
    .add_node("build_metadata_filter", build_metadata_filter)
    .add_node("hybrid_search", hybrid_search)
    .add_node("rerank_docs", rerank_docs)
    # define workflow
    .set_entry_point("extract_keyword")
    .set_entry_point("extract_date")
    .add_edge("merge", "build_metadata_filter")
    .add_edge("extract_date", "build_metadata_filter")
    .add_edge("build_metadata_filter", "hybrid_search")
    .add_edge("hybrid_search", "rerank_docs")
    .set_finish_point("rerank_docs")
    # compile the graph
    .compile(name="rag_agent_graph")
)
