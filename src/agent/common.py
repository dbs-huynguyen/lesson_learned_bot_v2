import os
from enum import Enum
from typing import Generic, Optional, TypeVar, Callable, Any
from pydantic import BaseModel as PyBaseModel, ConfigDict, Field, model_validator
from functools import lru_cache

from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client.http.models import (
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
    DatetimeRange,
)
from langchain.messages import SystemMessage, RemoveMessage
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.documents import Document
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.runnables import RunnablePassthrough, RunnableAssign
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.output_parsers.boolean import BooleanOutputParser
from langchain_classic.retrievers.document_compressors import LLMChainFilter
from langchain_community.embeddings import InfinityEmbeddings
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langgraph.config import get_stream_writer
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from src.lib.reranker import MyReranker
from src.lib.prompts import (
    ROUTE_QUERY_STAGE_ONE_PROMPT,
    EXTRACT_KEYWORD_PROMPT,
    EXTRACT_DATE_PROMPT,
    RETRIEVAL_DECISION_PROMPT,
    ROUTE_QUERY_STAGE_THREE_PROMPT,
    ROUTE_QUERY_STAGE_TWO_PROMPT,
    SUMMARY_SYSTEM_PROMPT,
    REWRITE_QUERY_PROMPT,
    GRADE_DOCS_PROMPT,
)
from src.lib.utils import canonicalize_value, canonicalize_date

T = TypeVar("T")


class BaseModel(PyBaseModel):
    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=True,
    )


class ScalarFilter(BaseModel, Generic[T]):
    eq: Optional[T] = None


class ListFilter(BaseModel, Generic[T]):
    in_: Optional[list[T]] = Field(default=None, alias="in")


class RangeFilter(BaseModel, Generic[T]):
    gt: Optional[T] = None
    gte: Optional[T] = None
    lt: Optional[T] = None
    lte: Optional[T] = None


class FieldFilter(ScalarFilter, ListFilter, RangeFilter, Generic[T]):
    pass


class MetadataFilter(BaseModel, Generic[T]):
    must: Optional[list[T]] = None


class AgentType(str, Enum):
    TREND = "trend_agent"
    CLASSIFICATION = "classification_agent"
    STATISTICS = "statistics_agent"
    BASIC = "basic_agent"


class AgentClassification(BaseModel):
    # agent: AgentType = Field(
    #     AgentType.BASIC,
    #     description=("The name of the sub-agent to route the query to."),
    # )
    intent: str = Field(
        default=None,
        description=(
            "Phân loại ý định của người dùng.\n"
            "Chỉ chọn 1 trong: 'CHI_TIET' hoặc 'TONG_HOP'"
        ),
    )
    complement: str = Field(
        default=None,
        description="Phần bổ ngữ đã được xác định hoặc rỗng.",
    )


class ProjectName(str, Enum):
    AUTH = "auth"
    AUTH_CONSOLE = "authconsole"
    AUTH_DATA = "authdata"
    PASS = "pass"
    TOKEN = "token"
    SIGN = "sign"
    CONTRACT = "contract"
    LIVE = "live"
    LEARN = "learn"
    PIVOT = "pivot"
    CARECONNE = "careconne"
    CDS = "cds"
    LEO_SOPHIA = "leosophia"
    MONSHIN_APP = "monshinapp"
    PREMONSHIN_APP = "premonshinapp"
    ALIVE_MONITORING = "alivemonitoring"


class ChunkType(str, Enum):
    DESCRIPTION = "mo_ta"
    ROOT_CAUSE = "nguyen_nhan"
    SOLUTION = "khac_phuc"
    LESSON = "bai_hoc"


class Department(str, Enum):
    SOFTWARE = "BP Phát triển phần mềm"
    ISO = "Bộ phận Ban ISO"


class ExtractionKeyword(BaseModel):
    project_name: Optional[ListFilter[ProjectName]] = Field(
        default=None,
        description="Dự án được nhắc đến trong truy vấn (nếu có).",
    )

    chunk_type: Optional[ListFilter[ChunkType]] = Field(
        default=None,
        description=(
            "Phần tài liệu được nhắc đến trong truy vấn.\n"
            "Ví dụ:\n"
            "- 'nguyên nhân', 'lý do', 'nguồn gốc',... -> nguyen_nhan\n"
            "- 'cách xử lý', 'giải pháp', 'hướng khắc phục', 'cách giải quyết', 'cách fix', 'cách sửa',... -> khac_phuc\n"
            "- 'bài học', 'kinh nghiệm', 'đề xuất', 'cải tiến',... -> bai_hoc\n"
            "- Giá trị mặc định -> mo_ta"
        ),
    )

    department: Optional[ListFilter[Department]] = Field(
        default=None,
        description=(
            "Phòng ban được nhắc đến trong truy vấn (nếu có).\n"
            "Ví dụ:\n"
            "- 'phòng phát triển phần mềm', 'bộ phận phát triển phần mềm', 'team phát triển phần mềm',... -> BP Phát triển phần mềm\n"
            "- 'bộ phận ISO', 'phòng ISO', 'team ISO',... -> Bộ phận Ban ISO"
        ),
    )

    @model_validator(mode="after")
    def normalize(self):
        if self.project_name and self.project_name.in_:
            self.project_name.in_ = list(map(canonicalize_value, self.project_name.in_))

        return self

    @model_validator(mode="after")
    def normalize(self):
        if self.chunk_type is None:
            self.chunk_type = {"in": ["mo_ta"]}

        return self


class ExtractionDate(BaseModel):
    occurred_at: Optional[RangeFilter[str]] = Field(
        default=None,
        description="Thời gian được đề cập trong truy vấn. Có thể là một thời điểm cụ thể hoặc một khoảng thời gian. Ví dụ: 'ngày 1/1/2025', 'tháng 1 năm 2025', 'năm 2025', hoặc 'từ ngày 1/1/2025 đến ngày 31/12/2025'.",
    )

    @model_validator(mode="after")
    def normalize(self):
        if self.occurred_at and isinstance(self.occurred_at, RangeFilter):
            for op in ["gt", "gte", "lt", "lte"]:
                if getattr(self.occurred_at, op) is not None:
                    setattr(
                        self.occurred_at,
                        op,
                        canonicalize_date(getattr(self.occurred_at, op)),
                    )

        return self


class GradeDoc(BaseModel):
    relevant: bool = Field(
        description="Whether the document is relevant to the query.",
    )


@lru_cache
def get_embeddings():
    return InfinityEmbeddings(
        model=os.getenv("EMBEDDING_MODEL"),
        infinity_api_url=os.getenv("EMBEDDING_BASE_URL"),
    )


@lru_cache(maxsize=1)
def get_qdrant_store(collection_name: str) -> QdrantVectorStore:
    # Lazy import to avoid slow PyTorch loading at module import time
    from langchain_qdrant import FastEmbedSparse

    client = QdrantClient(url=os.getenv("QDRANT_URL"))

    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=get_embeddings(),
        vector_name="dense",
        sparse_embedding=FastEmbedSparse(),
        sparse_vector_name="sparse",
        retrieval_mode=RetrievalMode.HYBRID,
    )


@lru_cache
def get_reranker(top_n: Optional[int] = None, score_threshold: Optional[float] = None):
    return MyReranker(
        base_url=os.getenv("RERANKER_BASE_URL"),
        model=os.getenv("RERANKER_MODEL"),
        top_n=top_n,
        score_threshold=score_threshold,
        timeout=120,
    )


rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.1,  # <-- Can only make a request once every 10 seconds!!
    check_every_n_seconds=0.1,  # Wake up every 100 ms to check whether allowed to make a request,
    max_bucket_size=20,  # Controls the maximum burst size.
)


def get_base_llm(**kwargs):
    # if "num_predict" in kwargs:
    #     kwargs["max_completion_tokens"] = kwargs.pop("num_predict")
    # return ChatOpenAI(
    #     model=os.getenv("OPENAI_LLM_MODEL"),
    #     api_key=os.getenv("OPENAI_API_KEY"),
    #     base_url=os.getenv("OPENAI_BASE_URL"),
    #     timeout=120,
    #     rate_limiter=rate_limiter,
    #     seed=9999,
    #     extra_body={
    #         "chat_template_kwargs": {"enable_thinking": False},
    #     },
    #     reasoning_effort=None,
    #     **kwargs,
    # )

    base_config = dict(
        model=os.getenv("OLLAMA_LLM_MODEL"),
        base_url=os.getenv("OLLAMA_BASE_URL"),
        client_kwargs={"timeout": 120},
        keep_alive=-1,
        # rate_limiter=rate_limiter,
        seed=9999,
        num_ctx=int(os.getenv("OLLAMA_NUM_CTX")),
        reasoning=False,
        presence_penalty=1.5,
        temperature=1,
        top_k=20,
        top_p=0.95,
    )

    config_with_kwargs = {**base_config, **kwargs}

    return ChatOllama(**config_with_kwargs)


@lru_cache
def answer_agent():
    # Lazy import to avoid slow loading at module import time
    from langchain.agents.middleware import SummarizationMiddleware

    class CustomSummarizationMiddleware(SummarizationMiddleware):
        pass
        # def before_model(self, state, runtime) -> dict[str, Any] | None:
        #     messages = state["messages"]
        #     self._ensure_message_ids(messages)

        #     total_tokens = self.token_counter(messages)
        #     if not self._should_summarize(messages, total_tokens):
        #         return None

        #     cutoff_index = self._determine_cutoff_index(messages)

        #     kind, value = self.keep
        #     if kind == "messages":
        #         cutoff_index -= value

        #     if cutoff_index <= 0:
        #         return None

        #     messages_to_summarize, preserved_messages = self._partition_messages(
        #         messages, cutoff_index
        #     )

        #     summary = self._create_summary(messages_to_summarize)
        #     new_messages = self._build_new_messages(summary)

        #     return {
        #         "messages": [
        #             RemoveMessage(id=REMOVE_ALL_MESSAGES),
        #             *new_messages,
        #             *preserved_messages,
        #         ]
        #     }

    class DynamicContextMiddleware(AgentMiddleware):
        def wrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], ModelResponse],
        ) -> ModelResponse:
            ctx = request.runtime.context or {}
            system_message = (
                request.system_message
                if request.system_message
                else SystemMessage(content=ctx.get("system_prompt", ""))
            )
            writer = get_stream_writer()
            writer(
                {
                    "type": "reasoning",
                    "message": "Hoàn tất bước suy luận, bắt đầu trả lời câu hỏi.",
                }
            )
            return handler(request.override(system_message=system_message))

    return create_agent(
        get_base_llm(temperature=0.3, presence_penalty=0.0),
        middleware=[
            CustomSummarizationMiddleware(
                get_base_llm(temperature=0.7, tags=["nostream"]),
                trigger=[("tokens", 5000)],
                keep=("messages", 1),
                summary_prompt=SUMMARY_SYSTEM_PROMPT,
            ),
            DynamicContextMiddleware(),
        ],
        name=f"answer_agent",
    )


@lru_cache
def task_classification_1_agent():
    return (
        ROUTE_QUERY_STAGE_ONE_PROMPT
        | get_base_llm(temperature=0.5, tags=["nostream"])
        | StrOutputParser()
    )


@lru_cache
def task_classification_2_agent():
    return (
        ROUTE_QUERY_STAGE_TWO_PROMPT
        | get_base_llm(temperature=0.5, tags=["nostream"])
        | StrOutputParser()
    )


@lru_cache
def task_classification_3_agent():
    return (
        ROUTE_QUERY_STAGE_THREE_PROMPT
        | get_base_llm(temperature=0.5, tags=["nostream"])
        | StrOutputParser()
    )


@lru_cache
def decision_making_agent():
    return (
        {"query": RunnablePassthrough()}
        | RETRIEVAL_DECISION_PROMPT
        | get_base_llm(temperature=0.5, tags=["nostream"])
        | BooleanOutputParser()
    )


@lru_cache
def rewrite_query_agent():
    return (
        {"messages": RunnablePassthrough()}
        | REWRITE_QUERY_PROMPT
        | get_base_llm(temperature=0.7, tags=["nostream"])
        | StrOutputParser()
    )


@lru_cache
def keyword_extraction_agent():
    return (
        dict(
            schema=lambda x: ExtractionKeyword.model_json_schema(),
            query=RunnablePassthrough(),
        )
        | EXTRACT_KEYWORD_PROMPT
        | get_base_llm(temperature=0, tags=["nostream"]).with_structured_output(
            ExtractionKeyword
        )
    )


@lru_cache
def date_extraction_agent():
    return (
        dict(
            schema=lambda x: ExtractionDate.model_json_schema(),
            now=RunnablePassthrough(),
            query=RunnablePassthrough(),
        )
        | EXTRACT_DATE_PROMPT
        | get_base_llm(temperature=0, tags=["nostream"]).with_structured_output(
            ExtractionDate
        )
    )


@lru_cache
def relevance_grading_agent():
    return LLMChainFilter.from_llm(
        llm=get_base_llm(temperature=0.5, tags=["nostream"]),
        prompt=GRADE_DOCS_PROMPT,
    )


def build_field_condition(
    field_name: str, field_filter: FieldFilter
) -> Optional[FieldCondition]:
    data = field_filter.model_dump(by_alias=True, exclude_none=True)

    if "eq" in data:
        return FieldCondition(
            key=f"metadata.{field_name}",
            match=MatchValue(value=data["eq"]),
        )

    if "in" in data:
        return FieldCondition(
            key=f"metadata.{field_name}",
            match=MatchAny(any=data["in"]),
        )

    range_ops = {}

    for op in ["gt", "gte", "lt", "lte"]:
        if op in data:
            range_ops[op] = data[op]

    if range_ops:
        return FieldCondition(
            key=f"metadata.{field_name}",
            range=DatetimeRange(**range_ops),
        )

    return None


def build_conditions(sections: list[MetadataFilter[T]]) -> list[FieldCondition]:
    if sections is None:
        return []

    conditions: list[FieldCondition] = []
    for section in sections:
        for field_name in section.model_dump(exclude_none=True).keys():
            filter_obj = getattr(section, field_name)

            condition = build_field_condition(field_name, filter_obj)

            if condition:
                conditions.append(condition)

    return conditions


def to_qdrant_filter(metadata_filter: MetadataFilter) -> Filter:
    return Filter(
        must=build_conditions(metadata_filter.must),
    )


def merge_documents(
    left: Optional[dict[str, Document]], right: Optional[dict[str, Document]]
) -> Optional[dict[str, Document]]:
    """Merge two document dictionaries, with right taking precedence."""
    if left is None:
        return right
    if right is None:
        return left
    return {**left, **right}


def build_relevant_docs_ctx(docs: list[Document]) -> tuple[str, dict[str, Document]]:
    if not docs:
        return "Không tìm thấy tài liệu phù hợp.", {}

    xml_docs = ["<documents>"]
    for idx, doc in enumerate(docs, 1):
        source = doc.metadata["source"]
        page = doc.metadata["page_number"]
        project = doc.metadata["project_name"]
        occurred_at = doc.metadata["occurred_at"]

        open_document_tag = f'<document source="{source}" page="{page}" project_name="{project.capitalize()}" occurred_at="{occurred_at}">'
        close_document_tag = "</document>"

        xml_docs.append(open_document_tag)
        xml_docs.append(doc.page_content.strip())
        xml_docs.append(close_document_tag)
    xml_docs.append("</documents>")

    relevant_docs = "\n".join(xml_docs)
    # print(relevant_docs)

    documents = {
        f"{doc.metadata['source']}#page={doc.metadata['page_number']}": doc
        for doc in docs
    }

    return relevant_docs, documents
