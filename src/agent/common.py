import os
import pytz
import typing as t
from enum import Enum
from datetime import datetime
from functools import lru_cache
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from sqlalchemy import create_engine, Column, String, Text
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client.http.models import (
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
    DatetimeRange,
)
from langchain.messages import SystemMessage
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.documents import Document
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.output_parsers.boolean import BooleanOutputParser
from langchain_community.embeddings import InfinityEmbeddings
from langchain_ollama import ChatOllama

from src.lib.reranker import MyReranker
from src.lib.prompts import (
    EXTRACT_COMPLEMENT_PROMPT,
    EXTRACT_KEYWORD_PROMPT,
    EXTRACT_DATE_PROMPT,
    RETRIEVAL_DECISION_PROMPT,
    ROUTE_QUERY_PROMPT,
    SUMMARY_SYSTEM_PROMPT,
)
from src.lib.utils import canonicalize_value, canonicalize_date


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
    # DESCRIPTION = "mo_ta"
    ROOT_CAUSE = "nguyen_nhan"
    SOLUTION = "khac_phuc"
    LESSON = "bai_hoc"


class Department(str, Enum):
    SOFTWARE = "Bộ phận Phát triển phần mềm"
    ISO = "Bộ phận Ban ISO"


T = t.TypeVar("T")


class ScalarFilter(BaseModel, t.Generic[T]):
    model_config = ConfigDict(use_enum_values=True)

    eq: t.Optional[T] = Field(default=None)


class ListFilter(BaseModel, t.Generic[T]):
    model_config = ConfigDict(use_enum_values=True)

    in_: t.Optional[list[T]] = Field(default=None, alias="in")


class RangeFilter(BaseModel, t.Generic[T]):
    model_config = ConfigDict(use_enum_values=True)

    gt: t.Optional[T] = Field(default=None)
    gte: t.Optional[T] = Field(default=None)
    lt: t.Optional[T] = Field(default=None)
    lte: t.Optional[T] = Field(default=None)


class ExtractionKeyword(BaseModel):
    project_name: t.Optional[ListFilter[ProjectName]] = Field(
        default=None,
        description="Dự án được nhắc đến trong truy vấn (nếu có).",
    )

    chunk_type: t.Optional[ListFilter[ChunkType]] = Field(
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

    @field_validator("chunk_type", "project_name", mode="before")
    @classmethod
    def coerce_string_to_list_filter(cls, v):
        if isinstance(v, str):
            return {"in": [v]}
        return v

    @model_validator(mode="after")
    def normalize(self):
        if self.project_name and self.project_name.in_:
            self.project_name.in_ = list(map(canonicalize_value, self.project_name.in_))

        return self


class ExtractionDate(BaseModel):
    occurred_at: t.Optional[RangeFilter[str]] = Field(
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


class MetadataFilter(BaseModel):
    must: t.Optional[list[t.Union[ExtractionKeyword, ExtractionDate]]] = Field(default_factory=list)

    def __add__(self, other):
        if not isinstance(other, MetadataFilter):
            return NotImplemented

        combined_must = (self.must or []) + (other.must or [])
        return MetadataFilter(must=combined_must)


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
def create_reranker(
    top_n: t.Optional[int] = None, score_threshold: t.Optional[float] = None
):
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
        top_k=64,
    )

    config_with_kwargs = {**base_config, **kwargs}

    return ChatOllama(**config_with_kwargs)


@lru_cache
def create_question_answering_agent():
    # Lazy import to avoid slow loading at module import time
    from langchain.agents.middleware import SummarizationMiddleware

    class CustomSummarizationMiddleware(SummarizationMiddleware):
        pass

    class DynamicSystemPromptMiddleware(AgentMiddleware):
        def wrap_model_call(
            self,
            request: ModelRequest,
            handler: t.Callable[[ModelRequest], ModelResponse],
        ) -> ModelResponse:
            ctx = request.runtime.context or {}
            system_message = (
                request.system_message
                if request.system_message
                else SystemMessage(content=ctx.get("system_prompt", ""))
            )
            return handler(request.override(system_message=system_message))

    return create_agent(
        get_base_llm(
            top_p=0.9,
            repeat_penalty=1,
            presence_penalty=0,
            temperature=0.3,
        ),
        middleware=[
            CustomSummarizationMiddleware(
                get_base_llm(
                    top_p=0.95,
                    repeat_penalty=1,
                    presence_penalty=0,
                    temperature=0.5,
                    tags=["nostream"],
                ),
                trigger=[("tokens", 5000)],
                keep=("messages", 1),
                summary_prompt=SUMMARY_SYSTEM_PROMPT,
            ),
            DynamicSystemPromptMiddleware(),
        ],
        name=f"answer_agent",
    )


@lru_cache
def create_complement_extraction_agent():
    return (
        EXTRACT_COMPLEMENT_PROMPT
        | get_base_llm(
            repeat_penalty=1,
            presence_penalty=0,
            temperature=0.1,
            tags=["nostream"],
        )
        | StrOutputParser()
    )


@lru_cache
def create_router_agent():
    return (
        ROUTE_QUERY_PROMPT
        | get_base_llm(
            repeat_penalty=1,
            presence_penalty=0,
            temperature=0.3,
            tags=["nostream"],
        )
        | StrOutputParser()
    )


@lru_cache
def create_retrieval_decision_agent():
    return (
        RETRIEVAL_DECISION_PROMPT
        | get_base_llm(
            repeat_penalty=1,
            presence_penalty=0,
            temperature=0.5,
            tags=["nostream"],
        )
        | BooleanOutputParser()
    )


@lru_cache
def create_keyword_extraction_agent():
    return (
        dict(
            schema=lambda _: ExtractionKeyword.model_json_schema(),
            query=RunnablePassthrough(),
        )
        | EXTRACT_KEYWORD_PROMPT
        | get_base_llm(
            repeat_penalty=1,
            presence_penalty=0,
            temperature=0,
            tags=["nostream"],
        ).with_structured_output(ExtractionKeyword)
    )


@lru_cache
def create_date_extraction_agent():
    return (
        dict(
            schema=lambda _: ExtractionDate.model_json_schema(),
            now=lambda _: datetime.now(pytz.timezone("Asia/Ho_Chi_Minh")).strftime(
                r"%Y-%m-%d"
            ),
            query=RunnablePassthrough(),
        )
        | EXTRACT_DATE_PROMPT
        | get_base_llm(
            repeat_penalty=1,
            presence_penalty=0,
            temperature=0,
            tags=["nostream"],
        ).with_structured_output(ExtractionDate)
    )


Base = declarative_base()


class QuestionModel(Base):
    __tablename__ = "questions"

    id = Column(String, primary_key=True)
    question = Column(Text, nullable=False)
    answer = Column(String, nullable=False)
    sources = Column(String)

    def __repr__(self):
        return f"<QuestionModel(id={self.id}, question={self.question}, answer={self.answer}, sources={self.sources})>"


@lru_cache
def create_session_maker() -> Session:
    # Tạo engine và session
    engine = create_engine(f"sqlite:///{os.getenv("SQLITE_DB_NAME")}", echo=False)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session()


def build_field_condition(key: str, value: t.Union[ScalarFilter, ListFilter, RangeFilter]) -> t.Optional[FieldCondition]:
    data = value.model_dump(by_alias=True, exclude_none=True)
    prefix_key = "metadata"

    if "eq" in data:
        return FieldCondition(
            key=f"{prefix_key}.{key}",
            match=MatchValue(value=data["eq"]),
        )

    if "in" in data:
        return FieldCondition(
            key=f"{prefix_key}.{key}",
            match=MatchAny(any=data["in"]),
        )

    range_ops = {}

    for op in ["gt", "gte", "lt", "lte"]:
        if op in data:
            range_ops[op] = data[op]

    if range_ops:
        return FieldCondition(
            key=f"{prefix_key}.{key}",
            range=DatetimeRange(**range_ops),
        )

    return None


def build_conditions(sections: list[t.Union[ExtractionKeyword, ExtractionDate]]) -> list[FieldCondition]:
    conditions = []
    for section in sections:
        for field_name in section.model_dump(exclude_none=True).keys():
            filter_obj = getattr(section, field_name)

            if condition := build_field_condition(field_name, filter_obj):
                conditions.append(condition)

    return conditions


def build_qdrant_filter(filter: t.Optional[MetadataFilter] = None) -> Filter:
    sections = getattr(filter, "must", []) if filter else []
    return Filter(must=build_conditions(sections))


def merge_dicts(a: dict, b: dict) -> dict:
    return {**a, **b}


def build_context(docs: list[Document], format: t.Optional[str] = None) -> str:
    if not docs:
        return "Không tìm thấy tài liệu phù hợp."

    output: list[str] = []

    if format == "xml":
        output.append("<documents>")

    for idx, doc in enumerate(docs, 1):
        source = doc.metadata["source"]
        page = doc.metadata["page_number"]
        project = doc.metadata["project_name"]
        occurred_at = doc.metadata["occurred_at"]

        if format == "xml":
            output.append(
                f'<document source="{source}" page="{page}" project_name="{project.capitalize()}" occurred_at="{occurred_at}">'
            )
            output.append(doc.page_content.strip())
            output.append("</document>")
        else:
            output.append(
                f"[SOURCE] {source}#page={page} - project={project.capitalize()} - occurred_at={occurred_at}"
            )
            output.append(doc.page_content.strip())
            output.append("")

    if format == "xml":
        output.append("</documents>")

    return "\n".join(output)


def document_from_point(
    scored_point: t.Any,
    collection_name: str,
    content_payload_key: str,
    metadata_payload_key: str,
) -> Document:
    metadata = scored_point.payload.get(metadata_payload_key) or {}
    metadata["_id"] = scored_point.id
    metadata["_collection_name"] = collection_name
    return Document(
        page_content=scored_point.payload.get(content_payload_key, ""),
        metadata=metadata,
    )


def search_for_basic(
    query: str,
    filter: Filter,
    collection_name: str,
    top_k: int,
    score_threshold: float,
) -> list[tuple[Document, float]]:
    # phase_one_filter = filter.model_copy(deep=True)
    # if isinstance(phase_one_filter.must, list):
    #     for item in phase_one_filter.must:
    #         if isinstance(item, FieldCondition) and item.key == "metadata.chunk_type":
    #             item.match = MatchAny(any=["mo_ta"])
    results = get_qdrant_store(collection_name).similarity_search_with_score(
        query=query, filter=filter, k=top_k, score_threshold=score_threshold
    )
    # print("-" * 10 + " Phase 1: Similarity Search " + "-" * 10)
    # print("Filter: " + phase_one_filter.model_dump_json(indent=2))
    print(
        *(
            f"source={doc.metadata['source']}#page={doc.metadata['page_number']}: {score}"
            for doc, score in results
        ),
        sep="\n",
    )

    # phase_two_filter = filter.model_copy(deep=True)
    # if isinstance(phase_two_filter.must, list):
    #     phase_two_filter.must.append(
    #         FieldCondition(
    #             key="metadata.source",
    #             match=MatchAny(
    #                 any=list(
    #                     map(
    #                         lambda x: getattr(x, "metadata")["source"],
    #                         map(itemgetter(0), results),
    #                     )
    #                 )
    #             ),
    #         )
    #     )
    # points = (
    #     get_qdrant_store(collection_name)
    #     .client.query_points(
    #         collection_name=collection_name,
    #         query_filter=phase_two_filter,
    #         limit=top_k,
    #     )
    #     .points
    # )
    # results = [
    #     (
    #         document_from_point(result, collection_name, "page_content", "metadata"),
    #         result.score,
    #     )
    #     for result in points
    # ]
    # print("-" * 10 + " Phase 2: Similarity Search " + "-" * 10)
    # # print("Filter: " + phase_two_filter.model_dump_json(indent=2))
    # print(
    #     *(
    #         f"source={doc.metadata['source']}#page={doc.metadata['page_number']}: {score}"
    #         for doc, score in results
    #     ),
    #     sep="\n",
    # )

    return results


def search_for_others(
    filter: Filter, collection_name: str, top_k: int
) -> list[tuple[Document, float]]:
    points = (
        get_qdrant_store(collection_name)
        .client.query_points(
            collection_name=collection_name, query_filter=filter, limit=top_k
        )
        .points
    )
    results = [
        (
            document_from_point(result, collection_name, "page_content", "metadata"),
            result.score,
        )
        for result in points
    ]
    # print("-" * 10 + " Phase 1: Similarity Search " + "-" * 10)
    # print("Filter: " + filter.model_dump_json(indent=2))
    print(
        *(
            f"source={doc.metadata['source']}#page={doc.metadata['page_number']}: {score}"
            for doc, score in results
        ),
        sep="\n",
    )

    return results
