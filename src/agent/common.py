import os
from enum import Enum
from typing import Generic, Optional, TypeVar, Callable
from pydantic import BaseModel as PyBaseModel, ConfigDict, Field, model_validator
from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
    DatetimeRange,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain.messages import SystemMessage
from langchain.agents import create_agent
from langchain.agents.middleware import (
    SummarizationMiddleware,
    AgentMiddleware,
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse
from langchain_ollama import ChatOllama

from src.lib.reranker import MyReranker
from src.lib.prompts import BASIC_AGENT_SYSTEM_PROMPT
from src.lib.utils import canonicalize_value, canonicalize_date

SYSTEM_PROMPT_REGISTRY: dict[str, ChatPromptTemplate] = {
    "trend_agent": BASIC_AGENT_SYSTEM_PROMPT,
    "classification_agent": BASIC_AGENT_SYSTEM_PROMPT,
    "statistics_agent": BASIC_AGENT_SYSTEM_PROMPT,
    "basic_agent": BASIC_AGENT_SYSTEM_PROMPT,
}


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
    agent: AgentType = Field(
        AgentType.BASIC,
        description=("The name of the sub-agent to route the query to."),
    )


class ProjectName(str, Enum):
    AUTH = "auth"
    PASS = "pass"
    TOKEN = "token"
    SIGN = "sign"
    CONTRACT = "contract"
    LINK = "link"
    CDS = "cds"
    MONSHIN_APP = "monshinapp"
    PREMONSHIN_APP = "premonshinapp"


class ExtractionKeyword(BaseModel):
    project_name: Optional[ListFilter[ProjectName]] = Field(
        default=None,
        description="Project name is mentioned in the query.",
    )

    @model_validator(mode="after")
    def normalize(self):
        if self.project_name and self.project_name.in_:
            self.project_name.in_ = list(map(canonicalize_value, self.project_name.in_))

        return self


class ExtractionDatetime(BaseModel):
    occurred_at: Optional[RangeFilter[str]] = Field(
        default=None,
        description="The datetime mentioned in the query. Can be a specific point in time or a range.",
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


@lru_cache
def get_qdrant_store():
    embeddings = OpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL"),
        base_url=os.getenv("EMBEDDING_BASE_URL"),
    )

    client = QdrantClient(path=os.getenv("QDRANT_INDEX_DIR"))

    return QdrantVectorStore(
        client=client,
        collection_name="bhkn",
        embedding=embeddings,
        vector_name="dense",
        sparse_embedding=FastEmbedSparse(),
        sparse_vector_name="sparse",
        retrieval_mode=RetrievalMode.HYBRID,
    )


@lru_cache
def get_reranker(top_n: int = 10):
    return MyReranker(
        base_url=os.getenv("RERANKER_BASE_URL"),
        model=os.getenv("RERANKER_MODEL"),
        top_n=top_n,
    )


class DynamicContextMiddleware(AgentMiddleware):
    def wrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        ctx = request.runtime.context or {}
        dynamic_context = ctx.get("context", "")
        base_content = list(request.system_message.content_blocks or [])
        new_content = base_content + [{"type": "text", "text": dynamic_context}]
        new_system_message = SystemMessage(content=new_content)
        return handler(request.override(system_message=new_system_message))


@lru_cache
def get_answer_agent(agent_type: AgentType):
    return create_agent(
        ChatOllama(
            model=os.getenv("OLLAMA_LLM_MODEL"),
            base_url=os.getenv("OLLAMA_BASE_URL"),
            client_kwargs={"timeout": 120},
            seed=9999,
            num_ctx=32000,
            reasoning=False,
            num_predict=-2,
            temperature=0.2,
            top_p=0.9,
        ),
        system_prompt=SYSTEM_PROMPT_REGISTRY[agent_type.value].format(),
        middleware=[
            DynamicContextMiddleware(),
            SummarizationMiddleware(
                model=ChatOllama(
                    model=os.getenv("OLLAMA_LLM_MODEL"),
                    base_url=os.getenv("OLLAMA_BASE_URL"),
                    client_kwargs={"timeout": 120},
                    seed=9999,
                    num_ctx=32000,
                    reasoning=False,
                    num_predict=1024,
                    temperature=0.3,
                    top_p=0.9,
                    tags=["nostream"],
                ),
                trigger=[("tokens", 6000), ("messages", 8)],
                keep=("messages", 4),
            ),
        ],
        name=f"answer_{agent_type.value}",
    )


@lru_cache
def get_llm_classification():
    return ChatOllama(
        model=os.getenv("OLLAMA_LLM_MODEL"),
        base_url=os.getenv("OLLAMA_BASE_URL"),
        client_kwargs={"timeout": 120},
        seed=9999,
        num_ctx=32000,
        reasoning=False,
        num_predict=256,
        temperature=0.0,
        top_p=0.9,
        tags=["nostream"],
    ).with_structured_output(AgentClassification, include_raw=True)


@lru_cache
def llm_extract_keyword():
    return ChatOllama(
        model=os.getenv("OLLAMA_LLM_MODEL"),
        base_url=os.getenv("OLLAMA_BASE_URL"),
        client_kwargs={"timeout": 120},
        seed=9999,
        num_ctx=32000,
        reasoning=False,
        num_predict=256,
        temperature=0.0,
        top_p=0.9,
        tags=["nostream"],
    ).with_structured_output(ExtractionKeyword, include_raw=True)


@lru_cache
def llm_extract_datetime():
    return ChatOllama(
        model=os.getenv("OLLAMA_LLM_MODEL"),
        base_url=os.getenv("OLLAMA_BASE_URL"),
        client_kwargs={"timeout": 120},
        seed=9999,
        num_ctx=32000,
        reasoning=False,
        num_predict=256,
        temperature=0.0,
        top_p=0.9,
        tags=["nostream"],
    ).with_structured_output(ExtractionDatetime, include_raw=True)


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
