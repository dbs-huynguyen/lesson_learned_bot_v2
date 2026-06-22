import json
import typing as t

from pydantic import BaseModel
from ragas.prompt import PydanticPrompt
from ragas.llms import BaseRagasLLM
from ragas.embeddings import BaseRagasEmbeddings
from ragas.utils import num_tokens_from_string
from ragas.testset.graph import Node
from ragas.testset.transforms import Transforms, Parallel
from ragas.testset.transforms.filters import CustomNodeFilter
from ragas.testset.transforms.extractors import EmbeddingExtractor, SummaryExtractor
from ragas.testset.transforms.extractors.llm_based import NERExtractor
from ragas.testset.transforms.relationship_builders import (
    CosineSimilarityBuilder,
    OverlapScoreBuilder,
)


class StringIO(BaseModel):
    text: str

    def __hash__(self):
        return hash(self.text)


class SummaryExtractorPrompt(PydanticPrompt[StringIO, StringIO]):
    instruction: str = "Summarize the given text in less than 10 sentences. Always use the same language as the input text for the summary."
    input_model: t.Type[StringIO] = StringIO
    output_model: t.Type[StringIO] = StringIO
    examples: t.List[t.Tuple[StringIO, StringIO]] = [
        (
            StringIO(
                text=(
                    "Trí tuệ nhân tạo\n\n"
                    "Trí tuệ nhân tạo đang biến đổi nhiều ngành công nghiệp bằng cách tự động hóa các nhiệm vụ trước đây yêu cầu trí tuệ con người. "
                    "Từ chăm sóc sức khỏe đến tài chính, AI đang được sử dụng để phân tích khối lượng lớn dữ liệu một cách nhanh chóng và chính xác. "
                    "Công nghệ này cũng đang thúc đẩy các đổi mới trong các lĩnh vực như xe tự lái và đề xuất cá nhân hóa."
                )
            ),
            StringIO(
                text=(
                    "AI đang cách mạng hóa các ngành công nghiệp bằng cách tự động hóa các nhiệm vụ, phân tích dữ liệu và thúc đẩy các đổi mới như xe tự lái và đề xuất cá nhân hóa."
                )
            ),
        )
    ]

    def _generate_output_signature(self, indent: int = 4) -> str:
        return (
            f"Please return the output in a JSON format that complies with the "
            f"following schema as specified in JSON Schema:\n"
            f"{json.dumps(self.output_model.model_json_schema())}\n"
            "Do not use single quotes in your response but double quotes,"
            "properly escaped with a backslash."
        )


class TextWithExtractionLimit(BaseModel):
    text: str
    max_num: int = 10


class NEROutput(BaseModel):
    entities: t.List[str]


class NERPrompt(PydanticPrompt[TextWithExtractionLimit, NEROutput]):
    instruction: str = (
        "Extract the named entities from the given text, limiting the output to the top entities. "
        "Ensure the number of entities does not exceed the specified maximum.\n"
        "Do not extract time and task-code entities."
    )
    input_model: t.Type[TextWithExtractionLimit] = TextWithExtractionLimit
    output_model: t.Type[NEROutput] = NEROutput
    examples: t.List[t.Tuple[TextWithExtractionLimit, NEROutput]] = [
        (
            TextWithExtractionLimit(
                text=(
                    "Elon Musk, the CEO of Tesla and SpaceX, announced plans to expand operations to new locations in Europe and Asia. "
                    "This expansion is expected to create thousands of jobs, particularly in cities like Berlin and Shanghai."
                ),
                max_num=10,
            ),
            NEROutput(
                entities=[
                    "Elon Musk",
                    "Tesla",
                    "SpaceX",
                    "Europe",
                    "Asia",
                    "Berlin",
                    "Shanghai",
                ]
            ),
        )
    ]

    def _generate_output_signature(self, indent: int = 4) -> str:
        return (
            f"Please return the output in a JSON format that complies with the "
            f"following schema as specified in JSON Schema:\n"
            f"{json.dumps(self.output_model.model_json_schema())}\n"
            "Do not use single quotes in your response but double quotes,"
            "properly escaped with a backslash."
        )


def my_transformers(
    llm: BaseRagasLLM, embedding_model: BaseRagasEmbeddings
) -> Transforms:

    def filter_doc_with_num_tokens(node: Node, min_num_tokens: int = 500):
        return num_tokens_from_string(node.properties["page_content"]) > min_num_tokens

    summary_extractor = SummaryExtractor(
        llm=llm,
        property_name="summary",
        filter_nodes=lambda node: filter_doc_with_num_tokens(node, 100),
        prompt=SummaryExtractorPrompt(),
    )
    node_filter = CustomNodeFilter(llm=llm)
    summary_emb_extractor = EmbeddingExtractor(
        embedding_model=embedding_model,
        property_name="summary_embedding",
        embed_property_name="summary",
        filter_nodes=lambda node: filter_doc_with_num_tokens(node, 100),
    )
    ner_extractor = NERExtractor(
        llm=llm,
        property_name="entities",
        prompt=NERPrompt(),
    )
    cosine_sim_builder = CosineSimilarityBuilder(
        property_name="summary_embedding",
        new_property_name="summary_similarity",
        threshold=0.5,
        filter_nodes=lambda node: filter_doc_with_num_tokens(node, 100),
    )
    ner_overlap_sim = OverlapScoreBuilder(
        threshold=0.01,
        property_name="entities",
        new_property_name="overlap_score",
    )

    transforms = [
        summary_extractor,
        node_filter,
        Parallel(summary_emb_extractor, ner_extractor),
        Parallel(cosine_sim_builder, ner_overlap_sim),
    ]

    return transforms
