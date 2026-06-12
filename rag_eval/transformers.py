import typing as t

from pydantic import BaseModel
from ragas.prompt import PydanticPrompt, StringIO
from ragas.llms import BaseRagasLLM
from ragas.embeddings import BaseRagasEmbeddings
from ragas.utils import num_tokens_from_string
from ragas.testset.graph import Node
from ragas.testset.transforms import Transforms, Parallel
from ragas.testset.transforms.filters import CustomNodeFilter
from ragas.testset.transforms.extractors import EmbeddingExtractor, SummaryExtractor
from ragas.testset.transforms.extractors.llm_based import NERExtractor, ThemesExtractor
from ragas.testset.transforms.relationship_builders import (
    CosineSimilarityBuilder,
    OverlapScoreBuilder,
)


class TextWithExtractionLimit(BaseModel):
    text: str
    max_num: int = 10


class NEROutput(BaseModel):
    entities: t.List[str]


class NERPrompt(PydanticPrompt[TextWithExtractionLimit, NEROutput]):
    instruction: str = (
        "Trích xuất các thực thể được đặt tên từ văn bản đã cho, giới hạn kết quả chỉ ở những thực thể hàng đầu. "
        "Đảm bảo số lượng thực thể không vượt quá số lượng tối đa được chỉ định."
    )
    input_model: t.Type[TextWithExtractionLimit] = TextWithExtractionLimit
    output_model: t.Type[NEROutput] = NEROutput
    examples: t.List[t.Tuple[TextWithExtractionLimit, NEROutput]] = [
        (
            TextWithExtractionLimit(
                text=(
                    "Khi triển khai kiểm tra hoạt động của module CARECONNE có thay đổi xử lý dùng chung: bước kiểm tra hoạt động của datacenter. "
                    "Hiệu chỉnh này đã thiếu sót xử lý ở kết quả trả về ở module HN, gây vấn đề Alive Monitoring gửi email thông báo lỗi ở HN mặc dù thực tế không có lỗi xảy ra."
                ),
                max_num=5,
            ),
            NEROutput(
                entities=[
                    "module CARECONNE",
                    "datacenter",
                    "module HN",
                    "Alive Monitoring",
                    "email thông báo lỗi",
                ]
            ),
        ),
    ]


class ThemesAndConcepts(BaseModel):
    output: t.List[str]


class ThemesAndConceptsExtractorPrompt(
    PydanticPrompt[TextWithExtractionLimit, ThemesAndConcepts]
):
    instruction: str = "Trích xuất các chủ đề và khái niệm chính từ văn bản đã cho."
    input_model: t.Type[TextWithExtractionLimit] = TextWithExtractionLimit
    output_model: t.Type[ThemesAndConcepts] = ThemesAndConcepts
    examples: t.List[t.Tuple[TextWithExtractionLimit, ThemesAndConcepts]] = [
        (
            TextWithExtractionLimit(
                text=(
                    "Khi triển khai kiểm tra hoạt động của module CARECONNE có thay đổi xử lý dùng chung: bước kiểm tra hoạt động của datacenter. "
                    "Hiệu chỉnh này đã thiếu sót xử lý ở kết quả trả về ở module HN, gây vấn đề Alive Monitoring gửi email thông báo lỗi ở HN mặc dù thực tế không có lỗi xảy ra."
                ),
                max_num=5,
            ),
            ThemesAndConcepts(
                output=[
                    "kiểm tra hoạt động",
                    "module CARECONNE",
                    "datacenter",
                    "module HN",
                    "Alive Monitoring",
                    "email thông báo lỗi",
                ]
            ),
        )
    ]


class SummaryExtractorPrompt(PydanticPrompt[StringIO, StringIO]):
    instruction: str = "Tóm tắt đoạn văn đã cho trong vòng chưa đến 10 câu."
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
        max_num_entities=5,
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
    theme_extractor = ThemesExtractor(
        llm=llm,
        property_name="themes",
        prompt=ThemesAndConceptsExtractorPrompt(),
        max_num_themes=5,
    )

    transforms = [
        summary_extractor,
        node_filter,
        Parallel(summary_emb_extractor, ner_extractor),
        Parallel(cosine_sim_builder, ner_overlap_sim, theme_extractor),
    ]

    return transforms
