import typing as t
from enum import Enum
from pathlib import Path

from ragas import EvaluationDataset
from ragas.prompt import PydanticPrompt
from ragas.testset import TestsetGenerator
from ragas.llms import BaseRagasLLM
from ragas.embeddings import BaseRagasEmbeddings
from ragas.testset.graph import KnowledgeGraph, Node
from ragas.testset.persona import Persona, PersonaList
from ragas.testset.synthesizers.single_hop.prompts import (
    QueryCondition,
    GeneratedQueryAnswer as SingleHopGeneratedQueryAnswer,
)
from ragas.testset.synthesizers.single_hop.specific import (
    SingleHopSpecificQuerySynthesizer,
)
from ragas.testset.synthesizers.multi_hop.specific import (
    MultiHopSpecificQuerySynthesizer,
)
from ragas.testset.synthesizers.multi_hop.prompts import (
    QueryConditions,
    GeneratedQueryAnswer as MultiHopGeneratedQueryAnswer,
)


class QueryLength(str, Enum):
    """
    Enumeration of query lengths. Available options are: MEDIUM, SHORT
    """

    MEDIUM = "medium"
    SHORT = "short"


class QueryStyle(str, Enum):
    """
    Enumeration of query styles. Available options are: PERFECT_GRAMMAR, POOR_GRAMMAR, WEB_SEARCH_LIKE
    """

    PERFECT_GRAMMAR = "Perfect grammar"
    POOR_GRAMMAR = "Poor grammar"
    WEB_SEARCH_LIKE = "Web search like queries"


class SingleHopQueryAnswerGenerationPrompt(
    PydanticPrompt[QueryCondition, SingleHopGeneratedQueryAnswer]
):
    instruction: str = (
        "Tạo một truy vấn và câu trả lời một bước dựa trên các điều kiện đã chỉ định (persona, term, style, length) và ngữ cảnh được cung cấp. "
        "Đảm bảo câu trả lời hoàn toàn phù hợp với ngữ cảnh, chỉ sử dụng thông tin trực tiếp từ ngữ cảnh được cung cấp.\n"
        "### Hướng dẫn:\n"
        "1. **Tạo Truy vấn**: Dựa trên ngữ cảnh, persona, term, style và length, hãy tạo một câu hỏi phù hợp với quan điểm của persona và kết hợp term.\n"
        "2. **Tạo Câu trả lời**: Chỉ sử dụng nội dung từ ngữ cảnh được cung cấp, hãy xây dựng một câu trả lời chi tiết cho truy vấn. "
        "Không thêm bất kỳ thông tin nào không có trong hoặc không thể suy ra từ ngữ cảnh.\n"
    )
    input_model: t.Type[QueryCondition] = QueryCondition
    output_model: t.Type[SingleHopGeneratedQueryAnswer] = SingleHopGeneratedQueryAnswer
    examples: t.List[t.Tuple[QueryCondition, SingleHopGeneratedQueryAnswer]] = [
        (
            QueryCondition(
                persona=Persona(
                    name="Software Engineer",
                    role_description="Tập trung vào các thực hành tốt nhất về lập trình và thiết kế hệ thống.",
                ),
                term="Bài học kinh nghiệm",
                query_style="ngữ pháp hoàn hảo",
                query_length="ngắn",
                context=(
                    "Khi triển khai kiểm tra hoạt động của module CARECONNE có thay đổi xử lý dùng chung: bước kiểm tra hoạt động của datacenter. "
                    "Hiệu chỉnh này đã thiếu sót xử lý ở kết quả trả về ở module HN, gây vấn đề Alive Monitoring gửi email thông báo lỗi ở HN mặc dù thực tế không có lỗi xảy ra."
                ),
            ),
            SingleHopGeneratedQueryAnswer(
                query="Nguyên nhân là gì dẫn đến việc Alive Monitoring gửi email thông báo lỗi đến HN?",
                answer="Nguyên nhân là do hiệu chỉnh trong module CARECONNE đã thiếu sót xử lý ở kết quả trả về ở module HN, dẫn đến việc Alive Monitoring gửi email thông báo lỗi mặc dù thực tế không có lỗi xảy ra.",
            ),
        ),
    ]


class SingleHopIncidentQuerySynthesizer(SingleHopSpecificQuerySynthesizer):

    name: str = "single_hop_specific_query_synthes"
    generate_query_reference_prompt: PydanticPrompt = (
        SingleHopQueryAnswerGenerationPrompt()
    )

    def prepare_combinations(
        self,
        node: Node,
        terms: t.List[str],
        personas: t.List[Persona],
        persona_concepts: t.Dict[str, t.List[str]],
    ) -> t.List[t.Dict[str, t.Any]]:

        sample = {"terms": terms, "node": node}
        valid_personas = []
        persona_list = PersonaList(personas=personas)
        for persona, concepts in persona_concepts.items():
            concepts = [concept.lower() for concept in concepts]
            if any(term.lower() in concepts for term in terms):
                if persona_list[persona]:
                    valid_personas.append(persona_list[persona])
        sample["personas"] = valid_personas
        sample["styles"] = list(QueryStyle)
        sample["lengths"] = list(QueryLength)

        return [sample]


class MultiHopQueryAnswerGenerationPrompt(
    PydanticPrompt[QueryConditions, MultiHopGeneratedQueryAnswer]
):
    instruction: str = (
        "Tạo truy vấn và câu trả lời đa bước dựa trên các điều kiện đã xác định (persona, themes, style, length) và ngữ cảnh được cung cấp. "
        "Các chủ đề (themes) đại diện cho một tập hợp các cụm từ được trích xuất hoặc tạo ra từ ngữ cảnh, làm nổi bật tính phù hợp của ngữ cảnh đã chọn để tạo truy vấn đa bước. "
        "Đảm bảo truy vấn tích hợp rõ ràng các chủ đề này.\n"
        "### Hướng dẫn:\n"
        "1. **Tạo truy vấn đa bước**: Sử dụng các phân đoạn ngữ cảnh và chủ đề được cung cấp để tạo một truy vấn yêu cầu kết hợp thông tin từ nhiều phân đoạn (ví dụ: `<1-hop>` và `<2-hop>`). "
        "Đảm bảo truy vấn kết hợp rõ ràng một hoặc nhiều chủ đề và phản ánh sự liên quan của chúng đến ngữ cảnh.\n"
        "2. **Tạo câu trả lời**: Chỉ sử dụng nội dung từ ngữ cảnh được cung cấp để tạo câu trả lời chi tiết và chính xác cho truy vấn. "
        "Tránh thêm thông tin không có trực tiếp hoặc không thể suy ra từ ngữ cảnh đã cho.\n"
        "3. **Thẻ ngữ cảnh đa bước**:\n"
        "  - Mỗi phân đoạn ngữ cảnh được gắn thẻ là `<1-hop>`, `<2-hop>`, v.v.\n"
        "  - Đảm bảo truy vấn sử dụng thông tin từ ít nhất hai phân đoạn và kết nối chúng một cách có ý nghĩa.\n"
    )
    input_model: t.Type[QueryConditions] = QueryConditions
    output_model: t.Type[MultiHopGeneratedQueryAnswer] = MultiHopGeneratedQueryAnswer
    examples: t.List[t.Tuple[QueryConditions, MultiHopGeneratedQueryAnswer]] = [
        (
            QueryConditions(
                persona=Persona(
                    name="Software Engineer",
                    role_description="Tập trung vào các thực hành tốt nhất về lập trình và thiết kế hệ thống.",
                ),
                themes=[
                    "Sự cố hệ thống",
                    "Lỗi logic code",
                    "Giải pháp khắc phục",
                    "Bài học kinh nghiệm",
                ],
                query_style="ngữ pháp hoàn hảo",
                query_length="vừa",
                context=[
                    (
                        "<1-hop> Mô tả sự cố: "
                        "Module CARECONNE triển khai thay đổi xử lý kiểm tra hoạt động datacenter nhưng thiếu sót xử lý kết quả trả về ở module HN, dẫn đến việc Alive Monitoring gửi email thông báo lỗi mặc dù thực tế không có sự cố."
                    ),
                    (
                        "<2-hop> Biện pháp khắc phục: "
                        "Liên lạc khách hàng và tắt cài đặt lịch kiểm tra hoạt động các module định kỳ ở môi trường STAGE; "
                        "Hiệu chỉnh mã nguồn, kiểm tra ở môi trường phát triển và gửi báo cáo kết quả kiểm tra đến khách hàng xác nhận; "
                        "Phát hành lại môi trường STAGE và mở lại cài đặt lịch kiểm tra hoạt động các module định kỳ ở môi trường STAGE;"
                    ),
                ],
            ),
            MultiHopGeneratedQueryAnswer(
                query="Nguyên nhân dẫn đến việc Alive Monitoring gửi email thông báo lỗi đến HN và biện pháp khắc phục là gì?",
                answer=(
                    "Nguyên nhân là do module CARECONNE triển khai thay đổi xử lý kiểm tra hoạt động datacenter nhưng thiếu sót xử lý kết quả trả về ở module HN, dẫn đến việc Alive Monitoring gửi email thông báo lỗi mặc dù thực tế không có sự cố. "
                    "Biện pháp khắc phục bao gồm: liên lạc khách hàng và tắt cài đặt lịch kiểm tra hoạt động các module định kỳ ở môi trường STAGE; "
                    "hiệu chỉnh mã nguồn, kiểm tra ở môi trường phát triển và gửi báo cáo kết quả kiểm tra đến khách hàng xác nhận; "
                    "phát hành lại môi trường STAGE và mở lại cài đặt lịch kiểm tra hoạt động các module định kỳ ở môi trường STAGE."
                ),
            ),
        ),
    ]


class MultiHopIncidentQuerySynthesizer(MultiHopSpecificQuerySynthesizer):

    name: str = "multi_hop_specific_query_synthes"
    generate_query_reference_prompt: PydanticPrompt = (
        MultiHopQueryAnswerGenerationPrompt()
    )

    def prepare_combinations(
        self,
        nodes,
        combinations: t.List[t.List[str]],
        personas: t.List[Persona],
        persona_item_mapping: t.Dict[str, t.List[str]],
        property_name: str,
    ) -> t.List[t.Dict[str, t.Any]]:

        persona_list = PersonaList(personas=personas)
        possible_combinations = []
        for combination in combinations:
            dict = {"combination": combination}
            valid_personas = []
            for persona, concept_list in persona_item_mapping.items():
                concept_list = [c.lower() for c in concept_list]
                if (
                    any(concept.lower() in concept_list for concept in combination)
                    and persona_list[persona]
                ):
                    valid_personas.append(persona_list[persona])
            dict["personas"] = valid_personas
            valid_nodes = []
            for node in nodes:
                node_themes = [
                    theme.lower() for theme in node.properties.get(property_name, [])
                ]
                if node.get_property(property_name) and any(
                    concept.lower() in node_themes for concept in combination
                ):
                    valid_nodes.append(node)

            dict["nodes"] = valid_nodes
            dict["styles"] = list(QueryStyle)
            dict["lengths"] = list(QueryLength)

            possible_combinations.append(dict)
        return possible_combinations


def build_testset(
    testset_size: int,
    llm: BaseRagasLLM,
    embedding_model: BaseRagasEmbeddings,
    kg: KnowledgeGraph,
    save_jsonl_path: str = "ragas_testset.jsonl",
    load_from_disk: bool = False,
) -> EvaluationDataset:

    if load_from_disk:
        return EvaluationDataset.from_jsonl(save_jsonl_path)

    generator = TestsetGenerator(
        llm=llm,
        embedding_model=embedding_model,
        knowledge_graph=kg,
        persona_list=[
            Persona(
                name="Software Engineer",
                role_description="Tập trung vào các thực hành tốt nhất về lập trình và thiết kế hệ thống.",
            ),
        ],
    )

    query_distribution = [
        (SingleHopIncidentQuerySynthesizer(llm=llm), 0.5),
        (MultiHopIncidentQuerySynthesizer(llm=llm), 0.5),
    ]
    testset = generator.generate(testset_size=testset_size, query_distribution=query_distribution)
    ragas_testset = testset.to_evaluation_dataset()
    ragas_testset.to_jsonl(Path("results", save_jsonl_path))

    return ragas_testset
