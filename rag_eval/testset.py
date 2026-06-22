import typing as t
from enum import Enum
from pathlib import Path

from ragas import EvaluationDataset
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

from metrics.base import PydanticPrompt


class QueryLength(str, Enum):
    """
    Enumeration of query lengths. Available options are: MEDIUM, SHORT
    """

    MEDIUM = "medium"
    SHORT = "short"


class QueryStyle(str, Enum):
    """
    Enumeration of query styles. Available options are: PERFECT_GRAMMAR, WEB_SEARCH_LIKE
    """

    PERFECT_GRAMMAR = "Perfect grammar"
    WEB_SEARCH_LIKE = "Web search like queries"


class SingleHopQueryAnswerGenerationPrompt(PydanticPrompt[QueryCondition, SingleHopGeneratedQueryAnswer]):
    instruction: str = (
        "Generate a single-hop query and answer based on the specified conditions (persona, term, style, length) "
        "and the provided context. Ensure the answer is entirely faithful to the context, using only the information "
        "directly from the provided context. Always use the Vietnamese language for the query and answer.\n"
        "### Instructions:\n"
        "1. **Generate a Query**: Based on the context, persona, term, style, and length, create a question "
        "that aligns with the persona's perspective and incorporates the term, but excludes specific date information.\n"
        "2. **Generate an Answer**: Using only the content from the provided context, construct a detailed answer "
        "to the query. Do not add any information not included in or inferable from the context.\n"
    )
    input_model: t.Type[QueryCondition] = QueryCondition
    output_model: t.Type[SingleHopGeneratedQueryAnswer] = SingleHopGeneratedQueryAnswer
    examples: t.List[t.Tuple[QueryCondition, SingleHopGeneratedQueryAnswer]] = [
        (
            QueryCondition(
                persona=Persona(
                    name="Software Engineer",
                    role_description="Focuses on coding best practices and system design.",
                ),
                term="microservices",
                query_style="PERFECT_GRAMMAR",
                query_length="MEDIUM",
                context=(
                    "Microservices are an architectural style where applications are structured as a collection of loosely coupled services. "
                    "Each service is fine-grained and focuses on a single functionality."
                ),
            ),
            SingleHopGeneratedQueryAnswer(
                query="What is the purpose of microservices in software architecture?",
                answer="Microservices are designed to structure applications as a collection of loosely coupled services, each focusing on a single functionality.",
            ),
        ),
    ]


class SingleHopIncidentQuerySynthesizer(SingleHopSpecificQuerySynthesizer):
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
        "Generate a multi-hop query and answer based on the specified conditions (persona, themes, style, length) "
        "and the provided context. The themes represent a set of phrases either extracted or generated from the "
        "context, which highlight the suitability of the selected context for multi-hop query creation. Ensure the query "
        "explicitly incorporates these themes. Always use the Vietnamese language for the query and answer.\n"
        "### Instructions:\n"
        "1. **Generate a Multi-Hop Query**: Use the provided context segments and themes to form a query that requires combining "
        "information from multiple segments (e.g., `<1-hop>` and `<2-hop>`). Ensure the query explicitly incorporates one or more "
        "themes and reflects their relevance to the context, but excludes specific date information.\n"
        "2. **Generate an Answer**: Use only the content from the provided context to create a detailed and faithful answer to "
        "the query. Avoid adding information that is not directly present or inferable from the given context.\n"
        "3. **Multi-Hop Context Tags**:\n"
        "  - Each context segment is tagged as `<1-hop>`, `<2-hop>`, etc.\n"
        "  - Ensure the query uses information from at least two segments and connects them meaningfully.\n"
    )
    input_model: t.Type[QueryConditions] = QueryConditions
    output_model: t.Type[MultiHopGeneratedQueryAnswer] = MultiHopGeneratedQueryAnswer
    examples: t.List[t.Tuple[QueryConditions, MultiHopGeneratedQueryAnswer]] = [
        (
            QueryConditions(
                persona=Persona(
                    name="Historian",
                    role_description="Focuses on major scientific milestones and their global impact.",
                ),
                themes=["Theory of Relativity", "Experimental Validation"],
                query_style="PERFECT_GRAMMAR",
                query_length="MEDIUM",
                context=[
                    "<1-hop> Albert Einstein developed the theory of relativity, introducing the concept of spacetime.",
                    "<2-hop> The bending of light by gravity was confirmed during the 1919 solar eclipse, supporting Einstein’s theory.",
                ],
            ),
            MultiHopGeneratedQueryAnswer(
                query="How was the experimental validation of the theory of relativity achieved during the 1919 solar eclipse?",
                answer=(
                    "The experimental validation of the theory of relativity was achieved during the 1919 solar eclipse by confirming "
                    "the bending of light by gravity, which supported Einstein’s concept of spacetime as proposed in the theory."
                ),
            ),
        ),
    ]


class MultiHopIncidentQuerySynthesizer(MultiHopSpecificQuerySynthesizer):
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
        return EvaluationDataset.from_jsonl(Path("results", save_jsonl_path))

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
        (
            SingleHopIncidentQuerySynthesizer(
                llm=llm,
                generate_query_reference_prompt=SingleHopQueryAnswerGenerationPrompt(),
            ),
            0.7,
        ),
        (
            MultiHopIncidentQuerySynthesizer(
                llm=llm,
                generate_query_reference_prompt=MultiHopQueryAnswerGenerationPrompt(),
            ),
            0.3,
        ),
    ]
    testset = generator.generate(testset_size=testset_size, query_distribution=query_distribution)
    ragas_testset = testset.to_evaluation_dataset()
    ragas_testset.to_jsonl(Path("results", save_jsonl_path))

    return ragas_testset
