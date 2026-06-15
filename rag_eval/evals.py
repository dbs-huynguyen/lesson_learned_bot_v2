import sys
from pathlib import Path

from langchain_core.documents import Document
from ragas import evaluate, EvaluationDataset
from ragas.llms import LangchainLLMWrapper, BaseRagasLLM
from ragas.embeddings import LangchainEmbeddingsWrapper, BaseRagasEmbeddings
from ragas.evaluation import EvaluationResult
from ragas.run_config import RunConfig
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.transforms import apply_transforms

# Add the current directory to the path so we can import rag module when run as a script
sys.path.insert(0, str(Path(__file__).parent))
from metrics import faithfulness, answer_relevancy, context_precision, context_recall
from transformers import my_transformers
from testset import build_testset
from rag import (
    RAGAgent,
    rag_pipeline,
    embeddings,
    llm,
    get_docs,
)  # noqa: E402


def build_kg(
    llm: BaseRagasLLM,
    embedding_model: BaseRagasEmbeddings,
    run_config: RunConfig,
    save_path: str = "knowledge_graph.json",
    load_from_disk: bool = False,
) -> KnowledgeGraph:

    if load_from_disk:
        return KnowledgeGraph.load(Path("results", save_path))

    docs = get_docs()
    print(f"Loaded {len(docs)} documents from vector store.")

    kg = KnowledgeGraph()
    for doc in docs:
        kg.nodes.append(
            Node(
                type=NodeType.DOCUMENT,
                properties={
                    "page_content": doc.page_content,
                    "document_metadata": doc.metadata,
                },
            )
        )

    trans = my_transformers(llm=llm, embedding_model=embedding_model)
    apply_transforms(kg, trans, run_config)

    kg.save(Path("results", save_path))

    return kg


def evaluate_ragas(
    rag_pipeline: RAGAgent,
    testset: EvaluationDataset,
    llm: BaseRagasLLM,
    embedding_model: BaseRagasEmbeddings,
    run_config: RunConfig,
    use_executor: bool = False,
    max_concurrent: int = 5,
    load_from_disk: bool = False,
    save_csv_path: str = "evaluation_results.csv",
) -> EvaluationResult:
    """
    Evaluate RAG pipeline using ragas metrics.

    Args:
        rag_pipeline: RAG pipeline instance
        testset: Evaluation dataset
        llm: LLM for evaluation
        embedding_model: Embedding model for evaluation
        run_config: Run configuration
        save_csv_path: Path to save results CSV
        use_executor: If True, use ragas Executor for progress tracking (default: False)
        max_concurrent: Maximum concurrent requests (default: 5)

    Returns:
        Evaluation results
    """
    if load_from_disk:
        testset = EvaluationDataset.from_jsonl(
            Path("results", "testset_with_responses.jsonl")
        )
    else:
        inputs = [{"question": row.user_input} for row in testset]

        if use_executor:
            results = rag_pipeline.batch_invoke_with_executor(
                inputs,
                max_workers=max_concurrent,
                show_progress=True,
            )
        else:
            results = rag_pipeline.batch_invoke(inputs, max_concurrent=max_concurrent)

        for row, answer in zip(testset, results):
            response: str = answer.get("response") or ""
            context: list[Document] = answer.get("context") or []
            retrieved_contexts = [doc.page_content for doc in context]
            row.response = response
            row.retrieved_contexts = retrieved_contexts

        testset.to_jsonl(Path("results", "testset_with_responses.jsonl"))

    results = evaluate(
        testset,
        metrics=[
            faithfulness(),
            answer_relevancy(),
            context_precision(),
            context_recall(),
        ],
        llm=llm,
        embeddings=embedding_model,
        run_config=run_config,
        raise_exceptions=True,
    )
    df = results.to_pandas()
    df.to_csv(Path("results", save_csv_path))

    return results


async def main():
    run_config = RunConfig(timeout=3600, max_retries=1, seed=9999)
    transformer_llm = LangchainLLMWrapper(llm, run_config)
    embedding_model = LangchainEmbeddingsWrapper(embeddings, run_config)

    kg = build_kg(transformer_llm, embedding_model, run_config, load_from_disk=True)
    print("Knowledge Graph:", kg)

    ragas_testset = build_testset(
        testset_size=100,
        llm=transformer_llm,
        embedding_model=embedding_model,
        kg=kg,
        load_from_disk=True,
    )

    results = evaluate_ragas(
        rag_pipeline,
        ragas_testset,
        transformer_llm,
        embedding_model,
        run_config,
        load_from_disk=True,
    )
    print("Experiment results:", results)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
