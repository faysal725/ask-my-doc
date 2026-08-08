import asyncio
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import Faithfulness, ResponseRelevancy, LLMContextRecall, LLMContextPrecisionWithReference
from langchain_openai import ChatOpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.services.hybrid_retrieval import hybrid_search
from app.services.reranker import rerank
from app.services.llm_client import generate_answer
from app.core.config import settings

from ragas.run_config import RunConfig


async def run_pipeline(question: str) -> tuple[str, list[str]]:
    hybrid_results = await hybrid_search(question, final_k=15)
    reranked = rerank(question, hybrid_results, top_k=5)
    answer = generate_answer(question, reranked, model="llama-3.1-8b-instant")
    contexts = [c["text"] for c in reranked]
    return answer, contexts


async def build_eval_dataset(testset_path: str) -> EvaluationDataset:
    with open(testset_path, "r", encoding="utf-8") as f:
        testset = json.load(f)

    samples = []
    for item in testset:
        answer, contexts = await run_pipeline(item["question"])
        samples.append({
            "user_input": item["question"],
            "retrieved_contexts": contexts,
            "response": answer,
            "reference": item["ground_truth"],
        })
        print(f"done: {item['question'][:60]}")

    return EvaluationDataset.from_list(samples)


async def main():
    testset_path = os.path.join(os.path.dirname(__file__), "qa_testset.json")
    dataset = await build_eval_dataset(testset_path)

    groq_client = ChatOpenAI(
        api_key=settings.groq_api_key,
        base_url="https://api.groq.com/openai/v1",
        model="llama-3.1-8b-instant",
        temperature=0,
        max_tokens=2048,
    )
    evaluator_llm = LangchainLLMWrapper(groq_client)

    evaluator_embeddings = LangchainEmbeddingsWrapper(
        GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=settings.gemini_api_key)
    )
    run_config = RunConfig(max_workers=3, timeout=120)
    result = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(),
            ResponseRelevancy(strictness=1),
            LLMContextRecall(),
            LLMContextPrecisionWithReference(),
        ],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        run_config=run_config,
    )

    print(result)

    df = result.to_pandas()
    output_path = os.path.join(os.path.dirname(__file__), "eval_results.csv")
    df.to_csv(output_path, index=False)
    print(f"Saved detailed results to {output_path}")

    summary_path = os.path.join(os.path.dirname(__file__), "eval_summary.json")
    try:
        summary = dict(result)
    except Exception:
        summary = {}
        for k in ["faithfulness", "answer_relevancy", "context_recall", "llm_context_precision_with_reference"]:
            try:
                v = result[k]
                summary[k] = v if v == v else None
            except Exception:
                summary[k] = None

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    asyncio.run(main())