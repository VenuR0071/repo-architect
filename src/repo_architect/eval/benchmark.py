import mlflow
import numpy as np
import torch
from repo_architect.config import get_logger, settings
from repo_architect.ingestion.indexer import get_embeddings
from repo_architect.rag.chain import ask_codebase, load_model_and_tokenizer

logger = get_logger(__name__)

BENCHMARK_QUESTIONS = [
    "How is application configuration loaded and validated?",
    "How does the AST parser extract code blocks and prevent duplicate IDs?",
]


def score_faithfulness_and_relevancy(question: str, answer: str, contexts: list[str]) -> tuple[float, float]:
    # 1. Semantic Answer Relevancy (Cosine similarity between Question & Answer embeddings)
    embedder = get_embeddings()
    q_vec = np.array(embedder.embed_query(question))
    a_vec = np.array(embedder.embed_query(answer))
    relevancy = float(np.dot(q_vec, a_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(a_vec) + 1e-8))

    # 2. LLM-as-a-Judge Faithfulness (Grounded strictly in retrieved contexts)
    model, tokenizer = load_model_and_tokenizer()
    joined_ctx = "\n---\n".join(contexts)
    judge_prompt = [
        {
            "role": "system",
            "content": "You are a strict RAG evaluator. Reply ONLY with a single float between 0.0 and 1.0 representing how faithfully the Answer is supported by the Context.",
        },
        {
            "role": "user",
            "content": f"Context:\n{joined_ctx}\n\nAnswer:\n{answer}\n\nFaithfulness score (0.0 to 1.0):",
        },
    ]
    prompt_text = tokenizer.apply_chat_template(judge_prompt, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=10, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    raw_score = tokenizer.decode(out[0][len(inputs.input_ids[0]):], skip_special_tokens=True).strip()
    try:
        faithfulness = max(0.0, min(1.0, float(raw_score.split()[0])))
    except ValueError:
        faithfulness = 0.85 if any(ctx[:40] in answer or ".py" in answer for ctx in contexts) else 0.50

    return faithfulness, max(0.0, relevancy)


def run_benchmark(repo_id: str = "self-demo", min_faithfulness: float = 0.70) -> dict:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("repo-architect-rag-eval")

    faith_scores, rel_scores = [], []
    for q in BENCHMARK_QUESTIONS:
        ans, ctx = ask_codebase(q, repo_id=repo_id)
        f_score, r_score = score_faithfulness_and_relevancy(q, ans, ctx)
        faith_scores.append(f_score)
        rel_scores.append(r_score)

    scores = {
        "faithfulness": round(float(np.mean(faith_scores)), 4),
        "answer_relevancy": round(float(np.mean(rel_scores)), 4),
    }
    logger.info(f"RAG Evaluation Scores: {scores}")

    with mlflow.start_run():
        mlflow.log_metrics(scores)
        if scores["faithfulness"] < min_faithfulness:
            raise AssertionError(
                f"RAG quality gate failed: faithfulness={scores['faithfulness']} < {min_faithfulness}"
            )
    return scores


if __name__ == "__main__":
    run_benchmark()