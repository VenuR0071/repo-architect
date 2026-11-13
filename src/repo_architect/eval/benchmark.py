import mlflow
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, faithfulness
from repo_architect.config import settings
from repo_architect.rag.chain import ask_codebase

BENCHMARK_QUESTIONS = ["How is application configuration loaded and validated?", "How does the AST parser extract code blocks?"]


def run_benchmark(repo_id: str = "default", min_faithfulness: float = 0.70) -> dict:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("repo-architect-rag-eval")
    questions, answers, contexts = [], [], []
    for q in BENCHMARK_QUESTIONS:
        ans, ctx = ask_codebase(q, repo_id=repo_id)
        questions.append(q)
        answers.append(ans)
        contexts.append(ctx)
    results = evaluate(Dataset.from_dict({"question": questions, "answer": answers, "contexts": contexts}), metrics=[faithfulness, answer_relevancy])
    scores = {k: float(v) for k, v in results.items()}
    with mlflow.start_run():
        mlflow.log_metrics(scores)
        assert scores.get("faithfulness", 0.0) >= min_faithfulness, f"RAG quality gate failed: {scores}"
    return scores
