import typer
from rich.console import Console
from rich.markdown import Markdown

app = typer.Typer(help="Autonomous Codebase Onboarding & Architecture Assistant")
console = Console()


@app.command()
def ingest(repo_path: str = typer.Argument(..., help="Local path to repository"), repo_id: str = "default"):
    """Parse repository AST and index into Pinecone + BM25 cache."""
    from repo_architect.ingestion.ast_parser import parse_repository
    from repo_architect.ingestion.indexer import index_chunks
    with console.status("[bold green]Parsing AST nodes with Tree-sitter..."):
        chunks = parse_repository(repo_path, repo_id)
    console.print(f"Extracted [bold]{len(chunks)}[/bold] AST code blocks.")
    with console.status("[bold blue]Embedding & uploading to Pinecone..."):
        count = index_chunks(chunks, repo_id)
    console.print(f"[bold green]Indexed {count} unique chunks under repo_id='{repo_id}'.[/bold green]")


@app.command()
def ask(question: str = typer.Argument(...), repo_id: str = "default", adapter: str | None = typer.Option(None)):
    """Query the indexed codebase using Hybrid RAG."""
    from repo_architect.rag.chain import ask_codebase
    with console.status("[bold cyan]Retrieving context & generating architectural answer..."):
        answer, _ = ask_codebase(question, repo_id=repo_id, adapter_path=adapter)
    console.print(Markdown(answer))


@app.command()
def prep_data(repo_path: str, output: str = "data/raw_commits.jsonl", max_commits: int = 400):
    """Extract (git diff -> commit message) pairs for PyTorch fine-tuning."""
    from repo_architect.training.dataset_prep import extract_commit_pairs
    count = extract_commit_pairs(repo_path, output_file=output, max_commits=max_commits)
    console.print(f"[bold green]Saved {count} training examples to {output}.[/bold green]")


@app.command()
def train(data_path: str = "data/raw_commits.jsonl", output_dir: str = "lora_code_reviewer"):
    """Fine-tune the LLM with PyTorch + LoRA and track metrics in MLflow."""
    from repo_architect.training.finetune import train_lora
    train_lora(data_path=data_path, output_dir=output_dir)
    console.print(f"[bold green]Training complete. Adapter saved to {output_dir}.[/bold green]")


if __name__ == "__main__":
    app()
