# Engineering Build Log & AI-Assisted Velocity

**Objective:** Design, build, test, and containerize a production-grade AST-aware Hybrid RAG and PyTorch LoRA codebase assistant.

---

## Sprint Timeline & Execution

### Phase 1: Core Architecture & AST Ingestion
* **Deliverables:** `pyproject.toml`, `config.py`, `ast_parser.py`, `indexer.py`
* **Key Decisions:**
  * Selected prebuilt official `tree-sitter-*` wheels (`python`, `java`, `javascript`) over legacy `tree-sitter-languages` to eliminate C++ compiler toolchain friction on Windows 11.
  * Standardized all relative file paths with `.as_posix()` and raw binary byte slicing (`file_path.read_bytes()`) to prevent CRLF (`\r\n`) byte-offset drift.
  * Wrapped Pinecone batch upserts in `tenacity` exponential backoff retries:

```python
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _upsert_batch(vectorstore: PineconeVectorStore, docs: List[Document], ids: List[str]) -> None:
    vectorstore.add_documents(documents=docs, ids=ids)
```

* **AI-Assisted Workflow:** Used AI pair-programming to scaffold Pydantic v2 `BaseSettings` validation, Tree-sitter node traversal boilerplate, and pytest fixtures—cutting initial setup time by ~60%.

### Phase 2: Hybrid Retrieval (RRF) & CLI UX
* **Deliverables:** `retriever.py`, `chain.py`, `cli.py`
* **Key Decisions:**
  * Implemented Reciprocal Rank Fusion (RRF) combining Pinecone cosine similarity with local `BM25Okapi` scoring to handle both conceptual architectural queries and exact function/variable lookups.
  * Refactored `cli.py` to use lazy subcommand imports after identifying eager `GitPython` and `PyTorch` import overhead during CLI startup.
  * Enforced strict grounding prompts requiring exact `file_path` and `start_line-end_line` citations.

### Phase 3: PyTorch LoRA Fine-Tuning & MLOps Pipeline
* **Deliverables:** `dataset_prep.py`, `finetune.py`, `benchmark.py`, `docker-compose.yml`, `.github/workflows/eval_pipeline.yml`
* **Key Decisions:**
  * Built an automated dataset generator extracting clean `(unified git diff -> commit message)` pairs while filtering out merge commits and oversized generated diffs.
  * Configured `TrainingArguments` with `dataloader_num_workers=0` and dynamic `bf16`/`fp16` detection for rock-solid execution on Windows 11 CUDA GPUs.
  * Integrated **MLflow** artifact logging for LoRA adapters and **RAGAS** faithfulness assertions as an automated quality gate.

---

## Verification Checklist

- [x] `pytest -v tests/` passes AST extraction assertions across classes and functions.
- [x] `repo-arch ingest . --repo-id self-demo` indexes repository cleanly into Pinecone + local BM25 cache.
- [x] `repo-arch ask` returns accurate architectural answers with exact line-number citations.
- [x] `docker compose up -d mlflow` launches local experiment tracking server.
- [x] GitHub Actions CI workflow validates package installation and unit tests on push.