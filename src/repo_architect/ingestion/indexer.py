import json
import time
from pathlib import Path
from typing import Dict, List
from pinecone import Pinecone, ServerlessSpec
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from tenacity import retry, stop_after_attempt, wait_exponential
from repo_architect.config import get_logger, settings

logger = get_logger(__name__)
LOCAL_CACHE_DIR = Path("data/indexed_chunks")


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": settings.device},
        encode_kwargs={"normalize_embeddings": True},
    )


def ensure_pinecone_index(pc: Pinecone) -> None:
    if not pc.has_index(settings.pinecone_index_name):
        logger.info(f"Creating Pinecone index '{settings.pinecone_index_name}'...")
        pc.create_index(
            name=settings.pinecone_index_name,
            dimension=settings.embedding_dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=settings.pinecone_cloud, region=settings.pinecone_region),
        )
        while not pc.describe_index(settings.pinecone_index_name).status["ready"]:
            time.sleep(1)


def get_vectorstore() -> PineconeVectorStore:
    pc = Pinecone(api_key=settings.pinecone_api_key)
    ensure_pinecone_index(pc)
    index = pc.Index(settings.pinecone_index_name)
    return PineconeVectorStore(index=index, embedding=get_embeddings())


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _upsert_batch(vectorstore: PineconeVectorStore, docs: List[Document], ids: List[str]) -> None:
    vectorstore.add_documents(documents=docs, ids=ids)


def index_chunks(chunks: List[Dict], repo_id: str, batch_size: int = 64) -> int:
    if not chunks:
        logger.warning("No code chunks extracted. Nothing to index.")
        return 0
    vectorstore = get_vectorstore()
    LOCAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = LOCAL_CACHE_DIR / f"{repo_id}.jsonl"
    with cache_file.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        docs = [
            Document(
                page_content=f"File: {c['metadata']['file_path']} (Lines {c['metadata']['start_line']}-{c['metadata']['end_line']})\n\n{c['text']}",
                metadata=c["metadata"],
            )
            for c in batch
        ]
        _upsert_batch(vectorstore, docs, [c["id"] for c in batch])
        logger.info(f"Indexed batch {i // batch_size + 1} ({len(docs)} chunks)")
    return len(chunks)
