import json
from typing import Dict, List
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from repo_architect.ingestion.indexer import LOCAL_CACHE_DIR, get_vectorstore
from repo_architect.config import get_logger

logger = get_logger(__name__)


class HybridCodeRetriever:
    def __init__(self, repo_id: str, k: int = 5):
        self.repo_id = repo_id
        self.k = k
        self.vectorstore = get_vectorstore()
        self.local_docs: List[Document] = self._load_local_docs()
        self.bm25 = BM25Okapi([doc.page_content.lower().split() for doc in self.local_docs]) if self.local_docs else None

    def _load_local_docs(self) -> List[Document]:
        cache_file = LOCAL_CACHE_DIR / f"{self.repo_id}.jsonl"
        if not cache_file.exists():
            return []
        docs = []
        with cache_file.open("r", encoding="utf-8") as f:
            for line in f:
                c = json.loads(line)
                docs.append(
                    Document(
                        page_content=f"File: {c['metadata']['file_path']} (Lines {c['metadata']['start_line']}-{c['metadata']['end_line']})\n\n{c['text']}",
                        metadata={**c["metadata"], "id": c["id"]},
                    )
                )
        return docs

    def retrieve(self, query: str) -> List[Document]:
        dense_docs = self.vectorstore.similarity_search(query, k=self.k * 2, filter={"repo_id": self.repo_id})
        if not self.bm25 or not self.local_docs:
            return dense_docs[: self.k]

        bm25_scores = self.bm25.get_scores(query.lower().split())
        top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda idx: bm25_scores[idx], reverse=True)[: self.k * 2]
        sparse_docs = [self.local_docs[i] for i in top_bm25_indices if bm25_scores[i] > 0]

        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}
        for rank, doc in enumerate(dense_docs):
            key = f"{doc.metadata.get('file_path')}:{doc.metadata.get('start_line')}"
            doc_map[key] = doc
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (60 + rank + 1)
        for rank, doc in enumerate(sparse_docs):
            key = f"{doc.metadata.get('file_path')}:{doc.metadata.get('start_line')}"
            doc_map[key] = doc
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (60 + rank + 1)

        ranked_keys = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        return [doc_map[key] for key in ranked_keys[: self.k]]
