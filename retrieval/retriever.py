"""
Retriever Module
Bridges FAISS vector search with SQLite metadata for comprehensive retrieval.
Implements hybrid search: semantic (FAISS) + keyword (SQLite).
Includes cross-encoder reranking for precision improvement.
"""

import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np


@dataclass
class RetrievalResult:
    """A single retrieval result with chunk content and metadata."""
    chunk_id: str
    content: str
    section_heading: str
    paper_id: str
    paper_title: str
    summary: str
    keywords: str
    similarity_score: float = 0.0
    keyword_score: float = 0.0
    combined_score: float = 0.0
    rerank_score: float = 0.0  # cross-encoder score (0 if not reranked)


class CrossEncoderReranker:
    """
    Wraps a sentence-transformers CrossEncoder for query-document reranking.
    Loaded lazily on first use to avoid slowing down startup when not needed.
    """

    def __init__(self, model_name: Optional[str] = None):
        """
        Args:
            model_name: Cross-encoder model name.
                        Defaults to CROSS_ENCODER_MODEL env var or
                        'cross-encoder/ms-marco-MiniLM-L-6-v2'.
        """
        if model_name is None:
            model_name = os.getenv(
                "CROSS_ENCODER_MODEL",
                "cross-encoder/ms-marco-MiniLM-L-6-v2",
            )
        self.model_name = model_name
        self._model = None  # lazy-loaded

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            print(f"[CrossEncoderReranker] Loading model: {self.model_name}")
            self._model = CrossEncoder(self.model_name)
            print("[CrossEncoderReranker] Model loaded.")

    def rerank(
        self,
        query: str,
        results: List["RetrievalResult"],
        top_k: int = 5,
    ) -> List["RetrievalResult"]:
        """
        Rerank retrieval results using a cross-encoder.

        Args:
            query: The user query.
            results: List of RetrievalResult from the initial retrieval.
            top_k: Number of results to keep after reranking.

        Returns:
            Reranked list of RetrievalResult (top_k best).
        """
        if not results:
            return results

        self._load()

        # Build (query, document) pairs
        pairs = [(query, r.content) for r in results]

        # Score all pairs in one batch
        scores = self._model.predict(pairs)

        # Assign scores and sort
        for result, score in zip(results, scores):
            result.rerank_score = float(score)

        results.sort(key=lambda r: r.rerank_score, reverse=True)
        return results[:top_k]


class Retriever:
    """
    Hybrid retriever combining semantic search (FAISS) and keyword search (SQLite).
    Optionally applies cross-encoder reranking for improved precision.
    """

    def __init__(self, faiss_store, metadata_db, embedder, use_reranker: bool = True):
        """
        Initialize the retriever.

        Args:
            faiss_store: An instance of vectorstore.faiss_store.FAISSStore.
            metadata_db: An instance of database.metadata_db.MetadataDB.
            embedder: An instance of embeddings.embedder.Embedder.
            use_reranker: Whether to enable cross-encoder reranking.
        """
        self.faiss_store = faiss_store
        self.metadata_db = metadata_db
        self.embedder = embedder
        self.reranker = CrossEncoderReranker() if use_reranker else None

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
        paper_id: Optional[str] = None
    ) -> List[RetrievalResult]:
        """
        Perform hybrid retrieval: semantic + keyword search, then cross-encoder reranking.

        Args:
            query: User query text.
            top_k: Number of results to return.
            semantic_weight: Weight for semantic similarity scores (0-1).
            keyword_weight: Weight for keyword match scores (0-1).
            paper_id: Optional filter to restrict to a specific paper.

        Returns:
            List of RetrievalResult sorted by combined/reranked score.
        """
        # Retrieve a wider candidate set for reranking
        candidate_k = top_k * 3 if self.reranker else top_k * 2

        # 1. Semantic search via FAISS
        query_embedding = self.embedder.embed_text(query)
        semantic_results = self.faiss_store.search(query_embedding, top_k=candidate_k)

        # 2. Keyword search via SQLite
        query_keywords = self._extract_query_keywords(query)
        keyword_results = []
        if query_keywords:
            keyword_results = self.metadata_db.search_by_keywords(
                query_keywords, limit=candidate_k
            )

        # 3. Merge and score
        results = self._merge_results(
            semantic_results=semantic_results,
            keyword_results=keyword_results,
            semantic_weight=semantic_weight,
            keyword_weight=keyword_weight,
            paper_id=paper_id
        )

        # 4. Sort by combined score
        results.sort(key=lambda r: r.combined_score, reverse=True)

        # 5. Cross-encoder reranking (if enabled)
        if self.reranker and results:
            results = self.reranker.rerank(query, results, top_k=top_k)
        else:
            results = results[:top_k]

        return results

    def retrieve_semantic(
        self,
        query: str,
        top_k: int = 5
    ) -> List[RetrievalResult]:
        """
        Perform pure semantic (vector) search with optional cross-encoder reranking.

        Args:
            query: User query text.
            top_k: Number of results.

        Returns:
            List of RetrievalResult sorted by similarity/reranked score.
        """
        candidate_k = top_k * 3 if self.reranker else top_k

        query_embedding = self.embedder.embed_text(query)
        search_results = self.faiss_store.search(query_embedding, top_k=candidate_k)

        results = []
        chunk_ids = [cid for cid, _ in search_results]
        chunk_data = self.metadata_db.get_chunks_by_ids(chunk_ids)
        chunk_map = {c["chunk_id"]: c for c in chunk_data}

        for chunk_id, score in search_results:
            meta = chunk_map.get(chunk_id, {})
            results.append(RetrievalResult(
                chunk_id=chunk_id,
                content=meta.get("content", ""),
                section_heading=meta.get("section_heading", ""),
                paper_id=meta.get("paper_id", ""),
                paper_title=self._get_paper_title(meta.get("paper_id", "")),
                summary=meta.get("summary", ""),
                keywords=meta.get("keywords", ""),
                similarity_score=score,
                combined_score=score
            ))

        # Cross-encoder reranking
        if self.reranker and results:
            results = self.reranker.rerank(query, results, top_k=top_k)
        else:
            results = results[:top_k]

        return results

    def _merge_results(
        self,
        semantic_results: List[Tuple[str, float]],
        keyword_results: List[Dict[str, Any]],
        semantic_weight: float,
        keyword_weight: float,
        paper_id: Optional[str]
    ) -> List[RetrievalResult]:
        """Merge semantic and keyword results with weighted scoring."""

        # Collect all unique chunk IDs
        all_chunk_ids = set()
        semantic_scores = {}
        keyword_ids = set()

        for cid, score in semantic_results:
            all_chunk_ids.add(cid)
            semantic_scores[cid] = score

        for result in keyword_results:
            cid = result["chunk_id"]
            all_chunk_ids.add(cid)
            keyword_ids.add(cid)

        # Fetch metadata for all chunks
        chunk_data = self.metadata_db.get_chunks_by_ids(list(all_chunk_ids))
        chunk_map = {c["chunk_id"]: c for c in chunk_data}

        results = []
        for cid in all_chunk_ids:
            meta = chunk_map.get(cid, {})

            # Apply paper filter
            if paper_id and meta.get("paper_id") != paper_id:
                continue

            sem_score = semantic_scores.get(cid, 0.0)
            kw_score = 1.0 if cid in keyword_ids else 0.0
            combined = (semantic_weight * sem_score) + (keyword_weight * kw_score)

            results.append(RetrievalResult(
                chunk_id=cid,
                content=meta.get("content", ""),
                section_heading=meta.get("section_heading", ""),
                paper_id=meta.get("paper_id", ""),
                paper_title=self._get_paper_title(meta.get("paper_id", "")),
                summary=meta.get("summary", ""),
                keywords=meta.get("keywords", ""),
                similarity_score=sem_score,
                keyword_score=kw_score,
                combined_score=combined
            ))

        return results

    def _extract_query_keywords(self, query: str) -> List[str]:
        """Extract simple keywords from a query for SQLite search."""
        import re
        # Common English stopwords to ignore
        stopwords = {
            "what", "is", "the", "how", "does", "do", "are", "was", "were",
            "can", "could", "would", "should", "will", "may", "might",
            "a", "an", "in", "on", "at", "to", "for", "of", "with", "by",
            "from", "about", "this", "that", "these", "those", "and", "or"
        }
        words = re.findall(r'\b[a-zA-Z]{3,}\b', query.lower())
        return [w for w in words if w not in stopwords]

    def _get_paper_title(self, paper_id: str) -> str:
        """Look up paper title from the database."""
        if not paper_id:
            return ""
        papers = self.metadata_db.list_papers()
        for paper in papers:
            if paper["paper_id"] == paper_id:
                return paper.get("title", "")
        return ""
