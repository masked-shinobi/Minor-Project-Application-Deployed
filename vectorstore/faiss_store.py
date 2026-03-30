"""
FAISS Vector Store Module
Manages a FAISS index for storing and searching text chunk embeddings.
"""

import os
import numpy as np
from typing import List, Tuple, Optional

try:
    import faiss
except ImportError:
    raise ImportError("faiss-cpu is required. Install via: pip install faiss-cpu")


class FAISSStore:
    """
    FAISS-backed vector store for semantic similarity search.
    Uses IndexFlatIP (Inner Product / Cosine Similarity with normalized vectors).
    """

    def __init__(self, embedding_dim: int = 384):
        """
        Initialize the FAISS store.

        Args:
            embedding_dim: Dimension of the embedding vectors (384 for all-MiniLM-L6-v2).
        """
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatIP(embedding_dim)  # Inner Product for cosine sim
        self.chunk_ids: List[str] = []  # Mapping from FAISS index position to chunk_id
        self._id_to_idx: dict = {}      # Reverse mapping

    @property
    def size(self) -> int:
        """Number of vectors in the index."""
        return self.index.ntotal

    def add_embeddings(
        self,
        embeddings: np.ndarray,
        chunk_ids: List[str]
    ) -> None:
        """
        Add embeddings to the FAISS index.

        Args:
            embeddings: 2D numpy array of shape (n, embedding_dim).
            chunk_ids: List of chunk IDs corresponding to each embedding.

        Raises:
            ValueError: If embeddings and chunk_ids have mismatched lengths.
        """
        if len(embeddings) != len(chunk_ids):
            raise ValueError(
                f"Mismatch: {len(embeddings)} embeddings vs {len(chunk_ids)} chunk_ids"
            )

        if len(embeddings) == 0:
            return

        # Normalize vectors for cosine similarity via inner product
        normalized = self._normalize(embeddings)

        # Track the starting index
        start_idx = len(self.chunk_ids)

        self.index.add(normalized)

        for i, cid in enumerate(chunk_ids):
            self.chunk_ids.append(cid)
            self._id_to_idx[cid] = start_idx + i

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Search for the most similar embeddings to a query.

        Args:
            query_embedding: 1D numpy array of shape (embedding_dim,).
            top_k: Number of results to return.

        Returns:
            List of (chunk_id, similarity_score) tuples, sorted by descending score.
        """
        if self.index.ntotal == 0:
            return []

        # Reshape and normalize
        query = query_embedding.reshape(1, -1).astype(np.float32)
        query = self._normalize(query)

        # Search
        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.chunk_ids):
                results.append((self.chunk_ids[idx], float(score)))

        return results

    def save(self, directory: str, index_name: str = "faiss_index") -> None:
        """
        Save the FAISS index and chunk ID mapping to disk.

        Args:
            directory: Directory to save files in.
            index_name: Base name for the saved files.
        """
        os.makedirs(directory, exist_ok=True)

        index_path = os.path.join(directory, f"{index_name}.bin")
        ids_path = os.path.join(directory, f"{index_name}_ids.txt")

        faiss.write_index(self.index, index_path)

        with open(ids_path, "w", encoding="utf-8") as f:
            for cid in self.chunk_ids:
                f.write(cid + "\n")

        print(f"[FAISSStore] Saved index ({self.size} vectors) to {directory}")

    def load(self, directory: str, index_name: str = "faiss_index") -> None:
        """
        Load a FAISS index and chunk ID mapping from disk.

        Args:
            directory: Directory containing the saved files.
            index_name: Base name of the saved files.

        Raises:
            FileNotFoundError: If index files are not found.
        """
        index_path = os.path.join(directory, f"{index_name}.bin")
        ids_path = os.path.join(directory, f"{index_name}_ids.txt")

        if not os.path.isfile(index_path):
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        if not os.path.isfile(ids_path):
            raise FileNotFoundError(f"Chunk IDs file not found: {ids_path}")

        self.index = faiss.read_index(index_path)

        with open(ids_path, "r", encoding="utf-8") as f:
            self.chunk_ids = [line.strip() for line in f if line.strip()]

        self._id_to_idx = {cid: i for i, cid in enumerate(self.chunk_ids)}

        print(f"[FAISSStore] Loaded index ({self.size} vectors) from {directory}")

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        """L2-normalize vectors for cosine similarity via inner product."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)  # Avoid division by zero
        return vectors / norms
