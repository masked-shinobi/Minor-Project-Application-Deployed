"""
Embedder Module
Wrapper around sentence-transformers for generating text embeddings.
Default model: all-MiniLM-L6-v2
"""

import os
import numpy as np
from typing import List, Optional, Union

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError(
        "sentence-transformers is required. "
        "Install via: pip install sentence-transformers"
    )


class Embedder:
    """
    Generates dense vector embeddings for text using sentence-transformers.
    """

    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the embedder.

        Args:
            model_name: Name of the sentence-transformers model. 
                       Defaults to 'all-MiniLM-L6-v2' or EMBEDDING_MODEL env var.
        """
        if model_name is None:
            model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

        self.model_name = model_name
        print(f"[Embedder] Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"[Embedder] Model loaded. Embedding dimension: {self.embedding_dim}")

    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate an embedding for a single text string.

        Args:
            text: Input text to embed.

        Returns:
            1D numpy array of shape (embedding_dim,).
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.astype(np.float32)

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of input texts.
            batch_size: Batch size for encoding.

        Returns:
            2D numpy array of shape (len(texts), embedding_dim).
        """
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self.embedding_dim)

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 10,
            convert_to_numpy=True
        )
        return embeddings.astype(np.float32)

    def embed_chunks(self, chunks: list, batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for a list of TextChunk objects.

        Args:
            chunks: List of TextChunk objects (must have .content attribute).
            batch_size: Batch size for encoding.

        Returns:
            2D numpy array of shape (len(chunks), embedding_dim).
        """
        texts = [chunk.content for chunk in chunks]
        return self.embed_texts(texts, batch_size=batch_size)
