"""
Keyword Extractor Module
Extracts important keywords from text chunks for metadata storage and search.
Uses a combination of TF-based scoring and stopword filtering.
"""

import re
import math
from typing import List, Set, Dict
from collections import Counter


# Common English stopwords (subset for lightweight operation)
STOPWORDS: Set[str] = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can", "need",
    "it", "its", "this", "that", "these", "those", "he", "she", "they",
    "we", "you", "i", "me", "my", "your", "his", "her", "our", "their",
    "not", "no", "nor", "if", "then", "than", "too", "very", "so",
    "just", "about", "above", "after", "again", "all", "also", "any",
    "each", "few", "more", "most", "other", "some", "such", "only",
    "own", "same", "into", "over", "under", "up", "down", "out", "off",
    "here", "there", "when", "where", "why", "how", "what", "which",
    "who", "whom", "while", "during", "before", "between", "through",
    "both", "however", "et", "al", "fig", "figure", "table", "ref",
    "pp", "vol", "no", "see", "also", "used", "using", "based",
}


class KeywordExtractor:
    """
    Extracts keywords from text using term frequency and basic filtering.
    """

    def __init__(self, max_keywords: int = 10, min_word_length: int = 3):
        """
        Args:
            max_keywords: Maximum number of keywords to extract per chunk.
            min_word_length: Minimum character length for keyword candidates.
        """
        self.max_keywords = max_keywords
        self.min_word_length = min_word_length

    def extract(self, text: str) -> List[str]:
        """
        Extract keywords from a text.

        Args:
            text: Input text.

        Returns:
            List of keywords sorted by relevance.
        """
        if not text.strip():
            return []

        # Tokenize
        words = self._tokenize(text)

        # Filter stopwords and short words
        filtered = [
            w for w in words
            if w not in STOPWORDS and len(w) >= self.min_word_length
        ]

        # Count frequencies
        freq = Counter(filtered)

        # Score: TF * boost for capitalized/technical terms
        scored = {}
        for word, count in freq.items():
            score = count
            # Boost words that appear capitalized in original text (likely proper nouns/terms)
            if any(word.lower() == w.lower() and w[0].isupper() for w in text.split()):
                score *= 1.5
            # Boost longer words (more likely to be technical terms)
            if len(word) > 8:
                score *= 1.2
            scored[word] = score

        # Sort by score descending
        ranked = sorted(scored.items(), key=lambda x: x[1], reverse=True)

        return [word for word, _ in ranked[:self.max_keywords]]

    def extract_from_chunks(self, chunks: list) -> Dict[str, List[str]]:
        """
        Extract keywords for multiple chunks.

        Args:
            chunks: List of TextChunk objects.

        Returns:
            Dict mapping chunk_id to list of keywords.
        """
        results = {}
        for chunk in chunks:
            chunk_id = chunk.chunk_id if hasattr(chunk, "chunk_id") else str(id(chunk))
            text = chunk.content if hasattr(chunk, "content") else str(chunk)
            results[chunk_id] = self.extract(text)
        return results

    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization: lowercase, remove non-alphanumeric, split.

        Args:
            text: Input text.

        Returns:
            List of lowercase tokens.
        """
        # Keep alphanumeric and hyphens (for compound terms)
        cleaned = re.sub(r"[^a-zA-Z0-9\-]", " ", text)
        tokens = cleaned.lower().split()
        return tokens
