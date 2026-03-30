"""
Boundary Detector Module
Splits sections into logical chunks for embedding and retrieval.
Uses a token-based sliding window algorithm with configurable window and stride.
"""

import re
from typing import List, Optional
from dataclasses import dataclass, field


@dataclass
class TextChunk:
    """Represents a single text chunk for embedding and retrieval."""
    chunk_id: str                   # Unique identifier: "{paper_id}_{section_id}_chunk_{n}"
    paper_id: str                   # Source paper identifier
    section_heading: str            # Section this chunk belongs to
    content: str                    # The actual chunk text
    char_start: int = 0             # Start position in the section content
    char_end: int = 0               # End position in the section content
    page_numbers: List[int] = field(default_factory=list)
    token_estimate: int = 0         # Rough token count (~words * 1.3)

    def __post_init__(self):
        # Rough token estimation (1 token ≈ 4 chars for English text)
        self.token_estimate = len(self.content) // 4


class BoundaryDetector:
    """
    Splits text into chunks suitable for embedding using a true sliding window.

    Strategy:
    1. Respect section boundaries from StructureAnalyzer.
    2. Tokenize each section into words.
    3. Slide a fixed-size window across the token sequence with a configurable stride.
    4. Each window becomes one chunk, preserving overlapping context between
       consecutive chunks.

    Parameters map for backward compatibility:
        max_chunk_size  → window_size  (in number of words/tokens)
        overlap_size    → derived from stride (window_size - stride)
    """

    def __init__(
        self,
        max_chunk_size: int = 200,
        min_chunk_size: int = 50,
        overlap_size: int = 100,
        window_size: Optional[int] = None,
        stride_size: Optional[int] = None,
    ):
        """
        Args:
            max_chunk_size: Legacy parameter — used as window_size if window_size
                           is not explicitly provided. Measured in WORDS (tokens).
            min_chunk_size: Minimum number of words for a chunk to be kept.
            overlap_size:  Legacy parameter — overlap in words between chunks.
                           stride = window_size - overlap_size.
            window_size:   (Preferred) Number of words per sliding window.
            stride_size:   (Preferred) Number of words to advance per step.
        """
        self.window_size = window_size or max_chunk_size
        if stride_size is not None:
            self.stride_size = stride_size
        else:
            # Derive stride from window minus overlap
            self.stride_size = max(1, self.window_size - overlap_size)
        self.min_chunk_size = min_chunk_size

    # ─────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────

    def chunk_section(
        self,
        content: str,
        paper_id: str,
        section_id: str,
        section_heading: str,
    ) -> List[TextChunk]:
        """
        Split a section's content into chunks using a sliding window.

        Args:
            content: The section text content.
            paper_id: Identifier for the source paper.
            section_id: Section identifier.
            section_heading: Section heading for metadata.

        Returns:
            List of TextChunk objects.
        """
        if not content.strip():
            return []

        # Tokenize into words while tracking character offsets
        tokens, offsets = self._tokenize_with_offsets(content)

        if not tokens:
            return []

        chunks: List[TextChunk] = []
        chunk_idx = 0
        start = 0

        while start < len(tokens):
            end = min(start + self.window_size, len(tokens))

            # Build chunk text from token span
            chunk_text = self._tokens_to_text(content, tokens, offsets, start, end)

            # Skip tiny trailing chunks
            word_count = end - start
            if word_count < self.min_chunk_size and chunks:
                # Merge the remainder into the last chunk instead of discarding
                last = chunks[-1]
                merged_text = last.content + " " + chunk_text
                chunks[-1] = TextChunk(
                    chunk_id=last.chunk_id,
                    paper_id=paper_id,
                    section_heading=section_heading,
                    content=merged_text.strip(),
                    char_start=last.char_start,
                    char_end=offsets[end - 1][1] if end > 0 else last.char_end,
                )
                break

            char_start = offsets[start][0]
            char_end = offsets[end - 1][1] if end > 0 else char_start

            chunks.append(TextChunk(
                chunk_id=f"{paper_id}_{section_id}_chunk_{chunk_idx}",
                paper_id=paper_id,
                section_heading=section_heading,
                content=chunk_text.strip(),
                char_start=char_start,
                char_end=char_end,
            ))
            chunk_idx += 1

            # Advance by stride
            start += self.stride_size

            # If we've reached the end, stop
            if end >= len(tokens):
                break

        return chunks

    def chunk_document(
        self,
        sections: list,
        paper_id: str,
    ) -> List[TextChunk]:
        """
        Chunk all sections of a document.

        Args:
            sections: List of Section objects from StructureAnalyzer.
            paper_id: Identifier for the source paper.

        Returns:
            List of all TextChunk objects across all sections.
        """
        all_chunks: List[TextChunk] = []
        for section in sections:
            section_chunks = self.chunk_section(
                content=section.content,
                paper_id=paper_id,
                section_id=section.section_id,
                section_heading=section.heading,
            )
            all_chunks.extend(section_chunks)
        return all_chunks

    # ─────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────

    @staticmethod
    def _tokenize_with_offsets(text: str) -> tuple:
        """
        Tokenize text into words and track each word's character offsets.

        Returns:
            (tokens, offsets) where offsets is a list of (start, end) char positions.
        """
        tokens: List[str] = []
        offsets: List[tuple] = []

        for match in re.finditer(r'\S+', text):
            tokens.append(match.group())
            offsets.append((match.start(), match.end()))

        return tokens, offsets

    @staticmethod
    def _tokens_to_text(
        original: str,
        tokens: List[str],
        offsets: List[tuple],
        start_idx: int,
        end_idx: int,
    ) -> str:
        """
        Reconstruct text from a token span, preserving original whitespace.

        Args:
            original: The original text string.
            tokens: Full list of tokens.
            offsets: Full list of (char_start, char_end) per token.
            start_idx: First token index (inclusive).
            end_idx: Last token index (exclusive).

        Returns:
            The substring of the original text covering the token span.
        """
        if start_idx >= len(offsets) or end_idx <= 0:
            return ""
        char_start = offsets[start_idx][0]
        char_end = offsets[min(end_idx, len(offsets)) - 1][1]
        return original[char_start:char_end]
