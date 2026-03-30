"""
Summary Generator Module
Generates concise summaries for text chunks to improve retrieval quality.
Uses the Groq LLM API for abstractive summarization.
"""

from typing import List, Optional


class SummaryGenerator:
    """
    Generates summaries for text chunks using an LLM.
    Summaries are stored alongside the original chunks to enhance retrieval.
    """

    def __init__(self, llm_client=None):
        """
        Initialize the summary generator.

        Args:
            llm_client: An instance of reasoning.llm_client.LLMClient.
                       If None, a fallback extractive method is used.
        """
        self.llm_client = llm_client

    def summarize_chunk(self, chunk_text: str) -> str:
        """
        Generate a summary for a single text chunk.

        Args:
            chunk_text: The text to summarize.

        Returns:
            A concise summary string.
        """
        if not chunk_text.strip():
            return ""

        if self.llm_client:
            return self._llm_summarize(chunk_text)
        else:
            return self._extractive_summarize(chunk_text)

    def summarize_chunks(self, chunks: list) -> List[str]:
        """
        Generate summaries for a list of TextChunk objects.

        Args:
            chunks: List of TextChunk objects.

        Returns:
            List of summary strings, one per chunk.
        """
        summaries = []
        for chunk in chunks:
            text = chunk.content if hasattr(chunk, "content") else str(chunk)
            summary = self.summarize_chunk(text)
            summaries.append(summary)
        return summaries

    def _llm_summarize(self, text: str) -> str:
        """
        Use the LLM client for abstractive summarization.

        Args:
            text: Text to summarize.

        Returns:
            LLM-generated summary.
        """
        prompt = (
            "Summarize the following academic text in 2-3 concise sentences. "
            "Focus on the key findings, methods, or concepts.\n\n"
            f"Text:\n{text}\n\n"
            "Summary:"
        )

        try:
            response = self.llm_client.generate(prompt, max_tokens=150)
            return response.strip()
        except Exception as e:
            print(f"[SummaryGenerator] LLM error: {e}. Falling back to extractive.")
            return self._extractive_summarize(text)

    def _extractive_summarize(self, text: str, num_sentences: int = 3) -> str:
        """
        Fallback: extractive summarization by selecting the first N sentences.

        Args:
            text: Text to summarize.
            num_sentences: Number of sentences to extract.

        Returns:
            The first N sentences as a summary.
        """
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        selected = sentences[:num_sentences]
        return " ".join(selected)
