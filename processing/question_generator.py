"""
Question Generator Module
Generates hypothetical questions from text chunks for indexing.
These questions improve retrieval by matching user queries to pre-generated
question embeddings (HyDE-like approach).
"""

from typing import List, Dict, Optional


class QuestionGenerator:
    """
    Generates hypothetical questions that could be answered by a given text chunk.
    Uses LLM when available; falls back to template-based generation.
    """

    def __init__(self, llm_client=None, questions_per_chunk: int = 3):
        """
        Args:
            llm_client: An instance of reasoning.llm_client.LLMClient.
            questions_per_chunk: Number of questions to generate per chunk.
        """
        self.llm_client = llm_client
        self.questions_per_chunk = questions_per_chunk

    def generate(self, chunk_text: str, section_heading: str = "") -> List[str]:
        """
        Generate questions from a text chunk.

        Args:
            chunk_text: The text to generate questions from.
            section_heading: The section heading for context.

        Returns:
            List of generated question strings.
        """
        if not chunk_text.strip():
            return []

        if self.llm_client:
            return self._llm_generate(chunk_text, section_heading)
        else:
            return self._template_generate(chunk_text, section_heading)

    def generate_for_chunks(self, chunks: list) -> Dict[str, List[str]]:
        """
        Generate questions for multiple chunks.

        Args:
            chunks: List of TextChunk objects.

        Returns:
            Dict mapping chunk_id to list of generated questions.
        """
        results = {}
        for chunk in chunks:
            chunk_id = chunk.chunk_id if hasattr(chunk, "chunk_id") else str(id(chunk))
            text = chunk.content if hasattr(chunk, "content") else str(chunk)
            heading = chunk.section_heading if hasattr(chunk, "section_heading") else ""
            results[chunk_id] = self.generate(text, heading)
        return results

    def _llm_generate(self, text: str, section_heading: str) -> List[str]:
        """
        Use the LLM to generate questions.
        """
        context = f" from the '{section_heading}' section" if section_heading else ""
        prompt = (
            f"Given the following academic text{context}, generate exactly "
            f"{self.questions_per_chunk} questions that this text answers. "
            "Each question should be specific and answerable from the text. "
            "Output only the questions, one per line, without numbering.\n\n"
            f"Text:\n{text}\n\n"
            "Questions:"
        )

        try:
            response = self.llm_client.generate(prompt, max_tokens=200)
            questions = [q.strip() for q in response.strip().split("\n") if q.strip()]
            # Remove numbering if present
            cleaned = []
            for q in questions:
                import re
                q = re.sub(r"^\d+[\.\)]\s*", "", q)
                if q and q.endswith("?"):
                    cleaned.append(q)
                elif q:
                    cleaned.append(q + "?")
            return cleaned[:self.questions_per_chunk]
        except Exception as e:
            print(f"[QuestionGenerator] LLM error: {e}. Falling back to templates.")
            return self._template_generate(text, section_heading)

    def _template_generate(self, text: str, section_heading: str) -> List[str]:
        """
        Fallback: generate questions using simple templates based on section type.
        """
        questions = []

        heading_lower = section_heading.lower() if section_heading else ""

        if "abstract" in heading_lower:
            questions.append("What is the main objective of this research paper?")
            questions.append("What are the key findings of this study?")
            questions.append("What methodology was used in this research?")

        elif "introduction" in heading_lower:
            questions.append("What problem does this paper address?")
            questions.append("What is the motivation behind this research?")
            questions.append("What gaps in existing research does this paper identify?")

        elif any(w in heading_lower for w in ["method", "approach", "proposed"]):
            questions.append("What methodology is proposed in this paper?")
            questions.append("How does the proposed approach work?")
            questions.append("What are the key steps in the methodology?")

        elif any(w in heading_lower for w in ["result", "experiment", "evaluation"]):
            questions.append("What results were obtained in the experiments?")
            questions.append("How does the proposed method compare to baselines?")
            questions.append("What metrics were used for evaluation?")

        elif "conclusion" in heading_lower:
            questions.append("What are the main conclusions of this paper?")
            questions.append("What future work is suggested?")
            questions.append("What are the limitations of this study?")

        elif "related" in heading_lower or "literature" in heading_lower:
            questions.append("What prior work is discussed in this paper?")
            questions.append("How does this work differ from existing approaches?")
            questions.append("What are the key references in this research area?")

        else:
            # Generic questions
            first_sentence = text.split(".")[0] if "." in text else text[:100]
            questions.append(f"What is discussed in the section about {section_heading or 'this topic'}?")
            questions.append(f"What are the key points regarding {section_heading or 'this content'}?")
            questions.append(f"Can you explain the concepts from {section_heading or 'this section'}?")

        return questions[:self.questions_per_chunk]
