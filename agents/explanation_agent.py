"""
Explanation Agent Module
Generates the final comprehensive answer for the user.
"""

from typing import Optional


class ExplanationAgent:
    """
    Agent responsible for drafting the final, human-readable answer.
    Acts as the last step in the multi-agent pipeline.
    """

    def __init__(self, llm_client):
        """
        Args:
            llm_client: An instance of reasoning.llm_client.LLMClient.
        """
        self.llm_client = llm_client

    def run(self, summary_output: dict, query: str = "") -> dict:
        """
        Generate the final explanation.

        Args:
            summary_output: Output dict from SummaryAgent.run().
            query: The original user query.

        Returns:
            Dict with keys:
              - "query": original query
              - "answer": the final answer text
              - "sources_used": number of sources referenced
              - "confidence": self-assessed confidence level
        """
        summary = summary_output.get("summary", "")
        original_context = summary_output.get("original_context", "")
        source_count = summary_output.get("source_count", 0)

        if not summary.strip() or summary.startswith("No relevant context"):
            return {
                "query": query,
                "answer": (
                    "I couldn't find sufficient information in the research papers "
                    "to answer your question. Please try rephrasing your query or "
                    "ensure the relevant papers have been ingested."
                ),
                "sources_used": 0,
                "confidence": "low"
            }

        system_prompt = (
            "You are an expert academic research analyst. Your task is to provide "
            "a clear, well-structured, and comprehensive answer to the user's question "
            "based on the summarized research context. "
            "Requirements:\n"
            "1. Be precise and factual\n"
            "2. Cite relevant sections using [Source: paper — section] format\n"
            "3. If information is uncertain, note it explicitly\n"
            "4. Organize the answer with clear structure\n"
            "5. End with a brief confidence assessment"
        )

        prompt = (
            f"User Question: {query}\n\n"
            f"Summarized Research Context:\n{summary}\n\n"
            f"Original Retrieved Context:\n{original_context}\n\n"
            "Please provide a comprehensive, well-structured answer to the "
            "user's question. Include citations to the source sections."
        )

        try:
            answer = self.llm_client.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=1500,
                temperature=0.3
            )

            # Simple confidence heuristic based on source count
            if source_count >= 3:
                confidence = "high"
            elif source_count >= 1:
                confidence = "medium"
            else:
                confidence = "low"

        except Exception as e:
            print(f"[ExplanationAgent] Error: {e}")
            answer = f"Error generating answer: {e}"
            confidence = "error"

        return {
            "query": query,
            "answer": answer,
            "sources_used": source_count,
            "confidence": confidence
        }
