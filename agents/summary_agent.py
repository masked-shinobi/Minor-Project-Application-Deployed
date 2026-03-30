"""
Summary Agent Module
Summarizes retrieved context for the final explanation agent.
Outputs structured JSON summaries for downstream consumption.
"""

import json
from typing import Optional, Dict, Any, List


class SummaryAgent:
    """
    Agent responsible for condensing retrieved passages into a coherent,
    structured summary. Acts as the second step in the multi-agent pipeline.

    Output schema:
        {
            "query": str,
            "summary": str,                    # narrative overview
            "structured": {
                "key_claims": [str, ...],       # main assertions from the sources
                "methodologies": [str, ...],    # methods / techniques mentioned
                "limitations": [str, ...],      # caveats or limitations noted
                "source_citations": [str, ...], # [Paper — Section] references
            },
            "source_count": int,
            "original_context": str,
        }
    """

    STRUCTURED_SYSTEM_PROMPT = (
        "You are an expert academic summarizer. Your task is to condense "
        "multiple retrieved passages into a coherent, well-organized summary "
        "that preserves key information and is relevant to the user's query.\n\n"
        "You MUST respond with ONLY a valid JSON object (no markdown fences, "
        "no extra text) with these keys:\n"
        '  "summary": a 3-5 sentence narrative overview addressing the query\n'
        '  "key_claims": a JSON array of 3-5 main assertions from the passages\n'
        '  "methodologies": a JSON array of methods/techniques mentioned (may be empty)\n'
        '  "limitations": a JSON array of caveats or limitations noted (may be empty)\n'
        '  "source_citations": a JSON array of "[Paper — Section]" references used\n'
    )

    def __init__(self, llm_client):
        """
        Args:
            llm_client: An instance of reasoning.llm_client.LLMClient.
        """
        self.llm_client = llm_client

    def run(self, retrieval_output: dict, query: str = "") -> dict:
        """
        Execute the summary agent on retrieval results.

        Args:
            retrieval_output: Output dict from RetrievalAgent.run().
            query: The original user query for context.

        Returns:
            Dict with keys: query, summary, structured, source_count, original_context.
        """
        context = retrieval_output.get("context", "")
        num_results = retrieval_output.get("num_results", 0)

        if not context.strip():
            return {
                "query": query,
                "summary": "No relevant context was found for this query.",
                "structured": {
                    "key_claims": [],
                    "methodologies": [],
                    "limitations": [],
                    "source_citations": [],
                },
                "source_count": 0,
                "original_context": "",
            }

        # ── Try structured LLM summarisation ──
        structured = self._llm_structured_summarize(query, context)

        if structured is not None:
            return {
                "query": query,
                "summary": structured.get("summary", ""),
                "structured": {
                    "key_claims": structured.get("key_claims", []),
                    "methodologies": structured.get("methodologies", []),
                    "limitations": structured.get("limitations", []),
                    "source_citations": structured.get("source_citations", []),
                },
                "source_count": num_results,
                "original_context": context,
            }

        # ── Fallback: plain-text LLM summary ──
        summary = self._plain_summarize(query, context)

        return {
            "query": query,
            "summary": summary,
            "structured": {
                "key_claims": [],
                "methodologies": [],
                "limitations": [],
                "source_citations": [],
            },
            "source_count": num_results,
            "original_context": context,
        }

    # ───────────────────────────────────────
    # Structured LLM summary
    # ───────────────────────────────────────

    def _llm_structured_summarize(
        self, query: str, context: str
    ) -> Optional[Dict[str, Any]]:
        """
        Prompt the LLM to return a structured JSON summary.

        Returns:
            Parsed dict on success, or None on failure.
        """
        # Truncate context to stay within 6000 token limit
        context_truncated = context[:5000] if len(context) > 5000 else context

        prompt = (
            f"User Query: {query}\n\n"
            f"Retrieved Passages:\n{context_truncated}\n\n"
            "Produce the JSON summary now."
        )

        try:
            response = self.llm_client.generate(
                prompt=prompt,
                system_prompt=self.STRUCTURED_SYSTEM_PROMPT,
                max_tokens=1000,
                temperature=0.2,
            )

            # Strip markdown fences if present
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[-1]
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]
                cleaned = cleaned.strip()

            parsed = json.loads(cleaned)

            # Ensure all expected keys exist with correct types
            for key in ("summary", "key_claims", "methodologies",
                        "limitations", "source_citations"):
                if key not in parsed:
                    parsed[key] = [] if key != "summary" else ""
                if key != "summary" and not isinstance(parsed[key], list):
                    parsed[key] = [str(parsed[key])]

            return parsed

        except Exception as e:
            print(f"[SummaryAgent] Structured summarization failed: {e}. "
                  "Falling back to plain-text summary.")
            return None

    # ───────────────────────────────────────
    # Plain-text fallback
    # ───────────────────────────────────────

    def _plain_summarize(self, query: str, context: str) -> str:
        """Fallback: generate a plain-text summary via LLM or extractive method."""
        system_prompt = (
            "You are an expert academic summarizer. Your task is to condense "
            "multiple retrieved passages into a coherent, well-organized summary "
            "that preserves key information and is relevant to the user's query."
        )

        # Truncate context to stay within token limit
        context_truncated = context[:5000] if len(context) > 5000 else context

        prompt = (
            f"User Query: {query}\n\n"
            f"Retrieved Passages:\n{context_truncated}\n\n"
            "Please provide a comprehensive summary of the above passages that "
            "addresses the user's query. Organize the information logically and "
            "note which sources support each point."
        )

        try:
            summary = self.llm_client.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=800,
                temperature=0.2,
            )
            return summary.strip()
        except Exception as e:
            print(f"[SummaryAgent] Plain summary also failed: {e}")
            # Last resort: extractive (first 3 sentences of context)
            import re
            sentences = re.split(r'(?<=[.!?])\s+', context.strip())
            return " ".join(sentences[:3])
