"""
Verification Agent Module
Verifies the final answer against retrieved context to flag hallucinations.
Acts as the 4th and final quality-gate in the multi-agent pipeline.
"""

import json
from typing import Dict, Any, Optional


class VerificationAgent:
    """
    Agent responsible for checking whether the generated answer is faithfully
    grounded in the retrieved context.

    Workflow:
        1. Receive the ExplanationAgent's answer + retrieved context.
        2. Prompt the LLM to identify claims NOT supported by context.
        3. If unsupported claims are found, produce a corrected answer.
        4. Return a verdict dict with verification status.

    Output schema:
        {
            "query": str,
            "verified": bool,
            "original_answer": str,
            "corrected_answer": str | None,
            "issues": [str, ...],
            "confidence": str,   # "high", "medium", "low"
        }
    """

    VERIFICATION_SYSTEM_PROMPT = (
        "You are a rigorous academic fact-checker. Your task is to verify "
        "whether an AI-generated answer is fully supported by the provided "
        "research context.\n\n"
        "You MUST respond with ONLY a valid JSON object (no markdown fences, "
        "no explanation outside the JSON) with these keys:\n"
        '  "verified": true if ALL claims in the answer are supported by the context, false otherwise\n'
        '  "issues": a JSON array of strings describing unsupported or hallucinated claims (empty if verified)\n'
        '  "corrected_answer": if verified is false, provide a corrected version of the answer '
        "that removes or fixes unsupported claims while preserving supported content. "
        "If verified is true, set this to null.\n\n"
        "Rules:\n"
        "- A claim is unsupported if the context does not contain evidence for it.\n"
        "- Do NOT penalize reasonable inferences that logically follow from the context.\n"
        "- Do NOT penalize stylistic phrasing or organizational differences.\n"
        "- Focus on factual accuracy and faithfulness.\n"
    )

    def __init__(self, llm_client):
        """
        Args:
            llm_client: An instance of reasoning.llm_client.LLMClient.
        """
        self.llm_client = llm_client

    def run(
        self,
        explanation_output: dict,
        retrieval_context: str,
        query: str = "",
    ) -> Dict[str, Any]:
        """
        Verify the explanation agent's answer against the retrieved context.

        Args:
            explanation_output: Output dict from ExplanationAgent.run().
            retrieval_context: The original retrieved context string.
            query: The original user query.

        Returns:
            Verification verdict dict.
        """
        answer = explanation_output.get("answer", "")
        original_confidence = explanation_output.get("confidence", "unknown")

        if not answer.strip() or not retrieval_context.strip():
            return {
                "query": query,
                "verified": True,
                "original_answer": answer,
                "corrected_answer": None,
                "issues": [],
                "confidence": original_confidence,
            }

        # ── LLM-based verification ──
        verdict = self._llm_verify(query, answer, retrieval_context)

        if verdict is not None:
            return {
                "query": query,
                "verified": verdict.get("verified", True),
                "original_answer": answer,
                "corrected_answer": verdict.get("corrected_answer"),
                "issues": verdict.get("issues", []),
                "confidence": self._adjust_confidence(
                    original_confidence, verdict.get("verified", True)
                ),
            }

        # ── Fallback: heuristic keyword-overlap check ──
        return self._heuristic_verify(query, answer, retrieval_context, original_confidence)

    # ───────────────────────────────────────────
    # LLM verification
    # ───────────────────────────────────────────

    def _llm_verify(
        self, query: str, answer: str, context: str
    ) -> Optional[Dict[str, Any]]:
        """
        Use the LLM to verify answer faithfulness.

        Returns:
            Parsed verdict dict, or None on failure.
        """
        # Truncate context to avoid token overflow
        context_truncated = context[:4000]

        prompt = (
            f"User Question: {query}\n\n"
            f"Retrieved Context:\n{context_truncated}\n\n"
            f"Generated Answer:\n{answer}\n\n"
            "Verify the answer now and produce the JSON verdict."
        )

        try:
            response = self.llm_client.generate(
                prompt=prompt,
                system_prompt=self.VERIFICATION_SYSTEM_PROMPT,
                max_tokens=800,
                temperature=0.0,
            )

            # Strip markdown fences if present
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[-1]
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]
                cleaned = cleaned.strip()

            parsed = json.loads(cleaned)

            # Normalise types
            parsed["verified"] = bool(parsed.get("verified", True))
            if not isinstance(parsed.get("issues"), list):
                parsed["issues"] = []
            if parsed["verified"]:
                parsed["corrected_answer"] = None

            return parsed

        except Exception as e:
            print(f"[VerificationAgent] LLM verification failed: {e}. "
                  "Falling back to heuristic.")
            return None

    # ───────────────────────────────────────────
    # Heuristic fallback
    # ───────────────────────────────────────────

    def _heuristic_verify(
        self,
        query: str,
        answer: str,
        context: str,
        original_confidence: str,
    ) -> Dict[str, Any]:
        """
        Simple heuristic verification: check what fraction of answer sentences
        have keyword overlap with the context.
        """
        import re

        context_lower = context.lower()
        sentences = re.split(r'(?<=[.!?])\s+', answer.strip())
        sentences = [s for s in sentences if len(s.strip()) > 15]

        if not sentences:
            return {
                "query": query,
                "verified": True,
                "original_answer": answer,
                "corrected_answer": None,
                "issues": [],
                "confidence": original_confidence,
            }

        grounded = 0
        issues = []

        for sent in sentences:
            words = re.findall(r'\b[a-z]{4,}\b', sent.lower())
            if not words:
                grounded += 1
                continue
            overlap = sum(1 for w in words if w in context_lower)
            ratio = overlap / len(words) if words else 0
            if ratio >= 0.3:
                grounded += 1
            else:
                issues.append(f"Possibly unsupported: \"{sent[:80]}...\"")

        grounded_ratio = grounded / len(sentences)
        verified = grounded_ratio >= 0.7  # 70% threshold

        return {
            "query": query,
            "verified": verified,
            "original_answer": answer,
            "corrected_answer": None,  # heuristic can't correct
            "issues": issues,
            "confidence": self._adjust_confidence(original_confidence, verified),
        }

    # ───────────────────────────────────────────
    # Helpers
    # ───────────────────────────────────────────

    @staticmethod
    def _adjust_confidence(original: str, verified: bool) -> str:
        """Downgrade confidence if verification failed."""
        if verified:
            return original
        # Downgrade by one level
        downgrade = {"high": "medium", "medium": "low", "low": "low"}
        return downgrade.get(original, "low")
