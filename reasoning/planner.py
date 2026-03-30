"""
Planner Module
Analyzes user queries and decides the retrieval and reasoning strategy.
Uses LLM-based classification with heuristic fallback.
"""

import json
from typing import Dict, Any, Optional, List


class Planner:
    """
    Analyzes the user query and creates an execution plan for the agents.
    Uses LLM-based intent classification for accurate query understanding,
    falling back to keyword heuristics when LLM is unavailable.

    Determines:
    - Query type (factual, comparative, methodological, summary, definition, general)
    - Search strategy (semantic, keyword, or hybrid)
    - Number of chunks needed
    - Whether multi-step reasoning is required
    - Special handling (e.g., comparison queries, table queries)
    """

    CLASSIFICATION_SYSTEM_PROMPT = (
        "You are a query classification engine for an academic research paper analyzer. "
        "Your job is to analyze the user's question and output a JSON execution plan.\n\n"
        "You MUST respond with ONLY a valid JSON object (no markdown, no explanation) with these keys:\n"
        '  "query_type": one of "factual", "comparative", "methodological", "summary", "definition", "general"\n'
        '  "search_mode": one of "hybrid", "semantic", "keyword"\n'
        '  "top_k": integer between 3 and 8 (how many chunks to retrieve)\n'
        '  "needs_summary": boolean (whether retrieved chunks need summarization)\n'
        '  "strategy_notes": a short one-line explanation of your reasoning\n\n'
        "Classification guidelines:\n"
        '- "factual": who/when/where/how many questions needing precise answers → semantic, top_k=3\n'
        '- "comparative": compare/contrast/difference/vs questions → hybrid, top_k=6\n'
        '- "methodological": how does/method/approach/algorithm questions → hybrid, top_k=5\n'
        '- "summary": summarize/overview/overall questions → hybrid, top_k=8, needs_summary=true\n'
        '- "definition": what is/define/meaning questions → semantic, top_k=3, needs_summary=false\n'
        '- "general": anything else → hybrid, top_k=5\n'
    )

    def __init__(self, llm_client=None):
        """
        Args:
            llm_client: Optional LLM client for advanced query classification.
        """
        self.llm_client = llm_client

    def plan(self, query: str, available_papers: list = None) -> Dict[str, Any]:
        """
        Create an execution plan for the given query.

        Args:
            query: The user's question.
            available_papers: List of available paper IDs/titles.

        Returns:
            Execution plan dict with keys:
              - "query_type": classification of the query
              - "search_mode": "hybrid", "semantic", or "keyword"
              - "top_k": number of chunks to retrieve
              - "paper_filter": optional paper_id filter
              - "needs_summary": whether summarization agent is needed
              - "strategy_notes": human-readable explanation
        """
        # ── LLM-based classification (primary) ──
        if self.llm_client:
            plan = self._llm_plan(query)
            if plan is not None:
                plan["paper_filter"] = self._detect_paper_filter(
                    query, available_papers
                )
                return plan

        # ── Heuristic fallback ──
        plan = self._heuristic_plan(query)
        plan["paper_filter"] = self._detect_paper_filter(query, available_papers)
        return plan

    # ─────────────────────────────────────────────
    # LLM-based classification
    # ─────────────────────────────────────────────

    def _llm_plan(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Use the LLM to classify the query and produce an execution plan.

        Returns:
            Execution plan dict, or None if the LLM call fails.
        """
        prompt = f'Classify the following academic research question:\n\n"{query}"'

        try:
            response = self.llm_client.generate(
                prompt=prompt,
                system_prompt=self.CLASSIFICATION_SYSTEM_PROMPT,
                max_tokens=200,
                temperature=0.0,
            )

            # Strip markdown fences if the LLM wraps the JSON
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[-1]
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]
                cleaned = cleaned.strip()

            parsed = json.loads(cleaned)

            # Validate and normalise
            valid_types = {
                "factual", "comparative", "methodological",
                "summary", "definition", "general",
            }
            valid_modes = {"hybrid", "semantic", "keyword"}

            query_type = parsed.get("query_type", "general")
            if query_type not in valid_types:
                query_type = "general"

            search_mode = parsed.get("search_mode", "hybrid")
            if search_mode not in valid_modes:
                search_mode = "hybrid"

            top_k = parsed.get("top_k", 5)
            if not isinstance(top_k, int) or top_k < 1:
                top_k = 5
            top_k = min(max(top_k, 3), 8)  # clamp to [3, 8]

            needs_summary = parsed.get("needs_summary", True)

            return {
                "query_type": query_type,
                "search_mode": search_mode,
                "top_k": top_k,
                "paper_filter": None,
                "needs_summary": bool(needs_summary),
                "strategy_notes": parsed.get(
                    "strategy_notes", f"LLM classified as {query_type}."
                ),
            }

        except Exception as e:
            print(f"[Planner] LLM classification failed: {e}. Falling back to heuristics.")
            return None

    # ─────────────────────────────────────────────
    # Heuristic fallback
    # ─────────────────────────────────────────────

    def _heuristic_plan(self, query: str) -> Dict[str, Any]:
        """Build an execution plan using keyword heuristics (no LLM needed)."""
        query_lower = query.lower().strip()
        query_type = self._heuristic_classify(query_lower)

        plan = {
            "query_type": query_type,
            "search_mode": "hybrid",
            "top_k": 5,
            "paper_filter": None,
            "needs_summary": True,
            "strategy_notes": "",
        }

        if query_type == "factual":
            plan["top_k"] = 3
            plan["search_mode"] = "semantic"
            plan["strategy_notes"] = "Factual query — targeted semantic search."

        elif query_type == "comparative":
            plan["top_k"] = 6
            plan["search_mode"] = "hybrid"
            plan["strategy_notes"] = "Comparative query — broader context needed."

        elif query_type == "methodological":
            plan["top_k"] = 5
            plan["search_mode"] = "hybrid"
            plan["strategy_notes"] = "Methodology query — hybrid search for detailed results."

        elif query_type == "summary":
            plan["top_k"] = 8
            plan["needs_summary"] = True
            plan["strategy_notes"] = "Summary request — retrieving broad context."

        elif query_type == "definition":
            plan["top_k"] = 3
            plan["search_mode"] = "semantic"
            plan["needs_summary"] = False
            plan["strategy_notes"] = "Definition query — narrow semantic search."

        else:
            plan["strategy_notes"] = "General query — standard hybrid search."

        return plan

    def _heuristic_classify(self, query: str) -> str:
        """
        Classify the query type using keyword heuristics.

        Returns one of: "factual", "comparative", "methodological",
                        "summary", "definition", "general"
        """
        if any(w in query for w in ["compare", "difference", "versus", "vs", "better"]):
            return "comparative"

        if any(w in query for w in ["how does", "method", "approach", "algorithm", "technique"]):
            return "methodological"

        if any(w in query for w in ["summarize", "overview", "overall", "summary of"]):
            return "summary"

        if any(w in query for w in ["what is", "define", "definition", "meaning of"]):
            return "definition"

        if any(w in query for w in ["who", "when", "where", "how many", "how much"]):
            return "factual"

        return "general"

    # ─────────────────────────────────────────────
    # Paper-filter detection (shared)
    # ─────────────────────────────────────────────

    def _detect_paper_filter(
        self, query: str, available_papers: Optional[List] = None
    ) -> Optional[str]:
        """Detect if the query mentions a specific paper and return its ID."""
        if not available_papers:
            return None

        query_lower = query.lower().strip()
        for paper in available_papers:
            paper_name = paper if isinstance(paper, str) else paper.get("title", "")
            if paper_name.lower() in query_lower:
                return paper if isinstance(paper, str) else paper.get("paper_id")
        return None
