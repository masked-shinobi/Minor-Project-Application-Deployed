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
        '  "query_type": one of "factual", "comparative", "methodological", "summary", "definition", "chitchat", "general"\n'
        '  "search_mode": one of "hybrid", "semantic", "keyword"\n'
        '  "top_k": integer between 3 and 8 (how many chunks to retrieve)\n'
        '  "needs_summary": boolean (whether retrieved chunks need summarization)\n'
        '  "strategy_notes": a short one-line explanation of your reasoning\n\n'
        "Classification guidelines:\n"
        '- "chitchat": greeting, "who are you", "thanks", or non-research talk → top_k=0, needs_summary=false\n'
        '- "factual": precise academic questions → semantic, top_k=3\n'
        '- "comparative": compare/contrast/vs questions → hybrid, top_k=6\n'
        '- "methodological": how does/approach/algorithm questions → hybrid, top_k=5\n'
        '- "summary": summarize/overview/overall questions → hybrid, top_k=8, needs_summary=true\n'
        '- "definition": what is/define/meaning questions → semantic, top_k=3, needs_summary=false\n'
        '- "general": academic questions not fitting above → hybrid, top_k=5\n'
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
              - "is_ambiguous": (optional) boolean if user needs to select a paper
        """
        # ── Detect paper filter first ──
        paper_filter = self._detect_paper_filter(query, available_papers)

        # ── LLM-based classification (primary) ──
        if self.llm_client:
            plan = self._llm_plan(query)
            if plan is not None:
                plan["paper_filter"] = paper_filter
                return self._finalize_plan(plan, query, available_papers)

        # ── Heuristic fallback ──
        plan = self._heuristic_plan(query)
        plan["paper_filter"] = paper_filter
        return self._finalize_plan(plan, query, available_papers)

    def _finalize_plan(self, plan: Dict, query: str, available_papers: list) -> Dict:
        """Apply final logic for ambiguity and chitchat bypass."""
        if plan["query_type"] == "chitchat":
            plan["top_k"] = 0
            plan["needs_summary"] = False
            return plan

        # Smart Ambiguity Detection
        # If it's a summary/overview request and no paper is specified
        if plan["query_type"] == "summary" and not plan.get("paper_filter"):
            if available_papers and len(available_papers) == 1:
                # Only 1 paper — auto-bypass
                paper = available_papers[0]
                plan["paper_filter"] = paper if isinstance(paper, str) else paper.get("paper_id")
                plan["strategy_notes"] += " (Auto-selected single existing paper)"
            elif available_papers and len(available_papers) > 1:
                # Multiple papers — flag as ambiguous
                plan["query_type"] = "ambiguous"
                plan["is_ambiguous"] = True
                plan["strategy_notes"] = "Multiple papers found. User needs to select target."
        
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
                "summary", "definition", "chitchat", "general",
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

        if query_type == "chitchat":
            plan["top_k"] = 0
            plan["needs_summary"] = False
            plan["strategy_notes"] = "General conversation detected."

        elif query_type == "factual":
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
        """
        if any(w in query for w in ["hi", "hello", "hey", "who are you", "thanks", "thank you"]):
            return "chitchat"

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
        """Detect if the query mentions a specific paper using keyword matching."""
        if not available_papers:
            return None

        query_lower = query.lower().strip()
        for paper in available_papers:
            # Extract title and paper_id for matching
            title = paper.get("title", "") if isinstance(paper, dict) else ""
            paper_id = paper.get("paper_id", "") if isinstance(paper, dict) else (paper if isinstance(paper, str) else "")
            
            # Simple keyword matching (fuzzy-ish)
            if paper_id.lower() in query_lower:
                return paper_id
            
            if title and title.lower() in query_lower:
                return paper_id
                
            # Partial ID matching (e.g. "attention" in "attention-is-all-you-need")
            if len(paper_id) > 5 and paper_id.lower() in query_lower:
                return paper_id
                
        return None
