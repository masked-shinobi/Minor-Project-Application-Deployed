# Resembler — 6 Architectural Upgrades Walkthrough

All 6 upgrades have been implemented and verified.

## Changes Made

### Phase 1 — LLM-Based Query Classification
```diff:planner.py
"""
Planner Module
Analyzes user queries and decides the retrieval and reasoning strategy.
"""

from typing import Dict, Any, Optional


class Planner:
    """
    Analyzes the user query and creates an execution plan for the agents.
    Determines:
    - Search strategy (semantic, keyword, or hybrid)
    - Number of chunks needed
    - Whether multi-step reasoning is required
    - Special handling (e.g., comparison queries, table queries)
    """

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
        query_lower = query.lower().strip()
        plan = {
            "query_type": "general",
            "search_mode": "hybrid",
            "top_k": 5,
            "paper_filter": None,
            "needs_summary": True,
            "strategy_notes": ""
        }

        # Classify query type
        plan["query_type"] = self._classify_query(query_lower)

        # Adjust strategy based on query type
        if plan["query_type"] == "factual":
            plan["top_k"] = 3
            plan["search_mode"] = "semantic"
            plan["strategy_notes"] = "Factual query — targeted semantic search."

        elif plan["query_type"] == "comparative":
            plan["top_k"] = 8
            plan["search_mode"] = "hybrid"
            plan["strategy_notes"] = "Comparative query — broader context needed."

        elif plan["query_type"] == "methodological":
            plan["top_k"] = 5
            plan["search_mode"] = "hybrid"
            plan["strategy_notes"] = "Methodology query — hybrid search for detailed results."

        elif plan["query_type"] == "summary":
            plan["top_k"] = 10
            plan["needs_summary"] = True
            plan["strategy_notes"] = "Summary request — retrieving broad context."

        elif plan["query_type"] == "definition":
            plan["top_k"] = 3
            plan["search_mode"] = "semantic"
            plan["needs_summary"] = False
            plan["strategy_notes"] = "Definition query — narrow semantic search."

        else:
            plan["strategy_notes"] = "General query — standard hybrid search."

        # Check if query mentions a specific paper
        if available_papers:
            for paper in available_papers:
                paper_name = paper if isinstance(paper, str) else paper.get("title", "")
                if paper_name.lower() in query_lower:
                    plan["paper_filter"] = paper if isinstance(paper, str) else paper.get("paper_id")
                    plan["strategy_notes"] += f" Filtered to paper: {paper_name}"
                    break

        return plan

    def _classify_query(self, query: str) -> str:
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
===
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
        '  "top_k": integer between 3 and 15 (how many chunks to retrieve)\n'
        '  "needs_summary": boolean (whether retrieved chunks need summarization)\n'
        '  "strategy_notes": a short one-line explanation of your reasoning\n\n'
        "Classification guidelines:\n"
        '- "factual": who/when/where/how many questions needing precise answers → semantic, top_k=3\n'
        '- "comparative": compare/contrast/difference/vs questions → hybrid, top_k=8\n'
        '- "methodological": how does/method/approach/algorithm questions → hybrid, top_k=5\n'
        '- "summary": summarize/overview/overall questions → hybrid, top_k=10, needs_summary=true\n'
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
            top_k = min(max(top_k, 3), 15)  # clamp to [3, 15]

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
            plan["top_k"] = 8
            plan["search_mode"] = "hybrid"
            plan["strategy_notes"] = "Comparative query — broader context needed."

        elif query_type == "methodological":
            plan["top_k"] = 5
            plan["search_mode"] = "hybrid"
            plan["strategy_notes"] = "Methodology query — hybrid search for detailed results."

        elif query_type == "summary":
            plan["top_k"] = 10
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

```

Replaced `if-elif` keyword heuristics with an LLM call that returns a structured JSON plan (`query_type`, `search_mode`, `top_k`, `needs_summary`). The old heuristic is preserved as a fallback.

---

### Phase 2 — Structured Summary Output
```diff:summary_agent.py
"""
Summary Agent Module
Summarizes retrieved context for the final explanation agent.
"""

from typing import Optional


class SummaryAgent:
    """
    Agent responsible for condensing retrieved passages into a coherent summary.
    Acts as the second step in the multi-agent pipeline.
    """

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
            Dict with keys:
              - "query": original query
              - "summary": condensed summary of retrieved context
              - "source_count": number of source chunks
              - "original_context": the original context (for reference)
        """
        context = retrieval_output.get("context", "")
        num_results = retrieval_output.get("num_results", 0)

        if not context.strip():
            return {
                "query": query,
                "summary": "No relevant context was found for this query.",
                "source_count": 0,
                "original_context": ""
            }

        system_prompt = (
            "You are an expert academic summarizer. Your task is to condense "
            "multiple retrieved passages into a coherent, well-organized summary "
            "that preserves key information and is relevant to the user's query."
        )

        prompt = (
            f"User Query: {query}\n\n"
            f"Retrieved Passages:\n{context}\n\n"
            "Please provide a comprehensive summary of the above passages that "
            "addresses the user's query. Organize the information logically and "
            "note which sources support each point."
        )

        try:
            summary = self.llm_client.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=800,
                temperature=0.2
            )
        except Exception as e:
            print(f"[SummaryAgent] Error: {e}")
            summary = f"Error generating summary: {e}"

        return {
            "query": query,
            "summary": summary,
            "source_count": num_results,
            "original_context": context
        }
===
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
        prompt = (
            f"User Query: {query}\n\n"
            f"Retrieved Passages:\n{context}\n\n"
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

        prompt = (
            f"User Query: {query}\n\n"
            f"Retrieved Passages:\n{context}\n\n"
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

```

The [SummaryAgent](file:///c:/GIT_MINOR/MinorProject_resembler/agents/summary_agent.py#11-198) now prompts the LLM to return JSON with `key_claims`, `methodologies`, `limitations`, and `source_citations`. Falls back to plain-text summary, then extractive fallback.

---

### Phase 3 — Cross-Encoder Reranking
```diff:retriever.py
"""
Retriever Module
Bridges FAISS vector search with SQLite metadata for comprehensive retrieval.
Implements hybrid search: semantic (FAISS) + keyword (SQLite).
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np


@dataclass
class RetrievalResult:
    """A single retrieval result with chunk content and metadata."""
    chunk_id: str
    content: str
    section_heading: str
    paper_id: str
    paper_title: str
    summary: str
    keywords: str
    similarity_score: float = 0.0
    keyword_score: float = 0.0
    combined_score: float = 0.0


class Retriever:
    """
    Hybrid retriever combining semantic search (FAISS) and keyword search (SQLite).
    """

    def __init__(self, faiss_store, metadata_db, embedder):
        """
        Initialize the retriever.

        Args:
            faiss_store: An instance of vectorstore.faiss_store.FAISSStore.
            metadata_db: An instance of database.metadata_db.MetadataDB.
            embedder: An instance of embeddings.embedder.Embedder.
        """
        self.faiss_store = faiss_store
        self.metadata_db = metadata_db
        self.embedder = embedder

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
        paper_id: Optional[str] = None
    ) -> List[RetrievalResult]:
        """
        Perform hybrid retrieval: semantic + keyword search.

        Args:
            query: User query text.
            top_k: Number of results to return.
            semantic_weight: Weight for semantic similarity scores (0-1).
            keyword_weight: Weight for keyword match scores (0-1).
            paper_id: Optional filter to restrict to a specific paper.

        Returns:
            List of RetrievalResult sorted by combined score.
        """
        # 1. Semantic search via FAISS
        query_embedding = self.embedder.embed_text(query)
        semantic_results = self.faiss_store.search(query_embedding, top_k=top_k * 2)

        # 2. Keyword search via SQLite
        query_keywords = self._extract_query_keywords(query)
        keyword_results = []
        if query_keywords:
            keyword_results = self.metadata_db.search_by_keywords(
                query_keywords, limit=top_k * 2
            )

        # 3. Merge and score
        results = self._merge_results(
            semantic_results=semantic_results,
            keyword_results=keyword_results,
            semantic_weight=semantic_weight,
            keyword_weight=keyword_weight,
            paper_id=paper_id
        )

        # 4. Sort by combined score and return top_k
        results.sort(key=lambda r: r.combined_score, reverse=True)
        return results[:top_k]

    def retrieve_semantic(
        self,
        query: str,
        top_k: int = 5
    ) -> List[RetrievalResult]:
        """
        Perform pure semantic (vector) search.

        Args:
            query: User query text.
            top_k: Number of results.

        Returns:
            List of RetrievalResult sorted by similarity score.
        """
        query_embedding = self.embedder.embed_text(query)
        search_results = self.faiss_store.search(query_embedding, top_k=top_k)

        results = []
        chunk_ids = [cid for cid, _ in search_results]
        chunk_data = self.metadata_db.get_chunks_by_ids(chunk_ids)
        chunk_map = {c["chunk_id"]: c for c in chunk_data}

        for chunk_id, score in search_results:
            meta = chunk_map.get(chunk_id, {})
            results.append(RetrievalResult(
                chunk_id=chunk_id,
                content=meta.get("content", ""),
                section_heading=meta.get("section_heading", ""),
                paper_id=meta.get("paper_id", ""),
                paper_title=self._get_paper_title(meta.get("paper_id", "")),
                summary=meta.get("summary", ""),
                keywords=meta.get("keywords", ""),
                similarity_score=score,
                combined_score=score
            ))

        return results

    def _merge_results(
        self,
        semantic_results: List[Tuple[str, float]],
        keyword_results: List[Dict[str, Any]],
        semantic_weight: float,
        keyword_weight: float,
        paper_id: Optional[str]
    ) -> List[RetrievalResult]:
        """Merge semantic and keyword results with weighted scoring."""

        # Collect all unique chunk IDs
        all_chunk_ids = set()
        semantic_scores = {}
        keyword_ids = set()

        for cid, score in semantic_results:
            all_chunk_ids.add(cid)
            semantic_scores[cid] = score

        for result in keyword_results:
            cid = result["chunk_id"]
            all_chunk_ids.add(cid)
            keyword_ids.add(cid)

        # Fetch metadata for all chunks
        chunk_data = self.metadata_db.get_chunks_by_ids(list(all_chunk_ids))
        chunk_map = {c["chunk_id"]: c for c in chunk_data}

        results = []
        for cid in all_chunk_ids:
            meta = chunk_map.get(cid, {})

            # Apply paper filter
            if paper_id and meta.get("paper_id") != paper_id:
                continue

            sem_score = semantic_scores.get(cid, 0.0)
            kw_score = 1.0 if cid in keyword_ids else 0.0
            combined = (semantic_weight * sem_score) + (keyword_weight * kw_score)

            results.append(RetrievalResult(
                chunk_id=cid,
                content=meta.get("content", ""),
                section_heading=meta.get("section_heading", ""),
                paper_id=meta.get("paper_id", ""),
                paper_title=self._get_paper_title(meta.get("paper_id", "")),
                summary=meta.get("summary", ""),
                keywords=meta.get("keywords", ""),
                similarity_score=sem_score,
                keyword_score=kw_score,
                combined_score=combined
            ))

        return results

    def _extract_query_keywords(self, query: str) -> List[str]:
        """Extract simple keywords from a query for SQLite search."""
        import re
        # Common English stopwords to ignore
        stopwords = {
            "what", "is", "the", "how", "does", "do", "are", "was", "were",
            "can", "could", "would", "should", "will", "may", "might",
            "a", "an", "in", "on", "at", "to", "for", "of", "with", "by",
            "from", "about", "this", "that", "these", "those", "and", "or"
        }
        words = re.findall(r'\b[a-zA-Z]{3,}\b', query.lower())
        return [w for w in words if w not in stopwords]

    def _get_paper_title(self, paper_id: str) -> str:
        """Look up paper title from the database."""
        if not paper_id:
            return ""
        papers = self.metadata_db.list_papers()
        for paper in papers:
            if paper["paper_id"] == paper_id:
                return paper.get("title", "")
        return ""
===
"""
Retriever Module
Bridges FAISS vector search with SQLite metadata for comprehensive retrieval.
Implements hybrid search: semantic (FAISS) + keyword (SQLite).
Includes cross-encoder reranking for precision improvement.
"""

import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np


@dataclass
class RetrievalResult:
    """A single retrieval result with chunk content and metadata."""
    chunk_id: str
    content: str
    section_heading: str
    paper_id: str
    paper_title: str
    summary: str
    keywords: str
    similarity_score: float = 0.0
    keyword_score: float = 0.0
    combined_score: float = 0.0
    rerank_score: float = 0.0  # cross-encoder score (0 if not reranked)


class CrossEncoderReranker:
    """
    Wraps a sentence-transformers CrossEncoder for query-document reranking.
    Loaded lazily on first use to avoid slowing down startup when not needed.
    """

    def __init__(self, model_name: Optional[str] = None):
        """
        Args:
            model_name: Cross-encoder model name.
                        Defaults to CROSS_ENCODER_MODEL env var or
                        'cross-encoder/ms-marco-MiniLM-L-6-v2'.
        """
        if model_name is None:
            model_name = os.getenv(
                "CROSS_ENCODER_MODEL",
                "cross-encoder/ms-marco-MiniLM-L-6-v2",
            )
        self.model_name = model_name
        self._model = None  # lazy-loaded

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            print(f"[CrossEncoderReranker] Loading model: {self.model_name}")
            self._model = CrossEncoder(self.model_name)
            print("[CrossEncoderReranker] Model loaded.")

    def rerank(
        self,
        query: str,
        results: List["RetrievalResult"],
        top_k: int = 5,
    ) -> List["RetrievalResult"]:
        """
        Rerank retrieval results using a cross-encoder.

        Args:
            query: The user query.
            results: List of RetrievalResult from the initial retrieval.
            top_k: Number of results to keep after reranking.

        Returns:
            Reranked list of RetrievalResult (top_k best).
        """
        if not results:
            return results

        self._load()

        # Build (query, document) pairs
        pairs = [(query, r.content) for r in results]

        # Score all pairs in one batch
        scores = self._model.predict(pairs)

        # Assign scores and sort
        for result, score in zip(results, scores):
            result.rerank_score = float(score)

        results.sort(key=lambda r: r.rerank_score, reverse=True)
        return results[:top_k]


class Retriever:
    """
    Hybrid retriever combining semantic search (FAISS) and keyword search (SQLite).
    Optionally applies cross-encoder reranking for improved precision.
    """

    def __init__(self, faiss_store, metadata_db, embedder, use_reranker: bool = True):
        """
        Initialize the retriever.

        Args:
            faiss_store: An instance of vectorstore.faiss_store.FAISSStore.
            metadata_db: An instance of database.metadata_db.MetadataDB.
            embedder: An instance of embeddings.embedder.Embedder.
            use_reranker: Whether to enable cross-encoder reranking.
        """
        self.faiss_store = faiss_store
        self.metadata_db = metadata_db
        self.embedder = embedder
        self.reranker = CrossEncoderReranker() if use_reranker else None

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
        paper_id: Optional[str] = None
    ) -> List[RetrievalResult]:
        """
        Perform hybrid retrieval: semantic + keyword search, then cross-encoder reranking.

        Args:
            query: User query text.
            top_k: Number of results to return.
            semantic_weight: Weight for semantic similarity scores (0-1).
            keyword_weight: Weight for keyword match scores (0-1).
            paper_id: Optional filter to restrict to a specific paper.

        Returns:
            List of RetrievalResult sorted by combined/reranked score.
        """
        # Retrieve a wider candidate set for reranking
        candidate_k = top_k * 3 if self.reranker else top_k * 2

        # 1. Semantic search via FAISS
        query_embedding = self.embedder.embed_text(query)
        semantic_results = self.faiss_store.search(query_embedding, top_k=candidate_k)

        # 2. Keyword search via SQLite
        query_keywords = self._extract_query_keywords(query)
        keyword_results = []
        if query_keywords:
            keyword_results = self.metadata_db.search_by_keywords(
                query_keywords, limit=candidate_k
            )

        # 3. Merge and score
        results = self._merge_results(
            semantic_results=semantic_results,
            keyword_results=keyword_results,
            semantic_weight=semantic_weight,
            keyword_weight=keyword_weight,
            paper_id=paper_id
        )

        # 4. Sort by combined score
        results.sort(key=lambda r: r.combined_score, reverse=True)

        # 5. Cross-encoder reranking (if enabled)
        if self.reranker and results:
            results = self.reranker.rerank(query, results, top_k=top_k)
        else:
            results = results[:top_k]

        return results

    def retrieve_semantic(
        self,
        query: str,
        top_k: int = 5
    ) -> List[RetrievalResult]:
        """
        Perform pure semantic (vector) search with optional cross-encoder reranking.

        Args:
            query: User query text.
            top_k: Number of results.

        Returns:
            List of RetrievalResult sorted by similarity/reranked score.
        """
        candidate_k = top_k * 3 if self.reranker else top_k

        query_embedding = self.embedder.embed_text(query)
        search_results = self.faiss_store.search(query_embedding, top_k=candidate_k)

        results = []
        chunk_ids = [cid for cid, _ in search_results]
        chunk_data = self.metadata_db.get_chunks_by_ids(chunk_ids)
        chunk_map = {c["chunk_id"]: c for c in chunk_data}

        for chunk_id, score in search_results:
            meta = chunk_map.get(chunk_id, {})
            results.append(RetrievalResult(
                chunk_id=chunk_id,
                content=meta.get("content", ""),
                section_heading=meta.get("section_heading", ""),
                paper_id=meta.get("paper_id", ""),
                paper_title=self._get_paper_title(meta.get("paper_id", "")),
                summary=meta.get("summary", ""),
                keywords=meta.get("keywords", ""),
                similarity_score=score,
                combined_score=score
            ))

        # Cross-encoder reranking
        if self.reranker and results:
            results = self.reranker.rerank(query, results, top_k=top_k)
        else:
            results = results[:top_k]

        return results

    def _merge_results(
        self,
        semantic_results: List[Tuple[str, float]],
        keyword_results: List[Dict[str, Any]],
        semantic_weight: float,
        keyword_weight: float,
        paper_id: Optional[str]
    ) -> List[RetrievalResult]:
        """Merge semantic and keyword results with weighted scoring."""

        # Collect all unique chunk IDs
        all_chunk_ids = set()
        semantic_scores = {}
        keyword_ids = set()

        for cid, score in semantic_results:
            all_chunk_ids.add(cid)
            semantic_scores[cid] = score

        for result in keyword_results:
            cid = result["chunk_id"]
            all_chunk_ids.add(cid)
            keyword_ids.add(cid)

        # Fetch metadata for all chunks
        chunk_data = self.metadata_db.get_chunks_by_ids(list(all_chunk_ids))
        chunk_map = {c["chunk_id"]: c for c in chunk_data}

        results = []
        for cid in all_chunk_ids:
            meta = chunk_map.get(cid, {})

            # Apply paper filter
            if paper_id and meta.get("paper_id") != paper_id:
                continue

            sem_score = semantic_scores.get(cid, 0.0)
            kw_score = 1.0 if cid in keyword_ids else 0.0
            combined = (semantic_weight * sem_score) + (keyword_weight * kw_score)

            results.append(RetrievalResult(
                chunk_id=cid,
                content=meta.get("content", ""),
                section_heading=meta.get("section_heading", ""),
                paper_id=meta.get("paper_id", ""),
                paper_title=self._get_paper_title(meta.get("paper_id", "")),
                summary=meta.get("summary", ""),
                keywords=meta.get("keywords", ""),
                similarity_score=sem_score,
                keyword_score=kw_score,
                combined_score=combined
            ))

        return results

    def _extract_query_keywords(self, query: str) -> List[str]:
        """Extract simple keywords from a query for SQLite search."""
        import re
        # Common English stopwords to ignore
        stopwords = {
            "what", "is", "the", "how", "does", "do", "are", "was", "were",
            "can", "could", "would", "should", "will", "may", "might",
            "a", "an", "in", "on", "at", "to", "for", "of", "with", "by",
            "from", "about", "this", "that", "these", "those", "and", "or"
        }
        words = re.findall(r'\b[a-zA-Z]{3,}\b', query.lower())
        return [w for w in words if w not in stopwords]

    def _get_paper_title(self, paper_id: str) -> str:
        """Look up paper title from the database."""
        if not paper_id:
            return ""
        papers = self.metadata_db.list_papers()
        for paper in papers:
            if paper["paper_id"] == paper_id:
                return paper.get("title", "")
        return ""

```

Added [CrossEncoderReranker](file:///c:/GIT_MINOR/MinorProject_resembler/retrieval/retriever.py#31-93) (lazy-loaded `ms-marco-MiniLM-L-6-v2`). The retriever now fetches `top_k * 3` candidates, then reranks to `top_k` using the cross-encoder. Disabled via `use_reranker=False`.

---

### Phase 4 — Answer Verification Agent

**New file:** [verification_agent.py](file:///c:/GIT_MINOR/MinorProject_resembler/agents/verification_agent.py)

```diff:router.py
"""
Router Module
Orchestrates the multi-agent pipeline: Retrieval → Summary → Explanation.
Controls the flow based on the Planner's execution plan.
"""

from typing import Dict, Any, Optional
import time


class Router:
    """
    Conditional router that orchestrates the sequential agent pipeline.
    Routes the query through: Planner → Retrieval Agent → Summary Agent → Explanation Agent
    """

    def __init__(
        self,
        planner,
        retrieval_agent,
        summary_agent,
        explanation_agent
    ):
        """
        Args:
            planner: An instance of reasoning.planner.Planner.
            retrieval_agent: An instance of agents.retrieval_agent.RetrievalAgent.
            summary_agent: An instance of agents.summary_agent.SummaryAgent.
            explanation_agent: An instance of agents.explanation_agent.ExplanationAgent.
        """
        self.planner = planner
        self.retrieval_agent = retrieval_agent
        self.summary_agent = summary_agent
        self.explanation_agent = explanation_agent

    def route(self, query: str, available_papers: list = None) -> Dict[str, Any]:
        """
        Route a query through the full multi-agent pipeline.

        Args:
            query: The user's question.
            available_papers: Optional list of available papers.

        Returns:
            Dict containing the full pipeline output:
              - "query": original query
              - "plan": the execution plan
              - "retrieval": retrieval agent output
              - "summary": summary agent output (if applicable)
              - "explanation": explanation agent output
              - "answer": the final answer text
              - "timing": execution time for each step (in seconds)
        """
        timing = {}
        output = {"query": query}

        # Step 1: Planning
        t0 = time.time()
        plan = self.planner.plan(query, available_papers)
        timing["planning"] = round(time.time() - t0, 3)
        output["plan"] = plan

        print(f"[Router] Query type: {plan['query_type']}")
        print(f"[Router] Strategy: {plan['strategy_notes']}")

        # Step 2: Retrieval
        t0 = time.time()
        retrieval_output = self.retrieval_agent.run(
            query=query,
            top_k=plan["top_k"],
            paper_id=plan.get("paper_filter"),
            search_mode=plan["search_mode"]
        )
        timing["retrieval"] = round(time.time() - t0, 3)
        output["retrieval"] = retrieval_output

        print(f"[Router] Retrieved {retrieval_output['num_results']} chunks")

        # Step 3: Summarization (conditional)
        if plan["needs_summary"] and retrieval_output["num_results"] > 0:
            t0 = time.time()
            summary_output = self.summary_agent.run(retrieval_output, query=query)
            timing["summarization"] = round(time.time() - t0, 3)
            output["summary"] = summary_output
        else:
            # Skip summarization — pass retrieval context directly
            output["summary"] = {
                "query": query,
                "summary": retrieval_output.get("context", ""),
                "source_count": retrieval_output.get("num_results", 0),
                "original_context": retrieval_output.get("context", "")
            }
            timing["summarization"] = 0

        # Step 4: Explanation (final answer)
        t0 = time.time()
        explanation_output = self.explanation_agent.run(
            output["summary"], query=query
        )
        timing["explanation"] = round(time.time() - t0, 3)
        output["explanation"] = explanation_output

        # Final answer
        output["answer"] = explanation_output["answer"]
        output["confidence"] = explanation_output.get("confidence", "unknown")
        output["timing"] = timing

        total_time = sum(timing.values())
        print(f"[Router] Total time: {total_time:.2f}s | "
              f"Confidence: {output['confidence']}")

        return output
===
"""
Router Module
Orchestrates the multi-agent pipeline: Retrieval → Summary → Explanation → Verification.
Controls the flow based on the Planner's execution plan.
"""

from typing import Dict, Any, Optional
import time


class Router:
    """
    Conditional router that orchestrates the sequential agent pipeline.
    Routes the query through:
        Planner → Retrieval Agent → Summary Agent → Explanation Agent → Verification Agent
    """

    def __init__(
        self,
        planner,
        retrieval_agent,
        summary_agent,
        explanation_agent,
        verification_agent=None,
    ):
        """
        Args:
            planner: An instance of reasoning.planner.Planner.
            retrieval_agent: An instance of agents.retrieval_agent.RetrievalAgent.
            summary_agent: An instance of agents.summary_agent.SummaryAgent.
            explanation_agent: An instance of agents.explanation_agent.ExplanationAgent.
            verification_agent: An instance of agents.verification_agent.VerificationAgent.
                               If None, the verification step is skipped.
        """
        self.planner = planner
        self.retrieval_agent = retrieval_agent
        self.summary_agent = summary_agent
        self.explanation_agent = explanation_agent
        self.verification_agent = verification_agent

    def route(self, query: str, available_papers: list = None) -> Dict[str, Any]:
        """
        Route a query through the full multi-agent pipeline.

        Args:
            query: The user's question.
            available_papers: Optional list of available papers.

        Returns:
            Dict containing the full pipeline output:
              - "query": original query
              - "plan": the execution plan
              - "retrieval": retrieval agent output
              - "summary": summary agent output (if applicable)
              - "explanation": explanation agent output
              - "verification": verification agent output (if applicable)
              - "answer": the final answer text
              - "timing": execution time for each step (in seconds)
        """
        timing = {}
        output = {"query": query}

        # Step 1: Planning
        t0 = time.time()
        plan = self.planner.plan(query, available_papers)
        timing["planning"] = round(time.time() - t0, 3)
        output["plan"] = plan

        print(f"[Router] Query type: {plan['query_type']}")
        print(f"[Router] Strategy: {plan['strategy_notes']}")

        # Step 2: Retrieval
        t0 = time.time()
        retrieval_output = self.retrieval_agent.run(
            query=query,
            top_k=plan["top_k"],
            paper_id=plan.get("paper_filter"),
            search_mode=plan["search_mode"]
        )
        timing["retrieval"] = round(time.time() - t0, 3)
        output["retrieval"] = retrieval_output

        print(f"[Router] Retrieved {retrieval_output['num_results']} chunks")

        # Step 3: Summarization (conditional)
        if plan["needs_summary"] and retrieval_output["num_results"] > 0:
            t0 = time.time()
            summary_output = self.summary_agent.run(retrieval_output, query=query)
            timing["summarization"] = round(time.time() - t0, 3)
            output["summary"] = summary_output
        else:
            # Skip summarization — pass retrieval context directly
            output["summary"] = {
                "query": query,
                "summary": retrieval_output.get("context", ""),
                "structured": {
                    "key_claims": [],
                    "methodologies": [],
                    "limitations": [],
                    "source_citations": [],
                },
                "source_count": retrieval_output.get("num_results", 0),
                "original_context": retrieval_output.get("context", "")
            }
            timing["summarization"] = 0

        # Step 4: Explanation (final answer)
        t0 = time.time()
        explanation_output = self.explanation_agent.run(
            output["summary"], query=query
        )
        timing["explanation"] = round(time.time() - t0, 3)
        output["explanation"] = explanation_output

        # Step 5: Verification (if agent is available)
        if self.verification_agent:
            t0 = time.time()
            verification_output = self.verification_agent.run(
                explanation_output=explanation_output,
                retrieval_context=retrieval_output.get("context", ""),
                query=query,
            )
            timing["verification"] = round(time.time() - t0, 3)
            output["verification"] = verification_output

            # Use corrected answer if verification failed
            if not verification_output["verified"] and verification_output.get("corrected_answer"):
                output["answer"] = verification_output["corrected_answer"]
                output["confidence"] = verification_output.get("confidence", "low")
                print(f"[Router] ⚠ Verification failed — using corrected answer. "
                      f"Issues: {verification_output['issues']}")
            else:
                output["answer"] = explanation_output["answer"]
                output["confidence"] = verification_output.get(
                    "confidence", explanation_output.get("confidence", "unknown")
                )
                if verification_output["verified"]:
                    print("[Router] ✓ Answer verified as faithful.")
                else:
                    print("[Router] ⚠ Verification failed but no correction available.")
        else:
            # No verification agent — use explanation directly
            output["answer"] = explanation_output["answer"]
            output["confidence"] = explanation_output.get("confidence", "unknown")

        output["timing"] = timing

        total_time = sum(timing.values())
        print(f"[Router] Total time: {total_time:.2f}s | "
              f"Confidence: {output['confidence']}")

        return output

```
```diff:main.py
"""
RAG-Based Academic Research Paper Analyzer
==========================================
Main CLI interface for ingesting papers and querying the knowledge base.

Usage:
    python main.py ingest             # Ingest all PDFs from data/papers
    python main.py query "question"   # Query the knowledge base
    python main.py evaluate           # Run evaluation metrics
    python main.py security           # Run security tests
    python main.py stats              # Show system statistics
"""

import os
import sys
import argparse
import time

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


def build_ingestion_pipeline():
    """Build and return the full ingestion pipeline components."""
    from ingestion.pdf_loader import PDFLoader
    from ingestion.document_parser import DocumentParser
    from processing.structure_analyzer import StructureAnalyzer
    from processing.boundary_detector import BoundaryDetector
    from processing.keyword_extractor import KeywordExtractor
    from processing.summary_generator import SummaryGenerator
    from processing.question_generator import QuestionGenerator
    from processing.table_parser import TableParser

    return {
        "loader": PDFLoader(),
        "parser": DocumentParser(),
        "structure_analyzer": StructureAnalyzer(),
        "boundary_detector": BoundaryDetector(max_chunk_size=512, overlap_size=50),
        "keyword_extractor": KeywordExtractor(max_keywords=10),
        "summary_generator": SummaryGenerator(),  # No LLM for offline mode
        "question_generator": QuestionGenerator(),
        "table_parser": TableParser(),
    }


def build_query_pipeline():
    """Build and return the full query pipeline components."""
    from embeddings.embedder import Embedder
    from vectorstore.faiss_store import FAISSStore
    from database.metadata_db import MetadataDB
    from retrieval.retriever import Retriever
    from reasoning.llm_client import LLMClient
    from reasoning.planner import Planner
    from agents.retrieval_agent import RetrievalAgent
    from agents.summary_agent import SummaryAgent
    from agents.explanation_agent import ExplanationAgent
    from reasoning.router import Router

    # Initialize components
    embedder = Embedder()
    faiss_store = FAISSStore(embedding_dim=embedder.embedding_dim)
    metadata_db = MetadataDB()

    # Load existing index if available
    index_dir = os.path.join(PROJECT_ROOT, "data")
    try:
        faiss_store.load(index_dir)
    except FileNotFoundError:
        print("[Main] No existing FAISS index found. Run 'ingest' first.")

    retriever = Retriever(faiss_store, metadata_db, embedder)

    # LLM-powered components
    llm_client = LLMClient()
    planner = Planner(llm_client)
    retrieval_agent = RetrievalAgent(retriever)
    summary_agent = SummaryAgent(llm_client)
    explanation_agent = ExplanationAgent(llm_client)
    router = Router(planner, retrieval_agent, summary_agent, explanation_agent)

    return {
        "embedder": embedder,
        "faiss_store": faiss_store,
        "metadata_db": metadata_db,
        "retriever": retriever,
        "llm_client": llm_client,
        "router": router,
    }


def cmd_ingest(args):
    """Ingest PDF papers into the knowledge base."""
    print("=" * 60)
    print("  📄 Ingesting Research Papers")
    print("=" * 60)

    pipeline = build_ingestion_pipeline()
    from embeddings.embedder import Embedder
    from vectorstore.faiss_store import FAISSStore
    from database.metadata_db import MetadataDB

    # Discover PDFs
    pdf_paths = pipeline["loader"].discover_pdfs()
    if not pdf_paths:
        print("No PDF files found in data/papers/. Add PDFs and try again.")
        return

    print(f"Found {len(pdf_paths)} PDF(s):")
    for p in pdf_paths:
        print(f"  • {os.path.basename(p)}")

    # Initialize embedding and storage
    embedder = Embedder()
    faiss_store = FAISSStore(embedding_dim=embedder.embedding_dim)
    metadata_db = MetadataDB()

    total_chunks = 0
    t_start = time.time()

    for pdf_path in pdf_paths:
        paper_name = pipeline["loader"].get_paper_name(pdf_path)
        print(f"\n--- Processing: {paper_name} ---")

        # Parse PDF
        doc = pipeline["parser"].parse(pdf_path, paper_id=paper_name)
        print(f"  Pages: {doc.total_pages} | Characters: {len(doc.full_text)}")

        # Analyze structure
        sections = pipeline["structure_analyzer"].analyze(doc.full_text)
        print(f"  Sections detected: {len(sections)}")
        for sec in sections:
            print(f"    • {sec.heading} ({len(sec.content)} chars)")

        # Chunk sections
        chunks = pipeline["boundary_detector"].chunk_document(sections, paper_id=paper_name)
        print(f"  Chunks created: {len(chunks)}")

        # Extract keywords for each chunk
        chunk_keywords = pipeline["keyword_extractor"].extract_from_chunks(chunks)

        # Generate summaries
        summaries = pipeline["summary_generator"].summarize_chunks(chunks)

        # Generate embeddings
        embeddings = embedder.embed_chunks(chunks)
        chunk_ids = [c.chunk_id for c in chunks]

        # Add to FAISS
        faiss_store.add_embeddings(embeddings, chunk_ids)

        # Store metadata in SQLite
        metadata_db.add_paper(
            paper_id=paper_name,
            title=doc.title,
            file_path=pdf_path,
            total_pages=doc.total_pages,
            metadata=doc.metadata
        )

        from database.metadata_db import ChunkMetadata
        chunk_meta_list = []
        for i, chunk in enumerate(chunks):
            keywords = chunk_keywords.get(chunk.chunk_id, [])
            chunk_meta_list.append(ChunkMetadata(
                chunk_id=chunk.chunk_id,
                paper_id=paper_name,
                paper_title=doc.title,
                section_heading=chunk.section_heading,
                content=chunk.content,
                summary=summaries[i] if i < len(summaries) else "",
                keywords=",".join(keywords),
                page_numbers="",
                char_start=chunk.char_start,
                char_end=chunk.char_end
            ))
        metadata_db.add_chunks_batch(chunk_meta_list)

        total_chunks += len(chunks)

    # Save FAISS index
    index_dir = os.path.join(PROJECT_ROOT, "data")
    faiss_store.save(index_dir)

    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"  ✅ Ingestion Complete!")
    print(f"  Papers: {len(pdf_paths)} | Chunks: {total_chunks}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"{'=' * 60}")


def cmd_query(args):
    """Query the knowledge base."""
    query = args.query
    if not query:
        print("Please provide a query. Usage: python main.py query 'your question'")
        return

    print(f"\n🔍 Query: {query}\n")

    pipeline = build_query_pipeline()
    result = pipeline["router"].route(query)

    print(f"\n{'=' * 60}")
    print(f"  📝 Answer (Confidence: {result['confidence']})")
    print(f"{'=' * 60}")
    print(result["answer"])

    timing = result.get("timing", {})
    if timing:
        print(f"\n⏱  Timing: {sum(timing.values()):.2f}s total")
        for step, t in timing.items():
            print(f"   • {step}: {t:.3f}s")


def cmd_interactive(args):
    """Interactive query mode."""
    print("=" * 60)
    print("  🤖 RAG Research Paper Analyzer — Interactive Mode")
    print("  Type 'quit' or 'exit' to stop.")
    print("=" * 60)

    pipeline = build_query_pipeline()

    while True:
        try:
            query = input("\n❓ Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query or query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        result = pipeline["router"].route(query)

        print(f"\n📝 Answer (Confidence: {result['confidence']}):")
        print("-" * 40)
        print(result["answer"])

        timing = result.get("timing", {})
        if timing:
            print(f"\n⏱  {sum(timing.values()):.2f}s")


def cmd_stats(args):
    """Show system statistics."""
    from database.metadata_db import MetadataDB
    db = MetadataDB()
    stats = db.get_stats()

    print(f"\n📊 System Statistics:")
    print(f"  Papers indexed: {stats['total_papers']}")
    print(f"  Total chunks: {stats['total_chunks']}")

    papers = db.list_papers()
    if papers:
        print(f"\n  Papers:")
        for paper in papers:
            print(f"    • {paper['title']} ({paper['total_pages']} pages)")


def cmd_evaluate(args):
    """Run evaluation metrics (13 metrics across 3 categories)."""
    from evaluation.evaluation_runner import EvaluationRunner

    pipeline = build_query_pipeline()
    runner = EvaluationRunner(
        router=pipeline["router"],
        embedder=pipeline.get("embedder"),
        llm_client=pipeline.get("llm_client")
    )

    if args.all:
        # Run full test dataset evaluation
        results = runner.run_all()
    elif args.query:
        # Run single query evaluation
        result = runner.run_single(
            query=args.query,
            ground_truth=""  # No ground truth for ad-hoc queries
        )

        # Print detailed single-query results
        print(f"\n{'=' * 70}")
        print(f"  [*] Evaluation for: '{args.query}'")
        print(f"{'=' * 70}")

        metrics = result["metrics"]
        for category in ["retrieval", "generation"]:
            if metrics.get(category):
                print(f"\n  -- {category.title()} --")
                for name, data in metrics[category].items():
                    if isinstance(data, dict) and "score" in data:
                        print(f"    {name:20s}  {data['score']:.3f}  {data.get('explanation', '')}")

        print(f"\n  Overall: {metrics.get('overall_score', 0):.3f}")

        results = {"per_query": [result], "aggregate": {}, "dataset_size": 1}
    else:
        # Default: run full dataset
        results = runner.run_all()

    # Save results if requested
    if args.save:
        runner.save_results(results)


def main():
    parser = argparse.ArgumentParser(
        description="RAG-Based Academic Research Paper Analyzer"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Ingest command
    subparsers.add_parser("ingest", help="Ingest PDFs from data/papers/")

    # Query command
    query_parser = subparsers.add_parser("query", help="Query the knowledge base")
    query_parser.add_argument("query", type=str, help="Your question")

    # Interactive command
    subparsers.add_parser("interactive", help="Interactive query mode")

    # Stats command
    subparsers.add_parser("stats", help="Show system statistics")

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Run evaluation metrics")
    eval_parser.add_argument("--query", type=str, default=None,
                             help="Single query to evaluate")
    eval_parser.add_argument("--all", action="store_true",
                             help="Run full test dataset evaluation")
    eval_parser.add_argument("--save", action="store_true",
                             help="Save results to data/evaluation_results.json")

    # Security command
    subparsers.add_parser("security", help="Run security tests")

    args = parser.parse_args()

    if args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "query":
        cmd_query(args)
    elif args.command == "interactive":
        cmd_interactive(args)
    elif args.command == "stats":
        cmd_stats(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "security":
        print("Running security tests...")
        pipeline = build_query_pipeline()
        from security.prompt_injection_test import PromptInjectionTest
        tester = PromptInjectionTest(router=pipeline["router"])
        results = tester.run_all_tests()
        for category, data in results.items():
            if isinstance(data, dict) and "pass_rate" in data:
                print(f"  {category}: {data['pass_rate']:.0%} pass rate")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
===
"""
RAG-Based Academic Research Paper Analyzer
==========================================
Main CLI interface for ingesting papers and querying the knowledge base.

Usage:
    python main.py ingest             # Ingest all PDFs from data/papers
    python main.py query "question"   # Query the knowledge base
    python main.py evaluate           # Run evaluation metrics
    python main.py security           # Run security tests
    python main.py stats              # Show system statistics
"""

import os
import sys
import argparse
import time

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


def build_ingestion_pipeline():
    """Build and return the full ingestion pipeline components."""
    from ingestion.pdf_loader import PDFLoader
    from ingestion.document_parser import DocumentParser
    from processing.structure_analyzer import StructureAnalyzer
    from processing.boundary_detector import BoundaryDetector
    from processing.keyword_extractor import KeywordExtractor
    from processing.summary_generator import SummaryGenerator
    from processing.question_generator import QuestionGenerator
    from processing.table_parser import TableParser

    return {
        "loader": PDFLoader(),
        "parser": DocumentParser(),
        "structure_analyzer": StructureAnalyzer(),
        "boundary_detector": BoundaryDetector(max_chunk_size=512, overlap_size=50),
        "keyword_extractor": KeywordExtractor(max_keywords=10),
        "summary_generator": SummaryGenerator(),  # No LLM for offline mode
        "question_generator": QuestionGenerator(),
        "table_parser": TableParser(),
    }


def build_query_pipeline():
    """Build and return the full query pipeline components."""
    from embeddings.embedder import Embedder
    from vectorstore.faiss_store import FAISSStore
    from database.metadata_db import MetadataDB
    from retrieval.retriever import Retriever
    from reasoning.llm_client import LLMClient
    from reasoning.planner import Planner
    from agents.retrieval_agent import RetrievalAgent
    from agents.summary_agent import SummaryAgent
    from agents.explanation_agent import ExplanationAgent
    from agents.verification_agent import VerificationAgent
    from reasoning.router import Router

    # Initialize components
    embedder = Embedder()
    faiss_store = FAISSStore(embedding_dim=embedder.embedding_dim)
    metadata_db = MetadataDB()

    # Load existing index if available
    index_dir = os.path.join(PROJECT_ROOT, "data")
    try:
        faiss_store.load(index_dir)
    except FileNotFoundError:
        print("[Main] No existing FAISS index found. Run 'ingest' first.")

    retriever = Retriever(faiss_store, metadata_db, embedder)

    # LLM-powered components
    llm_client = LLMClient()
    planner = Planner(llm_client)
    retrieval_agent = RetrievalAgent(retriever)
    summary_agent = SummaryAgent(llm_client)
    explanation_agent = ExplanationAgent(llm_client)
    verification_agent = VerificationAgent(llm_client)
    router = Router(planner, retrieval_agent, summary_agent, explanation_agent, verification_agent)

    return {
        "embedder": embedder,
        "faiss_store": faiss_store,
        "metadata_db": metadata_db,
        "retriever": retriever,
        "llm_client": llm_client,
        "router": router,
    }


def cmd_ingest(args):
    """Ingest PDF papers into the knowledge base."""
    print("=" * 60)
    print("  📄 Ingesting Research Papers")
    print("=" * 60)

    pipeline = build_ingestion_pipeline()
    from embeddings.embedder import Embedder
    from vectorstore.faiss_store import FAISSStore
    from database.metadata_db import MetadataDB

    # Discover PDFs
    pdf_paths = pipeline["loader"].discover_pdfs()
    if not pdf_paths:
        print("No PDF files found in data/papers/. Add PDFs and try again.")
        return

    print(f"Found {len(pdf_paths)} PDF(s):")
    for p in pdf_paths:
        print(f"  • {os.path.basename(p)}")

    # Initialize embedding and storage
    embedder = Embedder()
    faiss_store = FAISSStore(embedding_dim=embedder.embedding_dim)
    metadata_db = MetadataDB()

    total_chunks = 0
    t_start = time.time()

    for pdf_path in pdf_paths:
        paper_name = pipeline["loader"].get_paper_name(pdf_path)
        print(f"\n--- Processing: {paper_name} ---")

        # Parse PDF
        doc = pipeline["parser"].parse(pdf_path, paper_id=paper_name)
        print(f"  Pages: {doc.total_pages} | Characters: {len(doc.full_text)}")

        # Analyze structure
        sections = pipeline["structure_analyzer"].analyze(doc.full_text)
        print(f"  Sections detected: {len(sections)}")
        for sec in sections:
            print(f"    • {sec.heading} ({len(sec.content)} chars)")

        # Chunk sections
        chunks = pipeline["boundary_detector"].chunk_document(sections, paper_id=paper_name)
        print(f"  Chunks created: {len(chunks)}")

        # Extract keywords for each chunk
        chunk_keywords = pipeline["keyword_extractor"].extract_from_chunks(chunks)

        # Generate summaries
        summaries = pipeline["summary_generator"].summarize_chunks(chunks)

        # Generate embeddings
        embeddings = embedder.embed_chunks(chunks)
        chunk_ids = [c.chunk_id for c in chunks]

        # Add to FAISS
        faiss_store.add_embeddings(embeddings, chunk_ids)

        # Store metadata in SQLite
        metadata_db.add_paper(
            paper_id=paper_name,
            title=doc.title,
            file_path=pdf_path,
            total_pages=doc.total_pages,
            metadata=doc.metadata
        )

        from database.metadata_db import ChunkMetadata
        chunk_meta_list = []
        for i, chunk in enumerate(chunks):
            keywords = chunk_keywords.get(chunk.chunk_id, [])
            chunk_meta_list.append(ChunkMetadata(
                chunk_id=chunk.chunk_id,
                paper_id=paper_name,
                paper_title=doc.title,
                section_heading=chunk.section_heading,
                content=chunk.content,
                summary=summaries[i] if i < len(summaries) else "",
                keywords=",".join(keywords),
                page_numbers="",
                char_start=chunk.char_start,
                char_end=chunk.char_end
            ))
        metadata_db.add_chunks_batch(chunk_meta_list)

        total_chunks += len(chunks)

    # Save FAISS index
    index_dir = os.path.join(PROJECT_ROOT, "data")
    faiss_store.save(index_dir)

    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"  ✅ Ingestion Complete!")
    print(f"  Papers: {len(pdf_paths)} | Chunks: {total_chunks}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"{'=' * 60}")


def cmd_query(args):
    """Query the knowledge base."""
    query = args.query
    if not query:
        print("Please provide a query. Usage: python main.py query 'your question'")
        return

    print(f"\n🔍 Query: {query}\n")

    pipeline = build_query_pipeline()
    result = pipeline["router"].route(query)

    print(f"\n{'=' * 60}")
    print(f"  📝 Answer (Confidence: {result['confidence']})")
    print(f"{'=' * 60}")
    print(result["answer"])

    timing = result.get("timing", {})
    if timing:
        print(f"\n⏱  Timing: {sum(timing.values()):.2f}s total")
        for step, t in timing.items():
            print(f"   • {step}: {t:.3f}s")


def cmd_interactive(args):
    """Interactive query mode."""
    print("=" * 60)
    print("  🤖 RAG Research Paper Analyzer — Interactive Mode")
    print("  Type 'quit' or 'exit' to stop.")
    print("=" * 60)

    pipeline = build_query_pipeline()

    while True:
        try:
            query = input("\n❓ Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query or query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        result = pipeline["router"].route(query)

        print(f"\n📝 Answer (Confidence: {result['confidence']}):")
        print("-" * 40)
        print(result["answer"])

        timing = result.get("timing", {})
        if timing:
            print(f"\n⏱  {sum(timing.values()):.2f}s")


def cmd_stats(args):
    """Show system statistics."""
    from database.metadata_db import MetadataDB
    db = MetadataDB()
    stats = db.get_stats()

    print(f"\n📊 System Statistics:")
    print(f"  Papers indexed: {stats['total_papers']}")
    print(f"  Total chunks: {stats['total_chunks']}")

    papers = db.list_papers()
    if papers:
        print(f"\n  Papers:")
        for paper in papers:
            print(f"    • {paper['title']} ({paper['total_pages']} pages)")


def cmd_evaluate(args):
    """Run evaluation metrics (13 metrics across 3 categories)."""
    from evaluation.evaluation_runner import EvaluationRunner

    pipeline = build_query_pipeline()
    runner = EvaluationRunner(
        router=pipeline["router"],
        embedder=pipeline.get("embedder"),
        llm_client=pipeline.get("llm_client")
    )

    if args.all:
        # Run full test dataset evaluation
        results = runner.run_all()
    elif args.query:
        # Run single query evaluation
        result = runner.run_single(
            query=args.query,
            ground_truth=""  # No ground truth for ad-hoc queries
        )

        # Print detailed single-query results
        print(f"\n{'=' * 70}")
        print(f"  [*] Evaluation for: '{args.query}'")
        print(f"{'=' * 70}")

        metrics = result["metrics"]
        for category in ["retrieval", "generation"]:
            if metrics.get(category):
                print(f"\n  -- {category.title()} --")
                for name, data in metrics[category].items():
                    if isinstance(data, dict) and "score" in data:
                        print(f"    {name:20s}  {data['score']:.3f}  {data.get('explanation', '')}")

        print(f"\n  Overall: {metrics.get('overall_score', 0):.3f}")

        results = {"per_query": [result], "aggregate": {}, "dataset_size": 1}
    else:
        # Default: run full dataset
        results = runner.run_all()

    # Save results if requested
    if args.save:
        runner.save_results(results)


def main():
    parser = argparse.ArgumentParser(
        description="RAG-Based Academic Research Paper Analyzer"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Ingest command
    subparsers.add_parser("ingest", help="Ingest PDFs from data/papers/")

    # Query command
    query_parser = subparsers.add_parser("query", help="Query the knowledge base")
    query_parser.add_argument("query", type=str, help="Your question")

    # Interactive command
    subparsers.add_parser("interactive", help="Interactive query mode")

    # Stats command
    subparsers.add_parser("stats", help="Show system statistics")

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Run evaluation metrics")
    eval_parser.add_argument("--query", type=str, default=None,
                             help="Single query to evaluate")
    eval_parser.add_argument("--all", action="store_true",
                             help="Run full test dataset evaluation")
    eval_parser.add_argument("--save", action="store_true",
                             help="Save results to data/evaluation_results.json")

    # Security command
    subparsers.add_parser("security", help="Run security tests")

    args = parser.parse_args()

    if args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "query":
        cmd_query(args)
    elif args.command == "interactive":
        cmd_interactive(args)
    elif args.command == "stats":
        cmd_stats(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "security":
        print("Running security tests...")
        pipeline = build_query_pipeline()
        from security.prompt_injection_test import PromptInjectionTest
        tester = PromptInjectionTest(router=pipeline["router"])
        results = tester.run_all_tests()
        for category, data in results.items():
            if isinstance(data, dict) and "pass_rate" in data:
                print(f"  {category}: {data['pass_rate']:.0%} pass rate")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
```

4th agent that fact-checks the explanation against retrieved context. If hallucinations are found, the Router uses the corrected answer.

---

### Phase 5 — Sliding Window Chunking
```diff:boundary_detector.py
"""
Boundary Detector Module
Splits sections into logical chunks for embedding and retrieval.
Uses a combination of section boundaries and paragraph-level splitting.
"""

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
    Splits text into chunks suitable for embedding.

    Strategy:
    1. Respect section boundaries from StructureAnalyzer.
    2. Within sections, split at paragraph boundaries.
    3. If a paragraph is too large, split at sentence boundaries.
    4. Maintain overlap between consecutive chunks for context continuity.
    """

    def __init__(
        self,
        max_chunk_size: int = 512,
        min_chunk_size: int = 100,
        overlap_size: int = 50
    ):
        """
        Args:
            max_chunk_size: Maximum number of characters per chunk.
            min_chunk_size: Minimum number of characters per chunk (merge small ones).
            overlap_size: Number of overlapping characters between consecutive chunks.
        """
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap_size = overlap_size

    def chunk_section(
        self,
        content: str,
        paper_id: str,
        section_id: str,
        section_heading: str
    ) -> List[TextChunk]:
        """
        Split a section's content into chunks.

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

        # Split into paragraphs first
        paragraphs = self._split_paragraphs(content)

        # Merge small paragraphs and split large ones
        raw_chunks = self._balance_chunks(paragraphs)

        # Create TextChunk objects with overlap
        chunks = []
        char_offset = 0

        for i, chunk_text in enumerate(raw_chunks):
            chunk = TextChunk(
                chunk_id=f"{paper_id}_{section_id}_chunk_{i}",
                paper_id=paper_id,
                section_heading=section_heading,
                content=chunk_text.strip(),
                char_start=char_offset,
                char_end=char_offset + len(chunk_text)
            )
            chunks.append(chunk)
            char_offset += len(chunk_text)

        return chunks

    def chunk_document(
        self,
        sections: list,
        paper_id: str
    ) -> List[TextChunk]:
        """
        Chunk all sections of a document.

        Args:
            sections: List of Section objects from StructureAnalyzer.
            paper_id: Identifier for the source paper.

        Returns:
            List of all TextChunk objects across all sections.
        """
        all_chunks = []
        for section in sections:
            section_chunks = self.chunk_section(
                content=section.content,
                paper_id=paper_id,
                section_id=section.section_id,
                section_heading=section.heading
            )
            all_chunks.extend(section_chunks)
        return all_chunks

    def _split_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs (double newline separated)."""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        return paragraphs

    def _balance_chunks(self, paragraphs: List[str]) -> List[str]:
        """
        Balance paragraphs into chunks:
        - Merge short adjacent paragraphs.
        - Split overly long paragraphs by sentence.
        """
        chunks = []
        current_chunk = ""

        for para in paragraphs:
            if len(para) > self.max_chunk_size:
                # Flush current buffer
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                # Split long paragraph by sentences
                sentences = self._split_sentences(para)
                sent_buffer = ""
                for sent in sentences:
                    if len(sent_buffer) + len(sent) + 1 > self.max_chunk_size:
                        if sent_buffer:
                            chunks.append(sent_buffer)
                        sent_buffer = sent
                    else:
                        sent_buffer = (sent_buffer + " " + sent).strip()
                if sent_buffer:
                    chunks.append(sent_buffer)
            elif len(current_chunk) + len(para) + 2 > self.max_chunk_size:
                # Current buffer would overflow — flush it
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = para
            else:
                # Merge paragraph into current buffer
                current_chunk = (current_chunk + "\n\n" + para).strip()

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        """Simple sentence splitter based on period + space."""
        import re
        # Split on sentence-ending punctuation followed by space or end of string
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
===
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

```

Token-based sliding window (default window=200 words, stride=100 words). Backward-compatible with old `max_chunk_size`/`overlap_size` params. Small trailing chunks are merged into the last chunk.

---

### Phase 6 — LLM-Judged Relevance Labels
```diff:rag_metrics.py
"""
RAG Metrics Module
Comprehensive evaluation of RAG system performance.

Three metric categories:
1. Retrieval Quality: Precision@K, Recall@K, F1@K, MRR, MAP, NDCG@K, Hit Rate
2. Generation Quality: Semantic Similarity, BLEU, ROUGE-L, Faithfulness, Completeness
3. System Performance: Latency breakdown
"""

import re
import math
import numpy as np
from typing import List, Dict, Any, Optional, Set


# ═══════════════════════════════════════════════════════════════════════════════
# RETRIEVAL QUALITY METRICS
# ═══════════════════════════════════════════════════════════════════════════════

class RetrievalMetrics:
    """
    Evaluates retrieval quality by comparing retrieved chunks against
    ground-truth relevant sections/keywords.
    """

    def evaluate_all(
        self,
        retrieved_sections: List[str],
        retrieved_chunk_ids: List[str],
        relevant_sections: List[str],
        relevant_keywords: List[str],
        retrieved_contents: List[str],
        k: int = 5
    ) -> Dict[str, Any]:
        """
        Run all retrieval metrics.

        Args:
            retrieved_sections: Section headings of retrieved chunks (ordered by rank).
            retrieved_chunk_ids: IDs of retrieved chunks.
            relevant_sections: Ground-truth relevant section headings.
            relevant_keywords: Ground-truth relevant keywords.
            retrieved_contents: Text content of retrieved chunks.
            k: Number of top results to evaluate.

        Returns:
            Dict with all retrieval metric scores.
        """
        # Build relevance labels: 1 if chunk's section matches any relevant section
        relevance = self._build_relevance_labels(
            retrieved_sections, retrieved_contents,
            relevant_sections, relevant_keywords
        )

        results = {}
        results["precision_at_k"] = self.precision_at_k(relevance, k)
        results["recall_at_k"] = self.recall_at_k(relevance, k, len(relevant_sections))
        results["f1_at_k"] = self.f1_at_k(results["precision_at_k"], results["recall_at_k"])
        results["mrr"] = self.mrr(relevance)
        results["map"] = self.average_precision(relevance)
        results["ndcg_at_k"] = self.ndcg_at_k(relevance, k)
        results["hit_rate_at_k"] = self.hit_rate_at_k(relevance, k)

        return results

    def _build_relevance_labels(
        self,
        retrieved_sections: List[str],
        retrieved_contents: List[str],
        relevant_sections: List[str],
        relevant_keywords: List[str]
    ) -> List[int]:
        """
        Determine which retrieved chunks are relevant.
        A chunk is relevant if its section heading matches a relevant section
        OR if it contains relevant keywords.
        """
        relevant_set = {s.lower().strip() for s in relevant_sections}
        keyword_set = {kw.lower().strip() for kw in relevant_keywords}

        labels = []
        for i, section in enumerate(retrieved_sections):
            section_lower = section.lower().strip()
            content_lower = retrieved_contents[i].lower() if i < len(retrieved_contents) else ""

            # Check section heading match (substring match for flexibility)
            section_match = any(
                rel in section_lower or section_lower in rel
                for rel in relevant_set
            )

            # Check keyword presence in content
            keyword_matches = sum(1 for kw in keyword_set if kw in content_lower)
            keyword_match = keyword_matches >= 2  # At least 2 keywords present

            labels.append(1 if (section_match or keyword_match) else 0)

        return labels

    def precision_at_k(self, relevance: List[int], k: int) -> Dict[str, Any]:
        """
        Precision@K = (relevant items in top-K) / K
        """
        top_k = relevance[:k]
        if not top_k:
            return {"score": 0.0, "explanation": "No results to evaluate."}

        relevant_count = sum(top_k)
        score = relevant_count / len(top_k)

        return {
            "score": round(score, 3),
            "relevant_in_top_k": relevant_count,
            "k": len(top_k),
            "explanation": f"{relevant_count}/{len(top_k)} retrieved chunks are relevant."
        }

    def recall_at_k(self, relevance: List[int], k: int, total_relevant: int) -> Dict[str, Any]:
        """
        Recall@K = (relevant items in top-K) / (total relevant items)
        """
        top_k = relevance[:k]
        if total_relevant == 0:
            return {"score": 0.0, "explanation": "No relevant items defined."}

        relevant_count = sum(top_k)
        score = relevant_count / total_relevant

        return {
            "score": round(min(score, 1.0), 3),
            "relevant_found": relevant_count,
            "total_relevant": total_relevant,
            "explanation": f"{relevant_count}/{total_relevant} relevant items found in top-{len(top_k)}."
        }

    def f1_at_k(self, precision: Dict, recall: Dict) -> Dict[str, Any]:
        """
        F1@K = 2 × (Precision × Recall) / (Precision + Recall)
        """
        p = precision["score"]
        r = recall["score"]

        if p + r == 0:
            score = 0.0
        else:
            score = 2 * (p * r) / (p + r)

        return {
            "score": round(score, 3),
            "precision": p,
            "recall": r,
            "explanation": f"F1 = {score:.3f} (P={p:.3f}, R={r:.3f})"
        }

    def mrr(self, relevance: List[int]) -> Dict[str, Any]:
        """
        Mean Reciprocal Rank = 1 / (rank of first relevant result)
        """
        for i, rel in enumerate(relevance):
            if rel == 1:
                rank = i + 1
                score = 1.0 / rank
                return {
                    "score": round(score, 3),
                    "first_relevant_rank": rank,
                    "explanation": f"First relevant result at rank {rank}. MRR = 1/{rank} = {score:.3f}"
                }

        return {
            "score": 0.0,
            "first_relevant_rank": None,
            "explanation": "No relevant results found."
        }

    def average_precision(self, relevance: List[int]) -> Dict[str, Any]:
        """
        Average Precision (AP) = mean of precision values at each relevant position.
        MAP is the mean of AP across multiple queries (computed in EvaluationRunner).
        """
        if not relevance or sum(relevance) == 0:
            return {"score": 0.0, "explanation": "No relevant results found."}

        precision_sum = 0.0
        relevant_count = 0

        for i, rel in enumerate(relevance):
            if rel == 1:
                relevant_count += 1
                precision_at_i = relevant_count / (i + 1)
                precision_sum += precision_at_i

        score = precision_sum / relevant_count if relevant_count > 0 else 0.0

        return {
            "score": round(score, 3),
            "relevant_positions": relevant_count,
            "explanation": f"Average Precision across {relevant_count} relevant positions = {score:.3f}"
        }

    def ndcg_at_k(self, relevance: List[int], k: int) -> Dict[str, Any]:
        """
        Normalized Discounted Cumulative Gain@K.
        NDCG = DCG / Ideal DCG
        DCG = sum(relevance[i] / log2(i+2)) for i in [0, k)
        """
        top_k = relevance[:k]
        if not top_k:
            return {"score": 0.0, "explanation": "No results to evaluate."}

        # DCG
        dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(top_k))

        # Ideal DCG (all relevant first)
        ideal = sorted(top_k, reverse=True)
        idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal))

        score = dcg / idcg if idcg > 0 else 0.0

        return {
            "score": round(score, 3),
            "dcg": round(dcg, 3),
            "idcg": round(idcg, 3),
            "explanation": f"NDCG@{len(top_k)} = {score:.3f} (DCG={dcg:.3f}, IDCG={idcg:.3f})"
        }

    def hit_rate_at_k(self, relevance: List[int], k: int) -> Dict[str, Any]:
        """
        Hit Rate@K = 1 if any relevant result in top-K, else 0.
        """
        top_k = relevance[:k]
        hit = 1 if any(r == 1 for r in top_k) else 0

        return {
            "score": float(hit),
            "explanation": f"{'At least one' if hit else 'No'} relevant result in top-{len(top_k)}."
        }


# ═══════════════════════════════════════════════════════════════════════════════
# GENERATION QUALITY METRICS
# ═══════════════════════════════════════════════════════════════════════════════

class GenerationMetrics:
    """
    Evaluates the quality of generated answers against ground truth
    and retrieved context.
    """

    def __init__(self, llm_client=None, embedder=None):
        """
        Args:
            llm_client: Optional LLM client for LLM-based metrics.
            embedder: Optional Embedder instance for semantic similarity.
        """
        self.llm_client = llm_client
        self.embedder = embedder

    def evaluate_all(
        self,
        query: str,
        answer: str,
        ground_truth: str,
        context: str
    ) -> Dict[str, Any]:
        """
        Run all generation quality metrics.

        Args:
            query: The user's question.
            answer: The system-generated answer.
            ground_truth: The expected correct answer.
            context: The retrieved context used to generate the answer.

        Returns:
            Dict with all generation metric scores.
        """
        results = {}

        results["semantic_similarity"] = self.semantic_similarity(answer, ground_truth)
        results["bleu_score"] = self.bleu_score(answer, ground_truth)
        results["rouge_l_score"] = self.rouge_l_score(answer, ground_truth)
        results["faithfulness"] = self.faithfulness(answer, context)
        results["answer_completeness"] = self.answer_completeness(answer, ground_truth, query)

        return results

    def semantic_similarity(self, answer: str, ground_truth: str) -> Dict[str, Any]:
        """
        Cosine similarity between answer and ground truth embeddings.
        Uses the existing Embedder (sentence-transformers).
        """
        if not self.embedder:
            return {"score": 0.0, "explanation": "No embedder available for semantic similarity."}

        if not answer.strip() or not ground_truth.strip():
            return {"score": 0.0, "explanation": "Empty answer or ground truth."}

        try:
            answer_emb = self.embedder.embed_text(answer)
            truth_emb = self.embedder.embed_text(ground_truth)

            # Cosine similarity
            dot = np.dot(answer_emb, truth_emb)
            norm_a = np.linalg.norm(answer_emb)
            norm_t = np.linalg.norm(truth_emb)
            score = float(dot / (norm_a * norm_t)) if (norm_a * norm_t) > 0 else 0.0

            # Clamp to [0, 1]
            score = max(0.0, min(1.0, score))

            return {
                "score": round(score, 3),
                "explanation": f"Cosine similarity between answer and ground truth embeddings = {score:.3f}"
            }
        except Exception as e:
            return {"score": 0.0, "explanation": f"Error computing semantic similarity: {e}"}

    def bleu_score(self, answer: str, ground_truth: str) -> Dict[str, Any]:
        """
        BLEU score (1-4 gram) using nltk.
        Measures n-gram precision of the answer vs ground truth.
        """
        if not answer.strip() or not ground_truth.strip():
            return {"score": 0.0, "explanation": "Empty answer or ground truth."}

        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

            reference = [ground_truth.lower().split()]
            hypothesis = answer.lower().split()

            # Use smoothing to avoid 0 scores for short texts
            smoothing = SmoothingFunction().method1

            # Individual n-gram scores
            bleu_1 = sentence_bleu(reference, hypothesis, weights=(1, 0, 0, 0), smoothing_function=smoothing)
            bleu_2 = sentence_bleu(reference, hypothesis, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
            bleu_4 = sentence_bleu(reference, hypothesis, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)

            return {
                "score": round(bleu_4, 3),
                "bleu_1": round(bleu_1, 3),
                "bleu_2": round(bleu_2, 3),
                "bleu_4": round(bleu_4, 3),
                "explanation": f"BLEU-4={bleu_4:.3f} (BLEU-1={bleu_1:.3f}, BLEU-2={bleu_2:.3f})"
            }
        except ImportError:
            return {"score": 0.0, "explanation": "nltk not available for BLEU calculation."}
        except Exception as e:
            return {"score": 0.0, "explanation": f"Error computing BLEU: {e}"}

    def rouge_l_score(self, answer: str, ground_truth: str) -> Dict[str, Any]:
        """
        ROUGE-L score based on Longest Common Subsequence (LCS).
        No external dependencies — custom implementation.

        ROUGE-L Precision = LCS / len(answer)
        ROUGE-L Recall    = LCS / len(ground_truth)
        ROUGE-L F1        = 2 × P × R / (P + R)
        """
        if not answer.strip() or not ground_truth.strip():
            return {"score": 0.0, "explanation": "Empty answer or ground truth."}

        answer_tokens = answer.lower().split()
        truth_tokens = ground_truth.lower().split()

        lcs_length = self._lcs_length(answer_tokens, truth_tokens)

        precision = lcs_length / len(answer_tokens) if answer_tokens else 0.0
        recall = lcs_length / len(truth_tokens) if truth_tokens else 0.0

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        return {
            "score": round(f1, 3),
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "lcs_length": lcs_length,
            "explanation": f"ROUGE-L F1={f1:.3f} (P={precision:.3f}, R={recall:.3f}, LCS={lcs_length} tokens)"
        }

    def _lcs_length(self, x: List[str], y: List[str]) -> int:
        """Compute length of Longest Common Subsequence using dynamic programming."""
        m, n = len(x), len(y)
        # Space-optimized: only need two rows
        prev = [0] * (n + 1)
        curr = [0] * (n + 1)

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i - 1] == y[j - 1]:
                    curr[j] = prev[j - 1] + 1
                else:
                    curr[j] = max(prev[j], curr[j - 1])
            prev, curr = curr, [0] * (n + 1)

        return prev[n]

    def faithfulness(self, answer: str, context: str) -> Dict[str, Any]:
        """
        Measure whether the answer is grounded in the retrieved context.
        Uses LLM if available; falls back to heuristic keyword overlap.
        """
        if not answer.strip() or not context.strip():
            return {"score": 0.0, "explanation": "Empty answer or context."}

        if self.llm_client:
            return self._llm_faithfulness(answer, context)

        return self._heuristic_faithfulness(answer, context)

    def _heuristic_faithfulness(self, answer: str, context: str) -> Dict[str, Any]:
        """Heuristic faithfulness: check sentence-level keyword grounding."""
        answer_sentences = self._split_sentences(answer)
        context_lower = context.lower()

        grounded_count = 0
        total = len(answer_sentences)

        for sent in answer_sentences:
            keywords = self._extract_keywords(sent)
            if not keywords:
                continue
            match_count = sum(1 for kw in keywords if kw in context_lower)
            if match_count / len(keywords) > 0.3:
                grounded_count += 1

        score = grounded_count / total if total > 0 else 0.0

        return {
            "score": round(score, 3),
            "grounded_sentences": grounded_count,
            "total_sentences": total,
            "explanation": f"{grounded_count}/{total} answer sentences are grounded in context."
        }

    def _llm_faithfulness(self, answer: str, context: str) -> Dict[str, Any]:
        """LLM-based faithfulness evaluation."""
        prompt = (
            "You are evaluating whether an AI-generated answer is faithful to the provided context.\n"
            "A faithful answer only contains claims that can be verified from the context.\n\n"
            f"Context:\n{context[:3000]}\n\n"
            f"Answer:\n{answer}\n\n"
            "Score the faithfulness from 0.0 (completely hallucinated) to 1.0 (fully grounded).\n"
            "Respond with ONLY a JSON object: {\"score\": <float>, \"explanation\": \"<text>\"}"
        )

        try:
            import json
            response = self.llm_client.generate(prompt, max_tokens=200, temperature=0.1)
            result = json.loads(response)
            return {
                "score": round(float(result.get("score", 0)), 3),
                "explanation": result.get("explanation", "LLM evaluation.")
            }
        except Exception as e:
            # Fallback to heuristic
            return self._heuristic_faithfulness(answer, context)

    def answer_completeness(self, answer: str, ground_truth: str, query: str) -> Dict[str, Any]:
        """
        Evaluate whether the answer covers all key points from the ground truth.
        Uses LLM if available; falls back to keyword coverage.
        """
        if not answer.strip() or not ground_truth.strip():
            return {"score": 0.0, "explanation": "Empty answer or ground truth."}

        if self.llm_client:
            return self._llm_completeness(answer, ground_truth, query)

        return self._heuristic_completeness(answer, ground_truth)

    def _heuristic_completeness(self, answer: str, ground_truth: str) -> Dict[str, Any]:
        """Heuristic completeness: keyword coverage of ground truth."""
        truth_keywords = set(self._extract_keywords(ground_truth))
        answer_keywords = set(self._extract_keywords(answer))

        if not truth_keywords:
            return {"score": 0.0, "explanation": "No keywords in ground truth."}

        covered = truth_keywords.intersection(answer_keywords)
        score = len(covered) / len(truth_keywords)

        return {
            "score": round(min(score, 1.0), 3),
            "covered_keywords": len(covered),
            "total_keywords": len(truth_keywords),
            "explanation": f"{len(covered)}/{len(truth_keywords)} ground truth keywords covered in answer."
        }

    def _llm_completeness(self, answer: str, ground_truth: str, query: str) -> Dict[str, Any]:
        """LLM-based answer completeness evaluation."""
        prompt = (
            "You are evaluating whether an AI-generated answer completely covers the expected answer.\n\n"
            f"Question: {query}\n\n"
            f"Expected Answer:\n{ground_truth}\n\n"
            f"Generated Answer:\n{answer}\n\n"
            "Score the completeness from 0.0 (misses everything) to 1.0 (covers all key points).\n"
            "Respond with ONLY a JSON object: {\"score\": <float>, \"explanation\": \"<text>\"}"
        )

        try:
            import json
            response = self.llm_client.generate(prompt, max_tokens=200, temperature=0.1)
            result = json.loads(response)
            return {
                "score": round(float(result.get("score", 0)), 3),
                "explanation": result.get("explanation", "LLM evaluation.")
            }
        except Exception as e:
            return self._heuristic_completeness(answer, ground_truth)

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from text."""
        stopwords = {
            "the", "a", "an", "is", "was", "are", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "can", "and",
            "or", "but", "in", "on", "at", "to", "for", "of", "with",
            "by", "from", "as", "it", "its", "this", "that", "not",
            "such", "these", "those", "which", "who", "how", "what",
            "when", "where", "than", "also", "each", "more", "other",
        }
        words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        return [w for w in words if w not in stopwords]

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]


# ═══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

class RAGMetrics:
    """
    Orchestrates all evaluation metrics: Retrieval + Generation + Latency.
    """

    def __init__(self, llm_client=None, embedder=None):
        """
        Args:
            llm_client: Optional LLM client for LLM-based evaluation.
            embedder: Optional Embedder instance for semantic similarity.
        """
        self.retrieval_metrics = RetrievalMetrics()
        self.generation_metrics = GenerationMetrics(llm_client=llm_client, embedder=embedder)

    def evaluate(
        self,
        query: str,
        answer: str,
        context: str,
        ground_truth: str = "",
        relevant_sections: List[str] = None,
        relevant_keywords: List[str] = None,
        retrieved_sections: List[str] = None,
        retrieved_chunk_ids: List[str] = None,
        retrieved_contents: List[str] = None,
        timing: Optional[Dict[str, float]] = None,
        k: int = 5
    ) -> Dict[str, Any]:
        """
        Run all evaluation metrics.

        Args:
            query: The user's question.
            answer: The system-generated answer.
            context: The retrieved context used to generate the answer.
            ground_truth: Expected correct answer.
            relevant_sections: Ground-truth relevant section headings.
            relevant_keywords: Ground-truth relevant keywords.
            retrieved_sections: Section headings of retrieved chunks (ordered).
            retrieved_chunk_ids: IDs of retrieved chunks.
            retrieved_contents: Text content of retrieved chunks.
            timing: Pipeline timing dict.
            k: Top-K for retrieval metrics.

        Returns:
            Dict with all metric scores organized by category.
        """
        results = {"retrieval": {}, "generation": {}, "system": {}}

        # ── Retrieval Metrics ──
        if (retrieved_sections and relevant_sections):
            results["retrieval"] = self.retrieval_metrics.evaluate_all(
                retrieved_sections=retrieved_sections or [],
                retrieved_chunk_ids=retrieved_chunk_ids or [],
                relevant_sections=relevant_sections or [],
                relevant_keywords=relevant_keywords or [],
                retrieved_contents=retrieved_contents or [],
                k=k
            )

        # ── Generation Metrics ──
        if ground_truth:
            results["generation"] = self.generation_metrics.evaluate_all(
                query=query,
                answer=answer,
                ground_truth=ground_truth,
                context=context
            )
        else:
            # Without ground truth, only faithfulness is available
            results["generation"]["faithfulness"] = self.generation_metrics.faithfulness(
                answer, context
            )

        # ── System Metrics ──
        if timing:
            results["system"]["latency"] = {
                "total_seconds": round(sum(timing.values()), 3),
                "breakdown": timing,
                "explanation": f"Total pipeline time: {sum(timing.values()):.3f}s"
            }

        # ── Overall Scores ──
        all_scores = []
        for category in ["retrieval", "generation"]:
            for metric_name, metric_data in results.get(category, {}).items():
                if isinstance(metric_data, dict) and "score" in metric_data:
                    all_scores.append(metric_data["score"])

        results["overall_score"] = round(
            sum(all_scores) / len(all_scores), 3
        ) if all_scores else 0.0

        return results
===
"""
RAG Metrics Module
Comprehensive evaluation of RAG system performance.

Three metric categories:
1. Retrieval Quality: Precision@K, Recall@K, F1@K, MRR, MAP, NDCG@K, Hit Rate
2. Generation Quality: Semantic Similarity, BLEU, ROUGE-L, Faithfulness, Completeness
3. System Performance: Latency breakdown
"""

import re
import math
import numpy as np
from typing import List, Dict, Any, Optional, Set


# ═══════════════════════════════════════════════════════════════════════════════
# RETRIEVAL QUALITY METRICS
# ═══════════════════════════════════════════════════════════════════════════════

class RetrievalMetrics:
    """
    Evaluates retrieval quality by comparing retrieved chunks against
    ground-truth relevant sections/keywords.
    Uses LLM-as-a-Judge for relevance labeling when available.
    """

    def __init__(self, llm_client=None):
        """
        Args:
            llm_client: Optional LLM client for LLM-based relevance judging.
        """
        self.llm_client = llm_client

    def evaluate_all(
        self,
        retrieved_sections: List[str],
        retrieved_chunk_ids: List[str],
        relevant_sections: List[str],
        relevant_keywords: List[str],
        retrieved_contents: List[str],
        k: int = 5,
        query: str = "",
    ) -> Dict[str, Any]:
        """
        Run all retrieval metrics.

        Args:
            retrieved_sections: Section headings of retrieved chunks (ordered by rank).
            retrieved_chunk_ids: IDs of retrieved chunks.
            relevant_sections: Ground-truth relevant section headings.
            relevant_keywords: Ground-truth relevant keywords.
            retrieved_contents: Text content of retrieved chunks.
            k: Number of top results to evaluate.
            query: The user query (required for LLM-based relevance judging).

        Returns:
            Dict with all retrieval metric scores.
        """
        # Build relevance labels
        relevance = self._build_relevance_labels(
            query, retrieved_sections, retrieved_contents,
            relevant_sections, relevant_keywords
        )

        results = {}
        results["precision_at_k"] = self.precision_at_k(relevance, k)
        results["recall_at_k"] = self.recall_at_k(relevance, k, len(relevant_sections))
        results["f1_at_k"] = self.f1_at_k(results["precision_at_k"], results["recall_at_k"])
        results["mrr"] = self.mrr(relevance)
        results["map"] = self.average_precision(relevance)
        results["ndcg_at_k"] = self.ndcg_at_k(relevance, k)
        results["hit_rate_at_k"] = self.hit_rate_at_k(relevance, k)

        return results

    def _build_relevance_labels(
        self,
        query: str,
        retrieved_sections: List[str],
        retrieved_contents: List[str],
        relevant_sections: List[str],
        relevant_keywords: List[str],
    ) -> List[int]:
        """
        Determine which retrieved chunks are relevant.
        Uses LLM-as-a-Judge when available; falls back to heuristic.
        """
        # ── LLM-based relevance judging (primary) ──
        if self.llm_client and query:
            llm_labels = self._llm_build_relevance_labels(
                query, retrieved_contents
            )
            if llm_labels is not None:
                return llm_labels

        # ── Heuristic fallback ──
        return self._heuristic_relevance_labels(
            retrieved_sections, retrieved_contents,
            relevant_sections, relevant_keywords
        )

    def _llm_build_relevance_labels(
        self, query: str, retrieved_contents: List[str]
    ) -> Optional[List[int]]:
        """
        Use the LLM to judge relevance of each retrieved chunk to the query.

        Returns:
            List of 0/1 labels, or None on failure.
        """
        import json as _json

        labels: List[int] = []
        for i, content in enumerate(retrieved_contents):
            # Truncate long chunks to save tokens
            snippet = content[:1500] if len(content) > 1500 else content

            prompt = (
                "You are a relevance judge for an academic research RAG system.\n\n"
                f"Query: {query}\n\n"
                f"Document chunk:\n{snippet}\n\n"
                "Is this document chunk relevant to answering the query?\n"
                'Respond with ONLY a JSON object: {"relevant": 0} or {"relevant": 1}'
            )

            try:
                response = self.llm_client.generate(
                    prompt=prompt, max_tokens=20, temperature=0.0
                )
                cleaned = response.strip()
                if cleaned.startswith("```"):
                    cleaned = cleaned.split("\n", 1)[-1]
                    if cleaned.endswith("```"):
                        cleaned = cleaned[:-3]
                    cleaned = cleaned.strip()
                parsed = _json.loads(cleaned)
                labels.append(1 if parsed.get("relevant", 0) else 0)
            except Exception:
                # If any single chunk fails, abort LLM path entirely
                print(f"[RetrievalMetrics] LLM relevance judging failed on chunk {i}. "
                      "Falling back to heuristic.")
                return None

        return labels

    def _heuristic_relevance_labels(
        self,
        retrieved_sections: List[str],
        retrieved_contents: List[str],
        relevant_sections: List[str],
        relevant_keywords: List[str],
    ) -> List[int]:
        """
        Heuristic fallback: determine relevance via section heading match
        and keyword overlap.
        """
        relevant_set = {s.lower().strip() for s in relevant_sections}
        keyword_set = {kw.lower().strip() for kw in relevant_keywords}

        labels = []
        for i, section in enumerate(retrieved_sections):
            section_lower = section.lower().strip()
            content_lower = retrieved_contents[i].lower() if i < len(retrieved_contents) else ""

            # Check section heading match (substring match for flexibility)
            section_match = any(
                rel in section_lower or section_lower in rel
                for rel in relevant_set
            )

            # Check keyword presence in content
            keyword_matches = sum(1 for kw in keyword_set if kw in content_lower)
            keyword_match = keyword_matches >= 2  # At least 2 keywords present

            labels.append(1 if (section_match or keyword_match) else 0)

        return labels

    def precision_at_k(self, relevance: List[int], k: int) -> Dict[str, Any]:
        """
        Precision@K = (relevant items in top-K) / K
        """
        top_k = relevance[:k]
        if not top_k:
            return {"score": 0.0, "explanation": "No results to evaluate."}

        relevant_count = sum(top_k)
        score = relevant_count / len(top_k)

        return {
            "score": round(score, 3),
            "relevant_in_top_k": relevant_count,
            "k": len(top_k),
            "explanation": f"{relevant_count}/{len(top_k)} retrieved chunks are relevant."
        }

    def recall_at_k(self, relevance: List[int], k: int, total_relevant: int) -> Dict[str, Any]:
        """
        Recall@K = (relevant items in top-K) / (total relevant items)
        """
        top_k = relevance[:k]
        if total_relevant == 0:
            return {"score": 0.0, "explanation": "No relevant items defined."}

        relevant_count = sum(top_k)
        score = relevant_count / total_relevant

        return {
            "score": round(min(score, 1.0), 3),
            "relevant_found": relevant_count,
            "total_relevant": total_relevant,
            "explanation": f"{relevant_count}/{total_relevant} relevant items found in top-{len(top_k)}."
        }

    def f1_at_k(self, precision: Dict, recall: Dict) -> Dict[str, Any]:
        """
        F1@K = 2 × (Precision × Recall) / (Precision + Recall)
        """
        p = precision["score"]
        r = recall["score"]

        if p + r == 0:
            score = 0.0
        else:
            score = 2 * (p * r) / (p + r)

        return {
            "score": round(score, 3),
            "precision": p,
            "recall": r,
            "explanation": f"F1 = {score:.3f} (P={p:.3f}, R={r:.3f})"
        }

    def mrr(self, relevance: List[int]) -> Dict[str, Any]:
        """
        Mean Reciprocal Rank = 1 / (rank of first relevant result)
        """
        for i, rel in enumerate(relevance):
            if rel == 1:
                rank = i + 1
                score = 1.0 / rank
                return {
                    "score": round(score, 3),
                    "first_relevant_rank": rank,
                    "explanation": f"First relevant result at rank {rank}. MRR = 1/{rank} = {score:.3f}"
                }

        return {
            "score": 0.0,
            "first_relevant_rank": None,
            "explanation": "No relevant results found."
        }

    def average_precision(self, relevance: List[int]) -> Dict[str, Any]:
        """
        Average Precision (AP) = mean of precision values at each relevant position.
        MAP is the mean of AP across multiple queries (computed in EvaluationRunner).
        """
        if not relevance or sum(relevance) == 0:
            return {"score": 0.0, "explanation": "No relevant results found."}

        precision_sum = 0.0
        relevant_count = 0

        for i, rel in enumerate(relevance):
            if rel == 1:
                relevant_count += 1
                precision_at_i = relevant_count / (i + 1)
                precision_sum += precision_at_i

        score = precision_sum / relevant_count if relevant_count > 0 else 0.0

        return {
            "score": round(score, 3),
            "relevant_positions": relevant_count,
            "explanation": f"Average Precision across {relevant_count} relevant positions = {score:.3f}"
        }

    def ndcg_at_k(self, relevance: List[int], k: int) -> Dict[str, Any]:
        """
        Normalized Discounted Cumulative Gain@K.
        NDCG = DCG / Ideal DCG
        DCG = sum(relevance[i] / log2(i+2)) for i in [0, k)
        """
        top_k = relevance[:k]
        if not top_k:
            return {"score": 0.0, "explanation": "No results to evaluate."}

        # DCG
        dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(top_k))

        # Ideal DCG (all relevant first)
        ideal = sorted(top_k, reverse=True)
        idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal))

        score = dcg / idcg if idcg > 0 else 0.0

        return {
            "score": round(score, 3),
            "dcg": round(dcg, 3),
            "idcg": round(idcg, 3),
            "explanation": f"NDCG@{len(top_k)} = {score:.3f} (DCG={dcg:.3f}, IDCG={idcg:.3f})"
        }

    def hit_rate_at_k(self, relevance: List[int], k: int) -> Dict[str, Any]:
        """
        Hit Rate@K = 1 if any relevant result in top-K, else 0.
        """
        top_k = relevance[:k]
        hit = 1 if any(r == 1 for r in top_k) else 0

        return {
            "score": float(hit),
            "explanation": f"{'At least one' if hit else 'No'} relevant result in top-{len(top_k)}."
        }


# ═══════════════════════════════════════════════════════════════════════════════
# GENERATION QUALITY METRICS
# ═══════════════════════════════════════════════════════════════════════════════

class GenerationMetrics:
    """
    Evaluates the quality of generated answers against ground truth
    and retrieved context.
    """

    def __init__(self, llm_client=None, embedder=None):
        """
        Args:
            llm_client: Optional LLM client for LLM-based metrics.
            embedder: Optional Embedder instance for semantic similarity.
        """
        self.llm_client = llm_client
        self.embedder = embedder

    def evaluate_all(
        self,
        query: str,
        answer: str,
        ground_truth: str,
        context: str
    ) -> Dict[str, Any]:
        """
        Run all generation quality metrics.

        Args:
            query: The user's question.
            answer: The system-generated answer.
            ground_truth: The expected correct answer.
            context: The retrieved context used to generate the answer.

        Returns:
            Dict with all generation metric scores.
        """
        results = {}

        results["semantic_similarity"] = self.semantic_similarity(answer, ground_truth)
        results["bleu_score"] = self.bleu_score(answer, ground_truth)
        results["rouge_l_score"] = self.rouge_l_score(answer, ground_truth)
        results["faithfulness"] = self.faithfulness(answer, context)
        results["answer_completeness"] = self.answer_completeness(answer, ground_truth, query)

        return results

    def semantic_similarity(self, answer: str, ground_truth: str) -> Dict[str, Any]:
        """
        Cosine similarity between answer and ground truth embeddings.
        Uses the existing Embedder (sentence-transformers).
        """
        if not self.embedder:
            return {"score": 0.0, "explanation": "No embedder available for semantic similarity."}

        if not answer.strip() or not ground_truth.strip():
            return {"score": 0.0, "explanation": "Empty answer or ground truth."}

        try:
            answer_emb = self.embedder.embed_text(answer)
            truth_emb = self.embedder.embed_text(ground_truth)

            # Cosine similarity
            dot = np.dot(answer_emb, truth_emb)
            norm_a = np.linalg.norm(answer_emb)
            norm_t = np.linalg.norm(truth_emb)
            score = float(dot / (norm_a * norm_t)) if (norm_a * norm_t) > 0 else 0.0

            # Clamp to [0, 1]
            score = max(0.0, min(1.0, score))

            return {
                "score": round(score, 3),
                "explanation": f"Cosine similarity between answer and ground truth embeddings = {score:.3f}"
            }
        except Exception as e:
            return {"score": 0.0, "explanation": f"Error computing semantic similarity: {e}"}

    def bleu_score(self, answer: str, ground_truth: str) -> Dict[str, Any]:
        """
        BLEU score (1-4 gram) using nltk.
        Measures n-gram precision of the answer vs ground truth.
        """
        if not answer.strip() or not ground_truth.strip():
            return {"score": 0.0, "explanation": "Empty answer or ground truth."}

        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

            reference = [ground_truth.lower().split()]
            hypothesis = answer.lower().split()

            # Use smoothing to avoid 0 scores for short texts
            smoothing = SmoothingFunction().method1

            # Individual n-gram scores
            bleu_1 = sentence_bleu(reference, hypothesis, weights=(1, 0, 0, 0), smoothing_function=smoothing)
            bleu_2 = sentence_bleu(reference, hypothesis, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
            bleu_4 = sentence_bleu(reference, hypothesis, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)

            return {
                "score": round(bleu_4, 3),
                "bleu_1": round(bleu_1, 3),
                "bleu_2": round(bleu_2, 3),
                "bleu_4": round(bleu_4, 3),
                "explanation": f"BLEU-4={bleu_4:.3f} (BLEU-1={bleu_1:.3f}, BLEU-2={bleu_2:.3f})"
            }
        except ImportError:
            return {"score": 0.0, "explanation": "nltk not available for BLEU calculation."}
        except Exception as e:
            return {"score": 0.0, "explanation": f"Error computing BLEU: {e}"}

    def rouge_l_score(self, answer: str, ground_truth: str) -> Dict[str, Any]:
        """
        ROUGE-L score based on Longest Common Subsequence (LCS).
        No external dependencies — custom implementation.

        ROUGE-L Precision = LCS / len(answer)
        ROUGE-L Recall    = LCS / len(ground_truth)
        ROUGE-L F1        = 2 × P × R / (P + R)
        """
        if not answer.strip() or not ground_truth.strip():
            return {"score": 0.0, "explanation": "Empty answer or ground truth."}

        answer_tokens = answer.lower().split()
        truth_tokens = ground_truth.lower().split()

        lcs_length = self._lcs_length(answer_tokens, truth_tokens)

        precision = lcs_length / len(answer_tokens) if answer_tokens else 0.0
        recall = lcs_length / len(truth_tokens) if truth_tokens else 0.0

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        return {
            "score": round(f1, 3),
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "lcs_length": lcs_length,
            "explanation": f"ROUGE-L F1={f1:.3f} (P={precision:.3f}, R={recall:.3f}, LCS={lcs_length} tokens)"
        }

    def _lcs_length(self, x: List[str], y: List[str]) -> int:
        """Compute length of Longest Common Subsequence using dynamic programming."""
        m, n = len(x), len(y)
        # Space-optimized: only need two rows
        prev = [0] * (n + 1)
        curr = [0] * (n + 1)

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i - 1] == y[j - 1]:
                    curr[j] = prev[j - 1] + 1
                else:
                    curr[j] = max(prev[j], curr[j - 1])
            prev, curr = curr, [0] * (n + 1)

        return prev[n]

    def faithfulness(self, answer: str, context: str) -> Dict[str, Any]:
        """
        Measure whether the answer is grounded in the retrieved context.
        Uses LLM if available; falls back to heuristic keyword overlap.
        """
        if not answer.strip() or not context.strip():
            return {"score": 0.0, "explanation": "Empty answer or context."}

        if self.llm_client:
            return self._llm_faithfulness(answer, context)

        return self._heuristic_faithfulness(answer, context)

    def _heuristic_faithfulness(self, answer: str, context: str) -> Dict[str, Any]:
        """Heuristic faithfulness: check sentence-level keyword grounding."""
        answer_sentences = self._split_sentences(answer)
        context_lower = context.lower()

        grounded_count = 0
        total = len(answer_sentences)

        for sent in answer_sentences:
            keywords = self._extract_keywords(sent)
            if not keywords:
                continue
            match_count = sum(1 for kw in keywords if kw in context_lower)
            if match_count / len(keywords) > 0.3:
                grounded_count += 1

        score = grounded_count / total if total > 0 else 0.0

        return {
            "score": round(score, 3),
            "grounded_sentences": grounded_count,
            "total_sentences": total,
            "explanation": f"{grounded_count}/{total} answer sentences are grounded in context."
        }

    def _llm_faithfulness(self, answer: str, context: str) -> Dict[str, Any]:
        """LLM-based faithfulness evaluation."""
        prompt = (
            "You are evaluating whether an AI-generated answer is faithful to the provided context.\n"
            "A faithful answer only contains claims that can be verified from the context.\n\n"
            f"Context:\n{context[:3000]}\n\n"
            f"Answer:\n{answer}\n\n"
            "Score the faithfulness from 0.0 (completely hallucinated) to 1.0 (fully grounded).\n"
            "Respond with ONLY a JSON object: {\"score\": <float>, \"explanation\": \"<text>\"}"
        )

        try:
            import json
            response = self.llm_client.generate(prompt, max_tokens=200, temperature=0.1)
            result = json.loads(response)
            return {
                "score": round(float(result.get("score", 0)), 3),
                "explanation": result.get("explanation", "LLM evaluation.")
            }
        except Exception as e:
            # Fallback to heuristic
            return self._heuristic_faithfulness(answer, context)

    def answer_completeness(self, answer: str, ground_truth: str, query: str) -> Dict[str, Any]:
        """
        Evaluate whether the answer covers all key points from the ground truth.
        Uses LLM if available; falls back to keyword coverage.
        """
        if not answer.strip() or not ground_truth.strip():
            return {"score": 0.0, "explanation": "Empty answer or ground truth."}

        if self.llm_client:
            return self._llm_completeness(answer, ground_truth, query)

        return self._heuristic_completeness(answer, ground_truth)

    def _heuristic_completeness(self, answer: str, ground_truth: str) -> Dict[str, Any]:
        """Heuristic completeness: keyword coverage of ground truth."""
        truth_keywords = set(self._extract_keywords(ground_truth))
        answer_keywords = set(self._extract_keywords(answer))

        if not truth_keywords:
            return {"score": 0.0, "explanation": "No keywords in ground truth."}

        covered = truth_keywords.intersection(answer_keywords)
        score = len(covered) / len(truth_keywords)

        return {
            "score": round(min(score, 1.0), 3),
            "covered_keywords": len(covered),
            "total_keywords": len(truth_keywords),
            "explanation": f"{len(covered)}/{len(truth_keywords)} ground truth keywords covered in answer."
        }

    def _llm_completeness(self, answer: str, ground_truth: str, query: str) -> Dict[str, Any]:
        """LLM-based answer completeness evaluation."""
        prompt = (
            "You are evaluating whether an AI-generated answer completely covers the expected answer.\n\n"
            f"Question: {query}\n\n"
            f"Expected Answer:\n{ground_truth}\n\n"
            f"Generated Answer:\n{answer}\n\n"
            "Score the completeness from 0.0 (misses everything) to 1.0 (covers all key points).\n"
            "Respond with ONLY a JSON object: {\"score\": <float>, \"explanation\": \"<text>\"}"
        )

        try:
            import json
            response = self.llm_client.generate(prompt, max_tokens=200, temperature=0.1)
            result = json.loads(response)
            return {
                "score": round(float(result.get("score", 0)), 3),
                "explanation": result.get("explanation", "LLM evaluation.")
            }
        except Exception as e:
            return self._heuristic_completeness(answer, ground_truth)

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from text."""
        stopwords = {
            "the", "a", "an", "is", "was", "are", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "can", "and",
            "or", "but", "in", "on", "at", "to", "for", "of", "with",
            "by", "from", "as", "it", "its", "this", "that", "not",
            "such", "these", "those", "which", "who", "how", "what",
            "when", "where", "than", "also", "each", "more", "other",
        }
        words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        return [w for w in words if w not in stopwords]

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]


# ═══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

class RAGMetrics:
    """
    Orchestrates all evaluation metrics: Retrieval + Generation + Latency.
    """

    def __init__(self, llm_client=None, embedder=None):
        """
        Args:
            llm_client: Optional LLM client for LLM-based evaluation.
            embedder: Optional Embedder instance for semantic similarity.
        """
        self.retrieval_metrics = RetrievalMetrics(llm_client=llm_client)
        self.generation_metrics = GenerationMetrics(llm_client=llm_client, embedder=embedder)

    def evaluate(
        self,
        query: str,
        answer: str,
        context: str,
        ground_truth: str = "",
        relevant_sections: List[str] = None,
        relevant_keywords: List[str] = None,
        retrieved_sections: List[str] = None,
        retrieved_chunk_ids: List[str] = None,
        retrieved_contents: List[str] = None,
        timing: Optional[Dict[str, float]] = None,
        k: int = 5
    ) -> Dict[str, Any]:
        """
        Run all evaluation metrics.

        Args:
            query: The user's question.
            answer: The system-generated answer.
            context: The retrieved context used to generate the answer.
            ground_truth: Expected correct answer.
            relevant_sections: Ground-truth relevant section headings.
            relevant_keywords: Ground-truth relevant keywords.
            retrieved_sections: Section headings of retrieved chunks (ordered).
            retrieved_chunk_ids: IDs of retrieved chunks.
            retrieved_contents: Text content of retrieved chunks.
            timing: Pipeline timing dict.
            k: Top-K for retrieval metrics.

        Returns:
            Dict with all metric scores organized by category.
        """
        results = {"retrieval": {}, "generation": {}, "system": {}}

        # ── Retrieval Metrics ──
        if (retrieved_sections and relevant_sections):
            results["retrieval"] = self.retrieval_metrics.evaluate_all(
                retrieved_sections=retrieved_sections or [],
                retrieved_chunk_ids=retrieved_chunk_ids or [],
                relevant_sections=relevant_sections or [],
                relevant_keywords=relevant_keywords or [],
                retrieved_contents=retrieved_contents or [],
                k=k,
                query=query,
            )

        # ── Generation Metrics ──
        if ground_truth:
            results["generation"] = self.generation_metrics.evaluate_all(
                query=query,
                answer=answer,
                ground_truth=ground_truth,
                context=context
            )
        else:
            # Without ground truth, only faithfulness is available
            results["generation"]["faithfulness"] = self.generation_metrics.faithfulness(
                answer, context
            )

        # ── System Metrics ──
        if timing:
            results["system"]["latency"] = {
                "total_seconds": round(sum(timing.values()), 3),
                "breakdown": timing,
                "explanation": f"Total pipeline time: {sum(timing.values()):.3f}s"
            }

        # ── Overall Scores ──
        all_scores = []
        for category in ["retrieval", "generation"]:
            for metric_name, metric_data in results.get(category, {}).items():
                if isinstance(metric_data, dict) and "score" in metric_data:
                    all_scores.append(metric_data["score"])

        results["overall_score"] = round(
            sum(all_scores) / len(all_scores), 3
        ) if all_scores else 0.0

        return results
```

[RetrievalMetrics](file:///c:/GIT_MINOR/MinorProject_resembler/evaluation/rag_metrics.py#21-317) now accepts `llm_client`. For each retrieved chunk, the LLM judges query-relevance (binary 0/1). Falls back to the existing heuristic (section heading + keyword overlap).

---

## Verification Results

| Test | Result |
|------|--------|
| All module imports | ✅ Pass |
| Sliding window (500 words, window=200, stride=100) | ✅ 4 chunks, correct offsets |
| Chunk content uniqueness (realistic text) | ✅ Pass |

> [!IMPORTANT]
> Full end-to-end testing (ingest → query → evaluate) requires a valid `GROQ_API_KEY` set in `.env`. Run:
> - `python main.py ingest` — verify sliding window chunking
> - `python main.py query "What NLP tasks are used?"` — verify verification step in timing output
> - `python main.py evaluate --all --save` — verify LLM-judged relevance labels
