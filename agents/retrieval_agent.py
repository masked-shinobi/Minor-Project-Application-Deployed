"""
Retrieval Agent Module
Interfaces with the retrieval layer to find relevant chunks for a given query.
"""

from typing import List, Optional


class RetrievalAgent:
    """
    Agent responsible for finding relevant text chunks from the knowledge base.
    Acts as the first step in the multi-agent pipeline.
    """

    def __init__(self, retriever):
        """
        Args:
            retriever: An instance of retrieval.retriever.Retriever.
        """
        self.retriever = retriever

    def run(
        self,
        query: str,
        top_k: int = 5,
        paper_id: Optional[str] = None,
        search_mode: str = "hybrid"
    ) -> dict:
        """
        Execute the retrieval agent.

        Args:
            query: The user's question or search query.
            top_k: Number of chunks to retrieve.
            paper_id: Optional filter to a specific paper.
            search_mode: "hybrid" (default), "semantic", or "keyword".

        Returns:
            Dict with keys:
              - "query": original query
              - "results": list of RetrievalResult objects
              - "context": combined context text for downstream agents
              - "num_results": count of results
        """
        if search_mode == "semantic":
            results = self.retriever.retrieve_semantic(query, top_k=top_k)
        else:
            results = self.retriever.retrieve(
                query, top_k=top_k, paper_id=paper_id
            )

        # Build combined context string
        context_parts = []
        for i, result in enumerate(results):
            source = f"[Source: {result.paper_title or result.paper_id} — {result.section_heading}]"
            context_parts.append(f"{source}\n{result.content}")

        context = "\n\n---\n\n".join(context_parts)

        return {
            "query": query,
            "results": results,
            "context": context,
            "num_results": len(results)
        }
