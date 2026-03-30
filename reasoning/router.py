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
        """
        timing = {}
        output = {"query": query}

        # Step 1: Planning (now with available_papers awareness)
        t0 = time.time()
        plan = self.planner.plan(query, available_papers)
        timing["planning"] = round(time.time() - t0, 3)
        output["plan"] = plan

        print(f"[Router] Query type: {plan['query_type']}")
        
        # --- Handle Chitchat (Bypass RAG) ---
        if plan["query_type"] == "chitchat":
            t0 = time.time()
            # Feed a special chitchat "summary" to the explanation agent
            chitchat_content = {
                "summary": "This is a general conversation or greeting. No research context needed.",
                "original_context": "",
                "source_count": 0,
                "is_chitchat": True
            }
            explanation_output = self.explanation_agent.run(chitchat_content, query=query)
            timing["explanation"] = round(time.time() - t0, 3)
            
            output["answer"] = explanation_output["answer"]
            output["confidence"] = "high"
            output["timing"] = timing
            return output

        # --- Handle Ambiguity (Stop and ask user) ---
        if plan.get("is_ambiguous"):
            output["needs_clarification"] = True
            output["available_papers"] = available_papers
            output["answer"] = "I found multiple papers. Which one should I analyze?"
            output["timing"] = timing
            return output

        # --- Standard RAG Flow ---
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
            output["summary"] = {
                "query": query,
                "summary": retrieval_output.get("context", ""),
                "structured": {"key_claims": [], "methodologies": [], "limitations": [], "source_citations": []},
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

            if not verification_output["verified"] and verification_output.get("corrected_answer"):
                output["answer"] = verification_output["corrected_answer"]
                output["confidence"] = verification_output.get("confidence", "low")
            else:
                output["answer"] = explanation_output["answer"]
                output["confidence"] = verification_output.get(
                    "confidence", explanation_output.get("confidence", "unknown")
                )
        else:
            output["answer"] = explanation_output["answer"]
            output["confidence"] = explanation_output.get("confidence", "unknown")

        output["timing"] = timing
        return output
