"""
Evaluation Runner Module
Runs the full evaluation pipeline against a test dataset.
Evaluates all 13 metrics (7 retrieval + 5 generation + 1 latency) per query,
prints formatted results, and optionally saves to JSON.
"""

import os
import json
import time
from typing import Dict, Any, List, Optional

from evaluation.rag_metrics import RAGMetrics


class EvaluationRunner:
    """
    Runs comprehensive evaluation of the RAG system against a ground-truth
    test dataset. Orchestrates the query pipeline and metrics computation.
    """

    DATASET_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "test_dataset.json"
    )

    def __init__(self, router, embedder=None, llm_client=None):
        """
        Args:
            router: The Router instance for running queries.
            embedder: The Embedder instance for semantic similarity.
            llm_client: The LLMClient instance for LLM-based metrics.
        """
        self.router = router
        self.metrics = RAGMetrics(llm_client=llm_client, embedder=embedder)

    def load_dataset(self, dataset_path: Optional[str] = None) -> List[Dict]:
        """Load the test dataset from JSON."""
        path = dataset_path or self.DATASET_PATH
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Test dataset not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def run_single(self, query: str, ground_truth: str = "",
                   relevant_sections: List[str] = None,
                   relevant_keywords: List[str] = None) -> Dict[str, Any]:
        """
        Run evaluation for a single query.

        Args:
            query: The question to evaluate.
            ground_truth: Expected answer (optional).
            relevant_sections: Expected relevant section headings.
            relevant_keywords: Expected relevant keywords.

        Returns:
            Dict with query info, pipeline output, and all metric scores.
        """
        # Run the query through the pipeline
        pipeline_output = self.router.route(query)

        # Extract retrieval details
        retrieval_data = pipeline_output.get("retrieval", {})
        results_list = retrieval_data.get("results", [])

        retrieved_sections = [
            getattr(r, "section_heading", "") for r in results_list
        ]
        retrieved_chunk_ids = [
            getattr(r, "chunk_id", "") for r in results_list
        ]
        retrieved_contents = [
            getattr(r, "content", "") for r in results_list
        ]

        # Run all metrics
        eval_result = self.metrics.evaluate(
            query=query,
            answer=pipeline_output.get("answer", ""),
            context=retrieval_data.get("context", ""),
            ground_truth=ground_truth,
            relevant_sections=relevant_sections or [],
            relevant_keywords=relevant_keywords or [],
            retrieved_sections=retrieved_sections,
            retrieved_chunk_ids=retrieved_chunk_ids,
            retrieved_contents=retrieved_contents,
            timing=pipeline_output.get("timing"),
            k=5
        )

        return {
            "query": query,
            "ground_truth": ground_truth,
            "answer": pipeline_output.get("answer", ""),
            "confidence": pipeline_output.get("confidence", "unknown"),
            "metrics": eval_result,
            "timing": pipeline_output.get("timing", {})
        }

    def run_all(self, dataset_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Run evaluation for all queries in the test dataset.

        Returns:
            Dict with per-query results and aggregate summary.
        """
        dataset = self.load_dataset(dataset_path)
        per_query_results = []

        print(f"\n{'=' * 70}")
        print(f"  [*] Running Evaluation on {len(dataset)} Queries")
        print(f"{'=' * 70}")

        for i, entry in enumerate(dataset):
            query = entry["query"]
            print(f"\n  [{i+1}/{len(dataset)}] {query}")

            result = self.run_single(
                query=query,
                ground_truth=entry.get("ground_truth", ""),
                relevant_sections=entry.get("relevant_sections", []),
                relevant_keywords=entry.get("relevant_keywords", [])
            )
            per_query_results.append(result)

            # Print summary for this query
            metrics = result["metrics"]
            overall = metrics.get("overall_score", 0)
            confidence = result['confidence']
            print(f"         Overall: {overall:.3f} | Confidence: {confidence}")

        # Compute aggregates
        aggregate = self._compute_aggregates(per_query_results)

        # Print summary
        self._print_summary(per_query_results, aggregate)

        return {
            "per_query": per_query_results,
            "aggregate": aggregate,
            "dataset_size": len(dataset)
        }

    def _compute_aggregates(self, results: List[Dict]) -> Dict[str, Any]:
        """Compute aggregate metrics across all queries."""
        # Collect all metric scores by name
        metric_scores: Dict[str, List[float]] = {}

        for result in results:
            metrics = result["metrics"]
            for category in ["retrieval", "generation"]:
                for metric_name, metric_data in metrics.get(category, {}).items():
                    if isinstance(metric_data, dict) and "score" in metric_data:
                        key = f"{category}/{metric_name}"
                        if key not in metric_scores:
                            metric_scores[key] = []
                        metric_scores[key].append(metric_data["score"])

        # Compute mean, min, max for each metric
        aggregate = {}
        for key, scores in metric_scores.items():
            if scores:
                aggregate[key] = {
                    "mean": round(sum(scores) / len(scores), 3),
                    "min": round(min(scores), 3),
                    "max": round(max(scores), 3),
                    "count": len(scores)
                }

        # Overall aggregate
        all_means = [v["mean"] for v in aggregate.values()]
        aggregate["overall_mean"] = round(
            sum(all_means) / len(all_means), 3
        ) if all_means else 0.0

        return aggregate

    def _print_summary(self, results: List[Dict], aggregate: Dict) -> None:
        """Print a formatted summary table."""
        print(f"\n{'=' * 70}")
        print(f"  [*] EVALUATION SUMMARY ({len(results)} queries)")
        print(f"{'=' * 70}")

        # Retrieval metrics
        print(f"\n  -- Retrieval Quality --")
        retrieval_keys = [k for k in aggregate if k.startswith("retrieval/")]
        for key in sorted(retrieval_keys):
            name = key.split("/")[1]
            data = aggregate[key]
            bar = self._score_bar(data["mean"])
            print(f"    {name:20s}  {bar}  mean={data['mean']:.3f}  "
                  f"[{data['min']:.3f} - {data['max']:.3f}]")

        # Generation metrics
        print(f"\n  -- Generation Quality --")
        gen_keys = [k for k in aggregate if k.startswith("generation/")]
        for key in sorted(gen_keys):
            name = key.split("/")[1]
            data = aggregate[key]
            bar = self._score_bar(data["mean"])
            print(f"    {name:20s}  {bar}  mean={data['mean']:.3f}  "
                  f"[{data['min']:.3f} - {data['max']:.3f}]")

        # Overall
        overall = aggregate.get("overall_mean", 0)
        bar = self._score_bar(overall)
        print(f"\n  -- Overall --")
        print(f"    {'OVERALL SCORE':20s}  {bar}  {overall:.3f}")
        print(f"{'=' * 70}")

    def _score_bar(self, score: float, width: int = 15) -> str:
        """Generate a visual score bar like: [########-------]"""
        filled = int(score * width)
        empty = width - filled
        return f"[{'#' * filled}{'-' * empty}]"

    def save_results(self, results: Dict, output_path: Optional[str] = None) -> str:
        """Save evaluation results to JSON."""
        if output_path is None:
            output_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "data", "evaluation_results.json"
            )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Make results JSON-serializable (strip non-serializable objects)
        clean = self._make_serializable(results)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(clean, f, indent=2, ensure_ascii=False)

        print(f"\n  [+] Results saved to: {output_path}")
        return output_path

    def _make_serializable(self, obj):
        """Recursively make an object JSON-serializable."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)
