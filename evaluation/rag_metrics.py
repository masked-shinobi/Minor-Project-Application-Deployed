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
