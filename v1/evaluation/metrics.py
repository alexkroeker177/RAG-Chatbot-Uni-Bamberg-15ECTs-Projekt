"""
Metrics calculation for RAG evaluation.

Computes aggregate metrics from evaluation results.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from statistics import mean, median, stdev

from v1.core.config import load_config, Config
from v1.core.logger import setup_logger


logger = setup_logger(__name__)


class MetricsCalculator:
    """Calculate evaluation metrics from results and scores."""

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize metrics calculator.

        Args:
            config: Optional config object
        """
        self.config = config or load_config()
        self.output_dir = Path(self.config.evaluation.output_directory)

    def load_data(
        self,
        results_path: Optional[Path] = None,
        scores_path: Optional[Path] = None
    ) -> tuple:
        """
        Load results and scores files.

        Returns:
            Tuple of (results_data, scores_data)
        """
        results_path = results_path or self.output_dir / "evaluation_results.json"
        scores_path = scores_path or self.output_dir / "evaluation_scores.json"

        with open(results_path, "r", encoding="utf-8") as f:
            results_data = json.load(f)

        with open(scores_path, "r", encoding="utf-8") as f:
            scores_data = json.load(f)

        return results_data, scores_data

    def calculate_retrieval_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate retrieval quality metrics.

        Args:
            results: List of execution results

        Returns:
            Retrieval metrics dictionary
        """
        metrics = {
            "with_beliefs": {
                "program_match_rate": 0.0,
                "degree_match_rate": 0.0,
                "avg_docs_retrieved": 0.0,
                "empty_retrieval_rate": 0.0
            },
            "without_beliefs": {
                "program_match_rate": 0.0,
                "degree_match_rate": 0.0,
                "avg_docs_retrieved": 0.0,
                "empty_retrieval_rate": 0.0
            }
        }

        for mode in ["with_beliefs", "without_beliefs"]:
            program_matches = 0
            degree_matches = 0
            total_docs = 0
            empty_retrievals = 0
            total = 0

            for r in results:
                if mode not in r:
                    continue

                mode_data = r[mode]
                if not mode_data.get("success", False):
                    continue

                total += 1
                docs = mode_data.get("retrieved_documents", [])

                if not docs:
                    empty_retrievals += 1
                    continue

                total_docs += len(docs)

                # Check if any retrieved doc matches user's program (ai for Master AI)
                for doc in docs:
                    if doc.get("program") == "ai":
                        program_matches += 1
                        break

                # Check if any retrieved doc matches user's degree (Master)
                for doc in docs:
                    if doc.get("degree") == "Master":
                        degree_matches += 1
                        break

            if total > 0:
                metrics[mode]["program_match_rate"] = round(program_matches / total, 3)
                metrics[mode]["degree_match_rate"] = round(degree_matches / total, 3)
                metrics[mode]["avg_docs_retrieved"] = round(total_docs / total, 2)
                metrics[mode]["empty_retrieval_rate"] = round(empty_retrievals / total, 3)

        return metrics

    def calculate_latency_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate latency/performance metrics.

        Args:
            results: List of execution results

        Returns:
            Latency metrics dictionary
        """
        metrics = {}

        for mode in ["with_beliefs", "without_beliefs"]:
            latencies = []

            for r in results:
                if mode not in r:
                    continue

                mode_data = r[mode]
                if mode_data.get("success", False):
                    latencies.append(mode_data.get("latency_ms", 0))

            if latencies:
                metrics[mode] = {
                    "mean_ms": int(mean(latencies)),
                    "median_ms": int(median(latencies)),
                    "min_ms": min(latencies),
                    "max_ms": max(latencies),
                    "std_ms": int(stdev(latencies)) if len(latencies) > 1 else 0,
                    "under_5s": round(sum(1 for l in latencies if l < 5000) / len(latencies), 3),
                    "under_10s": round(sum(1 for l in latencies if l < 10000) / len(latencies), 3)
                }
            else:
                metrics[mode] = {
                    "mean_ms": 0,
                    "median_ms": 0,
                    "min_ms": 0,
                    "max_ms": 0,
                    "std_ms": 0,
                    "under_5s": 0,
                    "under_10s": 0
                }

        return metrics

    def calculate_answer_quality_metrics(self, evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate answer quality metrics from evaluations.

        Args:
            evaluations: List of evaluation entries

        Returns:
            Answer quality metrics
        """
        metrics = {}

        for mode in ["with_beliefs", "without_beliefs"]:
            hallucination_count = 0
            no_answer_count = 0
            perfect_count = 0
            total = 0
            fact_match_counts = []

            for e in evaluations:
                if mode not in e:
                    continue

                eval_data = e[mode]
                total += 1

                score = eval_data.get("score", 0)

                # Count hallucinations
                if eval_data.get("hallucinations"):
                    hallucination_count += 1

                # Count "no answer" (score <= 0.2)
                if score <= 0.2:
                    no_answer_count += 1

                # Count perfect answers (score == 1.0)
                if score == 1.0:
                    perfect_count += 1

                # Count matched facts
                matched = len(eval_data.get("key_facts_matched", []))
                fact_match_counts.append(matched)

            if total > 0:
                metrics[mode] = {
                    "hallucination_rate": round(hallucination_count / total, 3),
                    "no_answer_rate": round(no_answer_count / total, 3),
                    "perfect_answer_rate": round(perfect_count / total, 3),
                    "avg_facts_matched": round(mean(fact_match_counts), 2) if fact_match_counts else 0
                }
            else:
                metrics[mode] = {
                    "hallucination_rate": 0,
                    "no_answer_rate": 0,
                    "perfect_answer_rate": 0,
                    "avg_facts_matched": 0
                }

        return metrics

    def identify_worst_performers(
        self,
        evaluations: List[Dict[str, Any]],
        threshold: float = 0.4,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Identify questions with lowest scores for review.

        Args:
            evaluations: List of evaluation entries
            threshold: Score threshold (below this = worst)
            limit: Maximum entries to return

        Returns:
            List of worst performing questions
        """
        worst = []

        for e in evaluations:
            with_score = e.get("with_beliefs", {}).get("score", 1.0)

            if with_score < threshold:
                worst.append({
                    "question_id": e["question_id"],
                    "variation_type": e["variation_type"],
                    "topic": e["topic"],
                    "score_with_beliefs": with_score,
                    "score_without_beliefs": e.get("without_beliefs", {}).get("score"),
                    "reasoning": e.get("with_beliefs", {}).get("reasoning", "")
                })

        # Sort by score (ascending)
        worst.sort(key=lambda x: x["score_with_beliefs"])

        return worst[:limit]

    def identify_belief_impact(
        self,
        evaluations: List[Dict[str, Any]],
        min_improvement: float = 0.4,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Identify questions where beliefs significantly improved results.

        Args:
            evaluations: List of evaluation entries
            min_improvement: Minimum score improvement to include
            limit: Maximum entries to return

        Returns:
            List of questions with largest belief impact
        """
        improved = []

        for e in evaluations:
            with_score = e.get("with_beliefs", {}).get("score", 0)
            without_score = e.get("without_beliefs", {}).get("score", 0)

            improvement = with_score - without_score

            if improvement >= min_improvement:
                improved.append({
                    "question_id": e["question_id"],
                    "variation_type": e["variation_type"],
                    "topic": e["topic"],
                    "score_with_beliefs": with_score,
                    "score_without_beliefs": without_score,
                    "improvement": round(improvement, 2)
                })

        # Sort by improvement (descending)
        improved.sort(key=lambda x: x["improvement"], reverse=True)

        return improved[:limit]

    def calculate_all_metrics(
        self,
        results_path: Optional[Path] = None,
        scores_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Calculate all metrics from results and scores.

        Returns:
            Complete metrics dictionary
        """
        results_data, scores_data = self.load_data(results_path, scores_path)

        results = results_data.get("results", [])
        evaluations = scores_data.get("evaluations", [])

        metrics = {
            "retrieval": self.calculate_retrieval_metrics(results),
            "latency": self.calculate_latency_metrics(results),
            "answer_quality": self.calculate_answer_quality_metrics(evaluations),
            "worst_performers": self.identify_worst_performers(evaluations),
            "belief_impact": self.identify_belief_impact(evaluations),
            "score_summary": scores_data.get("summary", {})
        }

        return metrics


def main():
    """Run metrics calculator as standalone script."""
    calculator = MetricsCalculator()
    metrics = calculator.calculate_all_metrics()

    print("\n" + "=" * 60)
    print("EVALUATION METRICS")
    print("=" * 60)

    print("\n--- Score Summary ---")
    summary = metrics["score_summary"]
    print(f"With Beliefs:    {summary.get('with_beliefs', {}).get('mean', 0):.2f}")
    print(f"Without Beliefs: {summary.get('without_beliefs', {}).get('mean', 0):.2f}")
    print(f"Improvement:     {summary.get('improvement', 0):+.2f}")

    print("\n--- Retrieval Metrics ---")
    for mode in ["with_beliefs", "without_beliefs"]:
        print(f"\n{mode}:")
        ret = metrics["retrieval"][mode]
        print(f"  Program Match Rate: {ret['program_match_rate']:.1%}")
        print(f"  Degree Match Rate:  {ret['degree_match_rate']:.1%}")

    print("\n--- Latency Metrics ---")
    for mode in ["with_beliefs", "without_beliefs"]:
        print(f"\n{mode}:")
        lat = metrics["latency"][mode]
        print(f"  Mean:   {lat['mean_ms']}ms")
        print(f"  Median: {lat['median_ms']}ms")

    print("\n--- Answer Quality ---")
    for mode in ["with_beliefs", "without_beliefs"]:
        print(f"\n{mode}:")
        qual = metrics["answer_quality"][mode]
        print(f"  Perfect Answer Rate:  {qual['perfect_answer_rate']:.1%}")
        print(f"  Hallucination Rate:   {qual['hallucination_rate']:.1%}")
        print(f"  No Answer Rate:       {qual['no_answer_rate']:.1%}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
