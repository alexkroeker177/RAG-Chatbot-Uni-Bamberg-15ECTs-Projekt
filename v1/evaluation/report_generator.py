"""
Markdown report generator for RAG evaluation.

Generates a comprehensive evaluation report from metrics.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from v1.core.config import load_config, Config
from v1.core.logger import setup_logger
from v1.evaluation.metrics import MetricsCalculator


logger = setup_logger(__name__)


class ReportGenerator:
    """Generate markdown evaluation reports."""

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize report generator.

        Args:
            config: Optional config object
        """
        self.config = config or load_config()
        self.output_dir = Path(self.config.evaluation.output_directory)
        self.metrics_calculator = MetricsCalculator(self.config)

    def _format_percentage(self, value: float) -> str:
        """Format a decimal as percentage."""
        return f"{value * 100:.1f}%"

    def _format_score(self, value: float) -> str:
        """Format a score value."""
        return f"{value:.2f}"

    def generate_report(
        self,
        results_path: Optional[Path] = None,
        scores_path: Optional[Path] = None
    ) -> str:
        """
        Generate markdown evaluation report.

        Returns:
            Markdown report string
        """
        # Load all data
        results_data, scores_data = self.metrics_calculator.load_data(results_path, scores_path)
        metrics = self.metrics_calculator.calculate_all_metrics(results_path, scores_path)

        results_meta = results_data.get("metadata", {})
        scores_meta = scores_data.get("metadata", {})
        summary = metrics["score_summary"]

        # Build report
        lines = []

        # Header
        lines.append("# RAG Evaluation Report")
        lines.append("")
        lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # Executive Summary
        lines.append("## Executive Summary")
        lines.append("")

        with_mean = summary.get("with_beliefs", {}).get("mean", 0)
        without_mean = summary.get("without_beliefs", {}).get("mean", 0)
        improvement = summary.get("improvement", 0)

        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Total Questions Evaluated | {scores_meta.get('total_evaluated', 0)} |")
        lines.append(f"| Mean Score (with beliefs) | **{self._format_score(with_mean)}** |")
        lines.append(f"| Mean Score (without beliefs) | {self._format_score(without_mean)} |")
        lines.append(f"| Improvement from Beliefs | **{improvement:+.2f}** ({self._format_percentage(improvement / without_mean if without_mean > 0 else 0)} relative) |")
        lines.append("")

        # Configuration
        lines.append("## Configuration")
        lines.append("")
        lines.append(f"- **RAG Model:** {results_meta.get('model', 'N/A')}")
        lines.append(f"- **Embedding Model:** {results_meta.get('embedding_model', 'N/A')}")
        lines.append(f"- **Judge Model:** {scores_meta.get('judge_model', 'N/A')}")
        lines.append(f"- **Default Beliefs:** {results_meta.get('beliefs', {})}")
        lines.append("")

        # Score Distribution
        lines.append("## Score Distribution")
        lines.append("")

        dist_with = summary.get("score_distribution", {}).get("with_beliefs", {})
        dist_without = summary.get("score_distribution", {}).get("without_beliefs", {})

        lines.append("| Score | With Beliefs | Without Beliefs |")
        lines.append("|-------|--------------|-----------------|")
        for score in ["1.0", "0.8", "0.6", "0.4", "0.2", "0.0"]:
            with_count = dist_with.get(score, 0)
            without_count = dist_without.get(score, 0)
            lines.append(f"| {score} | {with_count} | {without_count} |")
        lines.append("")

        # Performance by Variation Type
        lines.append("## Performance by Variation Type")
        lines.append("")
        lines.append("This shows how well the RAG system handles different question phrasings.")
        lines.append("")

        by_var = summary.get("by_variation_type", {})
        lines.append("| Variation Type | With Beliefs | Without Beliefs | Δ |")
        lines.append("|----------------|--------------|-----------------|---|")

        for vtype in ["original", "formal", "informal", "keyword", "verbose", "synonym", "inverse"]:
            if vtype in by_var:
                with_score = by_var[vtype].get("with_beliefs", {}).get("mean", 0)
                without_score = by_var[vtype].get("without_beliefs", {}).get("mean", 0)
                delta = with_score - without_score
                lines.append(f"| {vtype} | {self._format_score(with_score)} | {self._format_score(without_score)} | {delta:+.2f} |")
        lines.append("")

        # Performance by Topic
        lines.append("## Performance by Topic")
        lines.append("")

        by_topic = summary.get("by_topic", {})
        lines.append("| Topic | With Beliefs | Count |")
        lines.append("|-------|--------------|-------|")

        # Sort by score descending
        sorted_topics = sorted(
            by_topic.items(),
            key=lambda x: x[1].get("with_beliefs", {}).get("mean", 0),
            reverse=True
        )

        for topic, data in sorted_topics[:15]:  # Limit to top 15
            with_score = data.get("with_beliefs", {}).get("mean", 0)
            count = data.get("with_beliefs", {}).get("count", 0)
            lines.append(f"| {topic[:40]} | {self._format_score(with_score)} | {count} |")
        lines.append("")

        # Retrieval Metrics
        lines.append("## Retrieval Quality")
        lines.append("")

        ret_with = metrics["retrieval"]["with_beliefs"]
        ret_without = metrics["retrieval"]["without_beliefs"]

        lines.append("| Metric | With Beliefs | Without Beliefs |")
        lines.append("|--------|--------------|-----------------|")
        lines.append(f"| Program Match Rate | {self._format_percentage(ret_with['program_match_rate'])} | {self._format_percentage(ret_without['program_match_rate'])} |")
        lines.append(f"| Degree Match Rate | {self._format_percentage(ret_with['degree_match_rate'])} | {self._format_percentage(ret_without['degree_match_rate'])} |")
        lines.append(f"| Avg Docs Retrieved | {ret_with['avg_docs_retrieved']} | {ret_without['avg_docs_retrieved']} |")
        lines.append(f"| Empty Retrieval Rate | {self._format_percentage(ret_with['empty_retrieval_rate'])} | {self._format_percentage(ret_without['empty_retrieval_rate'])} |")
        lines.append("")

        # Latency Metrics
        lines.append("## Latency Performance")
        lines.append("")

        lat_with = metrics["latency"]["with_beliefs"]
        lat_without = metrics["latency"]["without_beliefs"]

        lines.append("| Metric | With Beliefs | Without Beliefs |")
        lines.append("|--------|--------------|-----------------|")
        lines.append(f"| Mean Latency | {lat_with['mean_ms']}ms | {lat_without['mean_ms']}ms |")
        lines.append(f"| Median Latency | {lat_with['median_ms']}ms | {lat_without['median_ms']}ms |")
        lines.append(f"| Min Latency | {lat_with['min_ms']}ms | {lat_without['min_ms']}ms |")
        lines.append(f"| Max Latency | {lat_with['max_ms']}ms | {lat_without['max_ms']}ms |")
        lines.append(f"| Under 5s | {self._format_percentage(lat_with['under_5s'])} | {self._format_percentage(lat_without['under_5s'])} |")
        lines.append(f"| Under 10s | {self._format_percentage(lat_with['under_10s'])} | {self._format_percentage(lat_without['under_10s'])} |")
        lines.append("")

        # Answer Quality
        lines.append("## Answer Quality")
        lines.append("")

        qual_with = metrics["answer_quality"]["with_beliefs"]
        qual_without = metrics["answer_quality"]["without_beliefs"]

        lines.append("| Metric | With Beliefs | Without Beliefs |")
        lines.append("|--------|--------------|-----------------|")
        lines.append(f"| Perfect Answer Rate | {self._format_percentage(qual_with['perfect_answer_rate'])} | {self._format_percentage(qual_without['perfect_answer_rate'])} |")
        lines.append(f"| Hallucination Rate | {self._format_percentage(qual_with['hallucination_rate'])} | {self._format_percentage(qual_without['hallucination_rate'])} |")
        lines.append(f"| No Answer Rate | {self._format_percentage(qual_with['no_answer_rate'])} | {self._format_percentage(qual_without['no_answer_rate'])} |")
        lines.append(f"| Avg Facts Matched | {qual_with['avg_facts_matched']} | {qual_without['avg_facts_matched']} |")
        lines.append("")

        # Worst Performers
        lines.append("## Worst Performers (Score < 0.4)")
        lines.append("")
        lines.append("These questions need manual review and potential system improvements.")
        lines.append("")

        worst = metrics["worst_performers"]
        if worst:
            lines.append("| Q ID | Type | Topic | Score | Reasoning |")
            lines.append("|------|------|-------|-------|-----------|")
            for w in worst[:10]:
                topic = w["topic"][:25] + "..." if len(w["topic"]) > 25 else w["topic"]
                reasoning = w["reasoning"][:40] + "..." if len(w["reasoning"]) > 40 else w["reasoning"]
                lines.append(f"| {w['question_id']} | {w['variation_type']} | {topic} | {w['score_with_beliefs']} | {reasoning} |")
        else:
            lines.append("*No questions scored below 0.4*")
        lines.append("")

        # Belief Impact
        lines.append("## Belief Impact Analysis")
        lines.append("")
        lines.append("Questions where beliefs significantly improved results (Δ ≥ 0.4).")
        lines.append("")

        improved = metrics["belief_impact"]
        if improved:
            lines.append("| Q ID | Type | Topic | With | Without | Δ |")
            lines.append("|------|------|-------|------|---------|---|")
            for imp in improved[:10]:
                topic = imp["topic"][:25] + "..." if len(imp["topic"]) > 25 else imp["topic"]
                lines.append(f"| {imp['question_id']} | {imp['variation_type']} | {topic} | {imp['score_with_beliefs']} | {imp['score_without_beliefs']} | +{imp['improvement']} |")
        else:
            lines.append("*No questions showed improvement ≥ 0.4 from beliefs*")
        lines.append("")

        # Conclusion
        lines.append("## Conclusion")
        lines.append("")

        if improvement > 0.1:
            lines.append(f"The belief-aware retrieval system shows a **{self._format_percentage(improvement / without_mean if without_mean > 0 else 0)} improvement** over the baseline. ")
        elif improvement > 0:
            lines.append(f"The belief-aware retrieval system shows a modest improvement of {improvement:+.2f} over the baseline. ")
        else:
            lines.append("The belief-aware retrieval system shows no significant improvement over baseline. Consider reviewing the retrieval strategy. ")

        # Key insights
        lines.append("")
        lines.append("### Key Insights")
        lines.append("")

        # Variation insights
        var_scores = [(v, by_var[v].get("with_beliefs", {}).get("mean", 0)) for v in by_var]
        var_scores.sort(key=lambda x: x[1])

        worst_var = var_scores[0][0] if var_scores else "N/A"
        best_var = var_scores[-1][0] if var_scores else "N/A"

        lines.append(f"- **Best performing variation type:** {best_var}")
        lines.append(f"- **Worst performing variation type:** {worst_var} (may need improved semantic matching)")
        lines.append(f"- **Perfect answer rate:** {self._format_percentage(qual_with['perfect_answer_rate'])}")
        lines.append(f"- **Hallucination rate:** {self._format_percentage(qual_with['hallucination_rate'])}")
        lines.append("")

        lines.append("---")
        lines.append(f"*Report generated by RAG Evaluation Pipeline*")

        return "\n".join(lines)

    def save_report(self, report: str, output_path: Optional[Path] = None) -> Path:
        """
        Save report to markdown file.

        Args:
            report: Report markdown string
            output_path: Output file path

        Returns:
            Path to saved file
        """
        output_path = output_path or self.output_dir / "evaluation_report.md"

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)

        logger.info(f"Saved report to {output_path}")
        return output_path

    def run(
        self,
        results_path: Optional[Path] = None,
        scores_path: Optional[Path] = None
    ) -> Path:
        """
        Generate and save the evaluation report.

        Returns:
            Path to saved report
        """
        logger.info("Generating evaluation report...")

        report = self.generate_report(results_path, scores_path)
        output_path = self.save_report(report)

        logger.info(f"Report generation complete: {output_path}")
        return output_path


def main():
    """Run report generator as standalone script."""
    generator = ReportGenerator()
    output_path = generator.run()
    print(f"Report saved to: {output_path}")


if __name__ == "__main__":
    main()
