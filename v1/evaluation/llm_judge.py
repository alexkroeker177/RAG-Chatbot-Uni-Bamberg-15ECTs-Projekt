"""
LLM-as-Judge evaluation module.

Uses Ollama to score RAG answers against ground truth.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import requests

from v1.core.config import load_config, Config
from v1.core.logger import setup_logger


logger = setup_logger(__name__)


JUDGE_PROMPT = """Du bist ein Evaluator für ein RAG-System zu Prüfungsordnungen der Universität Bamberg.

FRAGE: {question}

REFERENZ-ANTWORT (Ground Truth aus der offiziellen FAQ):
{ground_truth}

RAG-ANTWORT (zu bewerten):
{rag_answer}

Bewerte die RAG-Antwort auf einer Skala von 0.0 bis 1.0 (NUR in 0.2 Schritten: 0.0, 0.2, 0.4, 0.6, 0.8, 1.0):

- 1.0: Vollständig korrekt, alle wichtigen Informationen enthalten
- 0.8: Korrekt mit kleinen Auslassungen (unwichtige Details fehlen)
- 0.6: Teilweise korrekt, Kerninfo vorhanden aber unvollständig
- 0.4: Einige korrekte Infos, aber große Lücken oder Ungenauigkeiten
- 0.2: Überwiegend falsch, irrelevant, oder "ich weiß nicht" Antwort
- 0.0: Komplett falsch, widerspricht der Referenz, oder keine Antwort

WICHTIG:
- Bewerte semantische Korrektheit, nicht wörtliche Übereinstimmung
- Eine kürzere aber korrekte Antwort ist besser als eine lange falsche
- "Ich weiß nicht" oder Verweigerung = 0.2 (nicht 0.0, da ehrlich)
- Halluzinationen (erfundene Fakten) = 0.0

Antworte NUR mit einem validen JSON-Objekt:
{{"score": <0.0|0.2|0.4|0.6|0.8|1.0>, "reasoning": "Kurze Begründung (1-2 Sätze)", "key_facts_matched": ["Fakt 1", "Fakt 2"], "key_facts_missed": ["Fehlender Fakt falls vorhanden"], "hallucinations": ["Erfundene Fakten falls vorhanden"]}}"""


class LLMJudge:
    """LLM-as-Judge for evaluating RAG answers using Ollama."""

    VALID_SCORES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize LLM judge.

        Args:
            config: Optional config object, loads from file if not provided
        """
        self.config = config or load_config()
        self.output_dir = Path(self.config.evaluation.output_directory)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.base_url = self.config.ollama.base_url
        self.judge_model = self.config.evaluation.judge_model

        logger.info(f"Initializing judge with Ollama: {self.judge_model}")

    def _call_ollama(self, prompt: str) -> str:
        """
        Call Ollama API with a prompt.

        Args:
            prompt: The prompt to send

        Returns:
            Response text
        """
        payload = {
            "model": self.judge_model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 512
            }
        }

        response = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=120
        )

        if response.status_code != 200:
            raise RuntimeError(f"Ollama API error: {response.status_code} - {response.text}")

        result = response.json()
        return result["message"]["content"]

    def _parse_judge_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse JSON response from judge LLM.

        Args:
            response_text: Raw LLM response

        Returns:
            Parsed evaluation dictionary
        """
        text = response_text.strip()

        # Handle markdown code blocks
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        # Find JSON object
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            text = text[start:end]

        try:
            result = json.loads(text)

            # Validate and clamp score
            score = result.get("score", 0.0)
            if isinstance(score, str):
                score = float(score)

            # Round to nearest valid score
            score = min(self.VALID_SCORES, key=lambda x: abs(x - score))
            result["score"] = score

            # Ensure required fields
            result.setdefault("reasoning", "No reasoning provided")
            result.setdefault("key_facts_matched", [])
            result.setdefault("key_facts_missed", [])
            result.setdefault("hallucinations", [])

            return result

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse judge response: {e}")
            logger.debug(f"Raw response: {response_text}")
            return {
                "score": 0.4,  # Default uncertain score
                "reasoning": "Failed to parse judge response",
                "key_facts_matched": [],
                "key_facts_missed": [],
                "hallucinations": [],
                "parse_error": str(e)
            }

    def evaluate_single(
        self,
        question: str,
        ground_truth: str,
        rag_answer: str
    ) -> Dict[str, Any]:
        """
        Evaluate a single RAG answer.

        Args:
            question: Original question
            ground_truth: Reference answer from FAQ
            rag_answer: RAG system's answer

        Returns:
            Evaluation result with score and reasoning
        """
        # Handle empty answers
        if not rag_answer or rag_answer.strip() == "":
            return {
                "score": 0.0,
                "reasoning": "Keine Antwort vom RAG-System",
                "key_facts_matched": [],
                "key_facts_missed": ["Alle Fakten"],
                "hallucinations": []
            }

        prompt = JUDGE_PROMPT.format(
            question=question,
            ground_truth=ground_truth,
            rag_answer=rag_answer
        )

        try:
            response = self._call_ollama(prompt)
            return self._parse_judge_response(response)

        except Exception as e:
            logger.error(f"Judge evaluation failed: {e}")
            return {
                "score": 0.4,
                "reasoning": f"Evaluation failed: {e}",
                "key_facts_matched": [],
                "key_facts_missed": [],
                "hallucinations": [],
                "error": str(e)
            }

    def evaluate_results(
        self,
        results_path: Optional[Path] = None,
        evaluate_baseline: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate all results from the RAG execution.

        Args:
            results_path: Path to evaluation_results.json
            evaluate_baseline: Also evaluate without_beliefs results

        Returns:
            Evaluation scores dictionary
        """
        results_path = results_path or self.output_dir / "evaluation_results.json"

        logger.info(f"Loading results from: {results_path}")

        with open(results_path, "r", encoding="utf-8") as f:
            results_data = json.load(f)

        results = results_data.get("results", [])
        total = len(results)
        logger.info(f"Loaded {total} results to evaluate")

        evaluations = []

        for i, r in enumerate(results):
            question_id = r["question_id"]
            question = r["question"]
            ground_truth = r["ground_truth_answer"]

            logger.info(f"[{i+1}/{total}] Evaluating Q{question_id}...")

            eval_entry = {
                "question_id": question_id,
                "original_faq_id": r["original_faq_id"],
                "variation_type": r["variation_type"],
                "topic": r["topic"]
            }

            # Evaluate with_beliefs answer
            with_beliefs = r.get("with_beliefs", {})
            if with_beliefs.get("success", False):
                logger.debug("  Evaluating with_beliefs answer...")
                eval_entry["with_beliefs"] = self.evaluate_single(
                    question,
                    ground_truth,
                    with_beliefs.get("rag_answer", "")
                )
            else:
                eval_entry["with_beliefs"] = {
                    "score": 0.0,
                    "reasoning": "RAG query failed",
                    "key_facts_matched": [],
                    "key_facts_missed": [],
                    "hallucinations": []
                }

            # Evaluate without_beliefs answer (baseline)
            if evaluate_baseline and "without_beliefs" in r:
                without_beliefs = r.get("without_beliefs", {})
                if without_beliefs.get("success", False):
                    logger.debug("  Evaluating without_beliefs answer...")
                    eval_entry["without_beliefs"] = self.evaluate_single(
                        question,
                        ground_truth,
                        without_beliefs.get("rag_answer", "")
                    )
                else:
                    eval_entry["without_beliefs"] = {
                        "score": 0.0,
                        "reasoning": "RAG query failed",
                        "key_facts_matched": [],
                        "key_facts_missed": [],
                        "hallucinations": []
                    }

            evaluations.append(eval_entry)

            # Small delay
            time.sleep(0.1)

        # Calculate summary statistics
        with_scores = [e["with_beliefs"]["score"] for e in evaluations]
        without_scores = [e.get("without_beliefs", {}).get("score") for e in evaluations if "without_beliefs" in e]
        without_scores = [s for s in without_scores if s is not None]

        def calc_stats(scores: List[float]) -> Dict[str, Any]:
            if not scores:
                return {"mean": 0, "count": 0}
            return {
                "mean": round(sum(scores) / len(scores), 3),
                "count": len(scores),
                "min": min(scores),
                "max": max(scores)
            }

        def count_distribution(scores: List[float]) -> Dict[str, int]:
            dist = {str(s): 0 for s in self.VALID_SCORES}
            for s in scores:
                dist[str(s)] = dist.get(str(s), 0) + 1
            return dist

        # Group by variation type
        by_variation = {}
        for e in evaluations:
            vtype = e["variation_type"]
            if vtype not in by_variation:
                by_variation[vtype] = {"with_beliefs": [], "without_beliefs": []}
            by_variation[vtype]["with_beliefs"].append(e["with_beliefs"]["score"])
            if "without_beliefs" in e:
                by_variation[vtype]["without_beliefs"].append(e["without_beliefs"]["score"])

        variation_summary = {}
        for vtype, scores_dict in by_variation.items():
            variation_summary[vtype] = {
                "with_beliefs": calc_stats(scores_dict["with_beliefs"]),
                "without_beliefs": calc_stats(scores_dict["without_beliefs"])
            }

        # Group by topic
        by_topic = {}
        for e in evaluations:
            topic = e["topic"]
            if topic not in by_topic:
                by_topic[topic] = {"with_beliefs": [], "without_beliefs": []}
            by_topic[topic]["with_beliefs"].append(e["with_beliefs"]["score"])
            if "without_beliefs" in e:
                by_topic[topic]["without_beliefs"].append(e["without_beliefs"]["score"])

        topic_summary = {}
        for topic, scores_dict in by_topic.items():
            topic_summary[topic] = {
                "with_beliefs": calc_stats(scores_dict["with_beliefs"]),
                "without_beliefs": calc_stats(scores_dict["without_beliefs"])
            }

        output_data = {
            "metadata": {
                "evaluated_at": datetime.now().isoformat(),
                "judge_model": self.judge_model,
                "total_evaluated": len(evaluations),
                "evaluate_baseline": evaluate_baseline
            },
            "summary": {
                "with_beliefs": calc_stats(with_scores),
                "without_beliefs": calc_stats(without_scores),
                "improvement": round(calc_stats(with_scores)["mean"] - calc_stats(without_scores)["mean"], 3) if without_scores else None,
                "score_distribution": {
                    "with_beliefs": count_distribution(with_scores),
                    "without_beliefs": count_distribution(without_scores)
                },
                "by_variation_type": variation_summary,
                "by_topic": topic_summary
            },
            "evaluations": evaluations
        }

        return output_data

    def save_scores(self, scores_data: Dict[str, Any], output_path: Optional[Path] = None) -> Path:
        """
        Save evaluation scores to JSON file.

        Args:
            scores_data: Scores data dictionary
            output_path: Output file path

        Returns:
            Path to saved file
        """
        output_path = output_path or self.output_dir / "evaluation_scores.json"

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(scores_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved evaluation scores to {output_path}")
        return output_path

    def run(
        self,
        results_path: Optional[Path] = None,
        evaluate_baseline: bool = True
    ) -> Path:
        """
        Run the full LLM judge evaluation.

        Args:
            results_path: Optional path to results file
            evaluate_baseline: Evaluate baseline results

        Returns:
            Path to scores file
        """
        logger.info("Starting LLM-as-Judge evaluation...")

        scores_data = self.evaluate_results(results_path, evaluate_baseline)
        output_path = self.save_scores(scores_data)

        # Print summary
        summary = scores_data["summary"]
        print("\n" + "=" * 50)
        print("EVALUATION SUMMARY")
        print("=" * 50)
        print(f"With Beliefs:    Mean Score = {summary['with_beliefs']['mean']:.2f}")
        if summary.get("without_beliefs", {}).get("count", 0) > 0:
            print(f"Without Beliefs: Mean Score = {summary['without_beliefs']['mean']:.2f}")
            print(f"Improvement:     {summary['improvement']:+.2f}")
        print("=" * 50 + "\n")

        logger.info(f"LLM judge evaluation complete: {output_path}")
        return output_path


def main():
    """Run LLM judge as standalone script."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate RAG answers with LLM judge")
    parser.add_argument("--results", type=str, help="Path to results JSON file")
    parser.add_argument("--no-baseline", action="store_true", help="Skip baseline evaluation")
    args = parser.parse_args()

    judge = LLMJudge()
    results_path = Path(args.results) if args.results else None
    output_path = judge.run(
        results_path=results_path,
        evaluate_baseline=not args.no_baseline
    )
    print(f"Scores saved to: {output_path}")


if __name__ == "__main__":
    main()
