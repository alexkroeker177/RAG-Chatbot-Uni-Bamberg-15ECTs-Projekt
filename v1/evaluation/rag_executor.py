"""
RAG executor for evaluation pipeline.

Runs questions through RAG system with and without beliefs for baseline comparison.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from v1.core.config import load_config, Config
from v1.core.logger import setup_logger
from v1.generation.conv_bdi import DynamicBeliefs
from v1.generation.rag_chain import RAGChain


logger = setup_logger(__name__)


class RAGExecutor:
    """Executes RAG queries for evaluation."""

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize RAG executor.

        Args:
            config: Optional config object, loads from file if not provided
        """
        self.config = config or load_config()
        self.output_dir = Path(self.config.evaluation.output_directory)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize RAG chain
        logger.info("Initializing RAG chain...")
        self.rag = RAGChain(self.config)
        logger.info("RAG chain initialized")

        # Default beliefs for testing (Master, AI, 2nd semester)
        self.default_beliefs = DynamicBeliefs(
            degree="Master",
            program="ai",
            semester=2
        )

    def _execute_single_query(
        self,
        question: str,
        session_id: str,
        beliefs: Optional[DynamicBeliefs] = None
    ) -> Dict[str, Any]:
        """
        Execute a single RAG query.

        Args:
            question: Question to ask
            session_id: Session ID for tracking
            beliefs: Optional beliefs (None for baseline without beliefs)

        Returns:
            Result dictionary with answer, docs, timing
        """
        start_time = time.time()

        try:
            # Set beliefs in belief_manager before query (if provided)
            if beliefs:
                self.rag.belief_manager.update_beliefs(session_id, beliefs)

            result = self.rag.invoke_with_intent(
                question=question,
                session_id=session_id
            )

            latency_ms = int((time.time() - start_time) * 1000)

            # Extract retrieved documents metadata
            retrieved_docs = []
            for doc in result.get("source_documents", []):
                doc_meta = {
                    "doc": doc.metadata.get("doc", "unknown"),
                    "page": doc.metadata.get("page", "n/a"),
                    "program": doc.metadata.get("program", "general"),
                    "degree": doc.metadata.get("degree", "general"),
                    "section_title": doc.metadata.get("section_title", ""),
                    "snippet": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                }
                retrieved_docs.append(doc_meta)

            return {
                "rag_answer": result.get("answer", ""),
                "retrieved_documents": retrieved_docs,
                "intention": result.get("intention", "unknown"),
                "latency_ms": latency_ms,
                "success": True,
                "error": None
            }

        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            logger.error(f"RAG query failed: {e}")
            return {
                "rag_answer": "",
                "retrieved_documents": [],
                "intention": "error",
                "latency_ms": latency_ms,
                "success": False,
                "error": str(e)
            }

    def execute_questions(
        self,
        questions_path: Optional[Path] = None,
        run_baseline: bool = True
    ) -> Dict[str, Any]:
        """
        Execute all questions from the questions file.

        Args:
            questions_path: Path to evaluation_questions.json
            run_baseline: If True, run both with and without beliefs

        Returns:
            Results dictionary with all executions
        """
        questions_path = questions_path or self.output_dir / "evaluation_questions.json"

        logger.info(f"Loading questions from: {questions_path}")

        with open(questions_path, "r", encoding="utf-8") as f:
            questions_data = json.load(f)

        questions = questions_data.get("questions", [])
        total = len(questions)
        logger.info(f"Loaded {total} questions")

        results = []

        for i, q in enumerate(questions):
            question_id = q["id"]
            question_text = q["question"]
            variation_type = q["variation_type"]

            logger.info(f"[{i+1}/{total}] Processing Q{question_id} ({variation_type}): {question_text[:50]}...")

            # Session IDs for tracking
            session_with_beliefs = f"eval-with-{question_id}"
            session_without_beliefs = f"eval-without-{question_id}"

            result_entry = {
                "question_id": question_id,
                "original_faq_id": q["original_faq_id"],
                "variation_type": variation_type,
                "topic": q["topic"],
                "question": question_text,
                "ground_truth_answer": q["ground_truth_answer"],
                "timestamp": datetime.now().isoformat()
            }

            # Execute WITH beliefs
            logger.debug("  Executing with beliefs...")
            with_beliefs_result = self._execute_single_query(
                question_text,
                session_with_beliefs,
                self.default_beliefs
            )
            result_entry["with_beliefs"] = with_beliefs_result

            # Execute WITHOUT beliefs (baseline)
            if run_baseline:
                logger.debug("  Executing without beliefs (baseline)...")
                without_beliefs_result = self._execute_single_query(
                    question_text,
                    session_without_beliefs,
                    None  # No beliefs
                )
                result_entry["without_beliefs"] = without_beliefs_result

            results.append(result_entry)

            # Small delay between queries
            time.sleep(0.2)

        # Calculate summary stats
        with_beliefs_latencies = [r["with_beliefs"]["latency_ms"] for r in results if r["with_beliefs"]["success"]]
        success_count = sum(1 for r in results if r["with_beliefs"]["success"])

        output_data = {
            "metadata": {
                "executed_at": datetime.now().isoformat(),
                "total_questions": len(results),
                "successful_queries": success_count,
                "failed_queries": len(results) - success_count,
                "run_baseline": run_baseline,
                "beliefs": self.default_beliefs.to_dict(),
                "model": self.config.ollama.generation_model,
                "embedding_model": self.config.ollama.embedding_model,
                "avg_latency_ms": int(sum(with_beliefs_latencies) / len(with_beliefs_latencies)) if with_beliefs_latencies else 0
            },
            "results": results
        }

        return output_data

    def save_results(self, results_data: Dict[str, Any], output_path: Optional[Path] = None) -> Path:
        """
        Save execution results to JSON file.

        Args:
            results_data: Results data dictionary
            output_path: Output file path

        Returns:
            Path to saved file
        """
        output_path = output_path or self.output_dir / "evaluation_results.json"

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved {results_data['metadata']['total_questions']} results to {output_path}")
        return output_path

    def run(
        self,
        questions_path: Optional[Path] = None,
        run_baseline: bool = True
    ) -> Path:
        """
        Run the full RAG execution pipeline.

        Args:
            questions_path: Optional path to questions file
            run_baseline: Run with and without beliefs

        Returns:
            Path to results file
        """
        logger.info("Starting RAG execution...")
        logger.info(f"Run baseline comparison: {run_baseline}")

        results_data = self.execute_questions(questions_path, run_baseline)
        output_path = self.save_results(results_data)

        logger.info(f"RAG execution complete: {output_path}")
        return output_path


def main():
    """Run RAG executor as standalone script."""
    import argparse

    parser = argparse.ArgumentParser(description="Execute RAG queries for evaluation")
    parser.add_argument("--questions", type=str, help="Path to questions JSON file")
    parser.add_argument("--no-baseline", action="store_true", help="Skip baseline (without beliefs)")
    args = parser.parse_args()

    executor = RAGExecutor()
    questions_path = Path(args.questions) if args.questions else None
    output_path = executor.run(
        questions_path=questions_path,
        run_baseline=not args.no_baseline
    )
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
