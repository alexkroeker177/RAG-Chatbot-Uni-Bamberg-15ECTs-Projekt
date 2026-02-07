"""
Main orchestration script for RAG evaluation pipeline.

Runs the complete evaluation workflow:
1. Generate question variations (DeepSeek API)
2. Execute RAG queries (with/without beliefs)
3. LLM-as-Judge scoring
4. Generate markdown report
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

from v1.core.config import load_config
from v1.core.logger import setup_logger
from v1.evaluation.question_generator import QuestionGenerator
from v1.evaluation.rag_executor import RAGExecutor
from v1.evaluation.llm_judge import LLMJudge
from v1.evaluation.report_generator import ReportGenerator


logger = setup_logger(__name__)


def run_full_pipeline(
    skip_generation: bool = False,
    skip_execution: bool = False,
    skip_judging: bool = False,
    run_baseline: bool = True
) -> Path:
    """
    Run the complete evaluation pipeline.

    Args:
        skip_generation: Skip question variation generation
        skip_execution: Skip RAG execution
        skip_judging: Skip LLM judging
        run_baseline: Run with/without beliefs comparison

    Returns:
        Path to final report
    """
    config = load_config()
    output_dir = Path(config.evaluation.output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    print("\n" + "=" * 60)
    print("RAG EVALUATION PIPELINE")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output directory: {output_dir}")
    print("=" * 60 + "\n")

    # Phase 1: Question Generation
    if not skip_generation:
        print("\n[Phase 1/4] Generating question variations...")
        print("-" * 40)

        try:
            generator = QuestionGenerator(config)
            questions_path = generator.run()
            print(f"✓ Questions saved to: {questions_path}")
        except Exception as e:
            logger.error(f"Question generation failed: {e}")
            print(f"✗ Question generation failed: {e}")
            if not skip_execution:
                print("Cannot continue without questions. Exiting.")
                sys.exit(1)
    else:
        print("\n[Phase 1/4] Skipping question generation (--skip-generation)")
        questions_path = output_dir / "evaluation_questions.json"
        if not questions_path.exists():
            print(f"✗ Questions file not found: {questions_path}")
            print("Run without --skip-generation first.")
            sys.exit(1)

    # Phase 2: RAG Execution
    if not skip_execution:
        print("\n[Phase 2/4] Executing RAG queries...")
        print("-" * 40)
        print(f"Baseline comparison: {'Yes' if run_baseline else 'No'}")

        try:
            executor = RAGExecutor(config)
            results_path = executor.run(run_baseline=run_baseline)
            print(f"✓ Results saved to: {results_path}")
        except Exception as e:
            logger.error(f"RAG execution failed: {e}")
            print(f"✗ RAG execution failed: {e}")
            if not skip_judging:
                print("Cannot continue without results. Exiting.")
                sys.exit(1)
    else:
        print("\n[Phase 2/4] Skipping RAG execution (--skip-execution)")
        results_path = output_dir / "evaluation_results.json"
        if not results_path.exists():
            print(f"✗ Results file not found: {results_path}")
            print("Run without --skip-execution first.")
            sys.exit(1)

    # Phase 3: LLM Judging
    if not skip_judging:
        print("\n[Phase 3/4] Running LLM-as-Judge evaluation...")
        print("-" * 40)

        try:
            judge = LLMJudge(config)
            scores_path = judge.run(evaluate_baseline=run_baseline)
            print(f"✓ Scores saved to: {scores_path}")
        except Exception as e:
            logger.error(f"LLM judging failed: {e}")
            print(f"✗ LLM judging failed: {e}")
            print("Cannot generate report without scores. Exiting.")
            sys.exit(1)
    else:
        print("\n[Phase 3/4] Skipping LLM judging (--skip-judging)")
        scores_path = output_dir / "evaluation_scores.json"
        if not scores_path.exists():
            print(f"✗ Scores file not found: {scores_path}")
            print("Run without --skip-judging first.")
            sys.exit(1)

    # Phase 4: Report Generation
    print("\n[Phase 4/4] Generating evaluation report...")
    print("-" * 40)

    try:
        report_gen = ReportGenerator(config)
        report_path = report_gen.run()
        print(f"✓ Report saved to: {report_path}")
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        print(f"✗ Report generation failed: {e}")
        sys.exit(1)

    # Summary
    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Total time: {hours}h {minutes}m {seconds}s")
    print(f"\nOutput files:")
    print(f"  - Questions:  {output_dir / 'evaluation_questions.json'}")
    print(f"  - Results:    {output_dir / 'evaluation_results.json'}")
    print(f"  - Scores:     {output_dir / 'evaluation_scores.json'}")
    print(f"  - Report:     {report_path}")
    print("=" * 60 + "\n")

    return report_path


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="RAG Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python -m v1.evaluation.run_evaluation

  # Skip question generation (reuse existing)
  python -m v1.evaluation.run_evaluation --skip-generation

  # Only regenerate report from existing scores
  python -m v1.evaluation.run_evaluation --skip-generation --skip-execution --skip-judging

  # Run without baseline comparison (faster)
  python -m v1.evaluation.run_evaluation --no-baseline
"""
    )

    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip question variation generation (use existing questions)"
    )
    parser.add_argument(
        "--skip-execution",
        action="store_true",
        help="Skip RAG execution (use existing results)"
    )
    parser.add_argument(
        "--skip-judging",
        action="store_true",
        help="Skip LLM judging (use existing scores)"
    )
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="Skip baseline (without beliefs) comparison"
    )

    args = parser.parse_args()

    try:
        report_path = run_full_pipeline(
            skip_generation=args.skip_generation,
            skip_execution=args.skip_execution,
            skip_judging=args.skip_judging,
            run_baseline=not args.no_baseline
        )
        print(f"\nView the report: {report_path}")

    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logger.exception("Pipeline failed with unexpected error")
        print(f"\n✗ Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
