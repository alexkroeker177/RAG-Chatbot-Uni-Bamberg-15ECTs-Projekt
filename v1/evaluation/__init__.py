"""
Evaluation pipeline for RAG system.

This module provides tools for:
- Generating question variations from FAQ entries
- Running RAG with/without beliefs for baseline comparison
- LLM-as-Judge evaluation with 0.0-1.0 scoring
- Aggregate metrics and report generation
"""

from v1.evaluation.question_generator import QuestionGenerator
from v1.evaluation.rag_executor import RAGExecutor
from v1.evaluation.llm_judge import LLMJudge
from v1.evaluation.metrics import MetricsCalculator
from v1.evaluation.report_generator import ReportGenerator

__all__ = [
    "QuestionGenerator",
    "RAGExecutor",
    "LLMJudge",
    "MetricsCalculator",
    "ReportGenerator",
]
