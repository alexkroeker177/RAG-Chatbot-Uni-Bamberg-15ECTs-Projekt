"""
Configuration management using Pydantic models.

Loads and validates configuration from YAML file.
"""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator


class OllamaConfig(BaseModel):
    """Ollama server configuration."""
    
    base_url: str = Field(
        default="http://ursa.ds.uni-bamberg.de:11434",
        description="Ollama server URL"
    )
    generation_model: str = Field(
        default="deepseek-r1:70b",
        description="LLM model for answer generation"
    )
    embedding_model: str = Field(
        default="bge-m3:latest",
        description="Embedding model for vector generation"
    )
    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="LLM temperature for generation"
    )
    max_tokens: int = Field(
        default=2048,
        gt=0,
        description="Maximum tokens for LLM generation"
    )


class ChromaConfig(BaseModel):
    """Chroma vector database configuration."""
    
    persist_directory: str = Field(
        default="./chroma_db",
        description="Directory for Chroma persistence"
    )
    collection_name: str = Field(
        default="wiai-regs-2024-11",
        description="Chroma collection name"
    )
    distance_metric: Literal["cosine", "l2", "ip"] = Field(
        default="cosine",
        description="Distance metric for similarity search"
    )


class RetrievalConfig(BaseModel):
    """Retrieval engine configuration."""
    
    fetch_k: int = Field(
        default=40,
        gt=0,
        description="Number of candidates to fetch"
    )
    final_k: int = Field(
        default=6,
        gt=0,
        description="Number of final results to return"
    )
    search_type: Literal["similarity", "mmr"] = Field(
        default="mmr",
        description="Search algorithm type"
    )
    lambda_mult: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="MMR diversity parameter (0=max diversity, 1=max relevance)"
    )
    
    @field_validator("final_k")
    @classmethod
    def validate_final_k(cls, v, info):
        """Ensure final_k <= fetch_k."""
        if "fetch_k" in info.data and v > info.data["fetch_k"]:
            raise ValueError(f"final_k ({v}) must be <= fetch_k ({info.data['fetch_k']})")
        return v


class ChunkingConfig(BaseModel):
    """Text chunking configuration."""
    
    chunk_size: int = Field(
        default=900,
        gt=0,
        description="Approximate tokens per chunk"
    )
    chunk_overlap: int = Field(
        default=120,
        ge=0,
        description="Token overlap between chunks"
    )
    
    @field_validator("chunk_overlap")
    @classmethod
    def validate_overlap(cls, v, info):
        """Ensure overlap < chunk_size."""
        if "chunk_size" in info.data and v >= info.data["chunk_size"]:
            raise ValueError(f"chunk_overlap ({v}) must be < chunk_size ({info.data['chunk_size']})")
        return v


class DataConfig(BaseModel):
    """Data source paths configuration."""
    
    pdf_directory: str = Field(
        default="./Studienordnungen",
        description="Directory containing PDF documents"
    )
    faq_file: str = Field(
        default="./QA.json",
        description="FAQ JSON file path"
    )
    departments_file: str = Field(
        default="./departments.json",
        description="Department routing JSON file path"
    )


class LoggingConfig(BaseModel):
    """Logging configuration."""
    
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="DEBUG",
        description="Logging level"
    )
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log message format"
    )
    file: str = Field(
        default="rag_system.log",
        description="Log file path"
    )


class IntentClassificationConfig(BaseModel):
    """Intent classification configuration."""

    enabled: bool = Field(
        default=True,
        description="Enable intent classification"
    )
    model: str = Field(
        default="qwen2.5:14b",
        description="Model for classification and conversational responses"
    )
    confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for classification"
    )
    timeout: float = Field(
        default=2.0,
        gt=0.0,
        description="Maximum seconds for classification"
    )
    fallback_to_query: bool = Field(
        default=True,
        description="Treat uncertain messages as queries"
    )


class EvaluationConfig(BaseModel):
    """Evaluation pipeline configuration."""

    evaluation_model: str = Field(
        default="deepseek-llm:67b",
        description="Ollama model for question variation generation"
    )
    judge_model: str = Field(
        default="deepseek-llm:67b",
        description="Ollama model for LLM-as-Judge evaluation"
    )
    output_directory: str = Field(
        default="./evaluation_output",
        description="Directory for evaluation results"
    )
    variations_per_question: int = Field(
        default=6,
        ge=1,
        le=10,
        description="Number of variations to generate per FAQ question"
    )


class Config(BaseModel):
    """Main configuration model."""

    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    chroma: ChromaConfig = Field(default_factory=ChromaConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    intent_classification: IntentClassificationConfig = Field(
        default_factory=IntentClassificationConfig
    )
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)


def load_config(config_path: str = "config.yaml") -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Validated Config object
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config validation fails
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            f"Please create it from config.example.yaml"
        )
    
    with open(config_file, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)
    
    try:
        config = Config(**config_dict)
        return config
    except Exception as e:
        raise ValueError(f"Configuration validation failed: {e}")


def get_config_summary(config: Config) -> str:
    """
    Generate a human-readable summary of the configuration.
    
    Args:
        config: Config object
        
    Returns:
        Formatted configuration summary
    """
    summary = [
        "=== Configuration Summary ===",
        "",
        "Ollama:",
        f"  Server: {config.ollama.base_url}",
        f"  Generation Model: {config.ollama.generation_model}",
        f"  Embedding Model: {config.ollama.embedding_model}",
        f"  Temperature: {config.ollama.temperature}",
        "",
        "Chroma:",
        f"  Directory: {config.chroma.persist_directory}",
        f"  Collection: {config.chroma.collection_name}",
        f"  Distance Metric: {config.chroma.distance_metric}",
        "",
        "Retrieval:",
        f"  Fetch K: {config.retrieval.fetch_k}",
        f"  Final K: {config.retrieval.final_k}",
        f"  Search Type: {config.retrieval.search_type}",
        "",
        "Chunking:",
        f"  Chunk Size: {config.chunking.chunk_size} tokens",
        f"  Overlap: {config.chunking.chunk_overlap} tokens",
        "",
        "Data Sources:",
        f"  PDFs: {config.data.pdf_directory}",
        f"  FAQs: {config.data.faq_file}",
        f"  Departments: {config.data.departments_file}",
        "",
        "Logging:",
        f"  Level: {config.logging.level}",
        f"  File: {config.logging.file}",
        "",
        "Intent Classification:",
        f"  Enabled: {config.intent_classification.enabled}",
        f"  Model: {config.intent_classification.model}",
        f"  Confidence Threshold: {config.intent_classification.confidence_threshold}",
        "",
    ]
    
    return "\n".join(summary)
