"""
Centralized logging setup with verbose debugging support.

Provides colored console output and file logging.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import colorlog


def setup_logger(
    name: str,
    level: str = "DEBUG",
    log_file: Optional[str] = "rag_system.log",
    log_format: Optional[str] = None
) -> logging.Logger:
    """
    Set up a logger with console and file handlers.
    
    Args:
        name: Logger name (typically __name__)
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Path to log file (None to disable file logging)
        log_format: Custom log format string
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    logger.setLevel(getattr(logging, level.upper()))
    
    # Default format
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Console handler with colors
    console_handler = colorlog.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    
    console_format = colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s - %(name)s - %(levelname)s%(reset)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        }
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)  # Always log everything to file
        
        file_format = logging.Formatter(
            log_format,
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger


class LoggerMixin:
    """Mixin class to add logging capabilities to any class."""
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        if not hasattr(self, "_logger"):
            self._logger = logging.getLogger(self.__class__.__name__)
        return self._logger


def log_ingestion_start(logger: logging.Logger, source: str, count: int):
    """Log start of document ingestion."""
    logger.info(f"Starting ingestion from {source}: {count} items")


def log_ingestion_progress(logger: logging.Logger, current: int, total: int, item_name: str):
    """Log ingestion progress."""
    logger.debug(f"Processing [{current}/{total}]: {item_name}")


def log_ingestion_complete(logger: logging.Logger, source: str, chunks_added: int, duration: float):
    """Log completion of document ingestion."""
    logger.info(
        f"Ingestion complete for {source}: "
        f"{chunks_added} chunks added in {duration:.2f}s"
    )


def log_query(logger: logging.Logger, query: str, latency: float, success: bool):
    """Log query execution."""
    status = "SUCCESS" if success else "FAILED"
    logger.info(f"Query [{status}] ({latency:.2f}s): {query[:100]}...")


def log_retrieval(logger: logging.Logger, query: str, num_results: int):
    """Log retrieval results."""
    logger.debug(f"Retrieved {num_results} chunks for query: {query[:100]}...")


def log_error(logger: logging.Logger, component: str, error: Exception, context: Optional[str] = None):
    """Log error with context."""
    error_msg = f"Error in {component}: {str(error)}"
    if context:
        error_msg += f" | Context: {context}"
    logger.error(error_msg, exc_info=True)


def log_health_check(logger: logging.Logger, service: str, success: bool, message: str = ""):
    """Log health check result."""
    if success:
        logger.info(f"Health check PASSED for {service}")
    else:
        logger.error(f"Health check FAILED for {service}: {message}")


# Example usage
if __name__ == "__main__":
    # Test the logger
    logger = setup_logger("test", level="DEBUG")
    
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    # Test helper functions
    log_ingestion_start(logger, "PDFs", 20)
    log_ingestion_progress(logger, 5, 20, "document.pdf")
    log_ingestion_complete(logger, "PDFs", 150, 45.3)
    log_query(logger, "Wie viele ECTS brauche ich?", 1.2, True)
    log_health_check(logger, "Ollama", True)
    
    print("\nCheck rag_system.log for file output")
