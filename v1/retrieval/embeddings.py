"""
Embedding generation using Ollama.

Wraps OllamaEmbeddings with retry logic and health checks.
"""

import time
from typing import List, Optional

from langchain_community.embeddings import OllamaEmbeddings

from v1.core.config import OllamaConfig
from v1.core.logger import setup_logger


logger = setup_logger(__name__)


class EmbeddingGenerator:
    """Generates embeddings via Ollama with retry logic."""
    
    def __init__(self, config: OllamaConfig):
        """
        Initialize embedding generator.
        
        Args:
            config: Ollama configuration
        """
        self.config = config
        self.embeddings = OllamaEmbeddings(
            base_url=config.base_url,
            model=config.embedding_model
        )
        
        logger.info(
            f"Initialized embeddings: {config.embedding_model} "
            f"at {config.base_url}"
        )
    
    def health_check(self) -> bool:
        """
        Verify embedding model is working.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            logger.debug("Running embedding health check...")
            test_embedding = self.embeddings.embed_query("test")
            
            if test_embedding and len(test_embedding) > 0:
                logger.info(
                    f"✓ Embedding health check passed "
                    f"(dim={len(test_embedding)})"
                )
                return True
            else:
                logger.error("✗ Embedding health check failed: empty embedding")
                return False
                
        except Exception as e:
            logger.error(f"✗ Embedding health check failed: {e}")
            return False
    
    def embed_documents(
        self,
        texts: List[str],
        max_retries: int = 3
    ) -> Optional[List[List[float]]]:
        """
        Generate embeddings for multiple texts with retry logic.
        
        Args:
            texts: List of texts to embed
            max_retries: Maximum number of retry attempts
            
        Returns:
            List of embeddings or None if failed
        """
        for attempt in range(max_retries):
            try:
                embeddings = self.embeddings.embed_documents(texts)
                
                if embeddings and len(embeddings) == len(texts):
                    logger.debug(
                        f"Generated {len(embeddings)} embeddings "
                        f"(dim={len(embeddings[0])})"
                    )
                    return embeddings
                else:
                    logger.warning(
                        f"Embedding count mismatch: "
                        f"expected {len(texts)}, got {len(embeddings) if embeddings else 0}"
                    )
                    
            except Exception as e:
                logger.warning(
                    f"Embedding attempt {attempt + 1}/{max_retries} failed: {e}"
                )
                
                if attempt < max_retries - 1:
                    # Wait before retry (exponential backoff)
                    wait_time = 2 ** attempt
                    logger.debug(f"Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"All {max_retries} embedding attempts failed")
        
        return None
    
    def embed_query(
        self,
        text: str,
        max_retries: int = 3
    ) -> Optional[List[float]]:
        """
        Generate embedding for a single query with retry logic.
        
        Args:
            text: Text to embed
            max_retries: Maximum number of retry attempts
            
        Returns:
            Embedding vector or None if failed
        """
        for attempt in range(max_retries):
            try:
                embedding = self.embeddings.embed_query(text)
                
                if embedding and len(embedding) > 0:
                    logger.debug(f"Generated query embedding (dim={len(embedding)})")
                    return embedding
                    
            except Exception as e:
                logger.warning(
                    f"Query embedding attempt {attempt + 1}/{max_retries} failed: {e}"
                )
                
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.debug(f"Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"All {max_retries} query embedding attempts failed")
        
        return None


def create_embeddings(config: OllamaConfig) -> OllamaEmbeddings:
    """
    Create and validate OllamaEmbeddings instance.
    
    Args:
        config: Ollama configuration
        
    Returns:
        OllamaEmbeddings instance
        
    Raises:
        RuntimeError: If embeddings fail health check
    """
    generator = EmbeddingGenerator(config)
    
    # Run health check
    if not generator.health_check():
        raise RuntimeError(
            f"Embedding model '{config.embedding_model}' failed health check. "
            f"Please verify Ollama server is running and model is available."
        )
    
    return generator.embeddings


# Example usage
if __name__ == "__main__":
    from v1.core.config import load_config
    
    # Load config
    config = load_config()
    
    try:
        # Create embedding generator
        generator = EmbeddingGenerator(config.ollama)
        
        # Run health check
        if generator.health_check():
            print("\n✓ Embedding generator is healthy")
            
            # Test embedding generation
            test_texts = [
                "Dies ist ein Test auf Deutsch.",
                "This is a test in English."
            ]
            
            print(f"\nGenerating embeddings for {len(test_texts)} texts...")
            embeddings = generator.embed_documents(test_texts)
            
            if embeddings:
                print(f"✓ Generated {len(embeddings)} embeddings")
                print(f"  Embedding dimension: {len(embeddings[0])}")
                print(f"  Sample values: {embeddings[0][:5]}...")
            else:
                print("✗ Failed to generate embeddings")
        else:
            print("\n✗ Embedding generator health check failed")
            
    except Exception as e:
        print(f"\nError: {e}")
        print("Make sure you're connected to the Ollama server")
