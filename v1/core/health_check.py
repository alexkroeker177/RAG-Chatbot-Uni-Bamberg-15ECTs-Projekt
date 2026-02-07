"""
Ollama server health check utilities.

Verifies connectivity and model availability before operations.
"""

import requests
from typing import Tuple, Optional

from v1.core.config import Config
from v1.core.logger import setup_logger


logger = setup_logger(__name__)


def check_ollama_server(base_url: str, timeout: int = 10) -> Tuple[bool, str]:
    """
    Check if Ollama server is reachable.
    
    Args:
        base_url: Ollama server URL
        timeout: Request timeout in seconds
        
    Returns:
        Tuple of (success, message)
    """
    try:
        # Try to get list of available models
        response = requests.get(
            f"{base_url}/api/tags",
            timeout=timeout
        )
        
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [m.get("name", "unknown") for m in models]
            return True, f"Server reachable. Available models: {len(model_names)}"
        else:
            return False, f"Server returned status code {response.status_code}"
            
    except requests.exceptions.ConnectionError:
        return False, f"Cannot connect to server at {base_url}"
    except requests.exceptions.Timeout:
        return False, f"Connection timeout after {timeout}s"
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"


def list_available_models(base_url: str, timeout: int = 10) -> Optional[list]:
    """
    Get list of available models from Ollama server.
    
    Args:
        base_url: Ollama server URL
        timeout: Request timeout in seconds
        
    Returns:
        List of model dictionaries or None if failed
    """
    try:
        response = requests.get(
            f"{base_url}/api/tags",
            timeout=timeout
        )
        
        if response.status_code == 200:
            return response.json().get("models", [])
        return None
            
    except Exception:
        return None


def check_embedding_model(base_url: str, model: str, timeout: int = 30) -> Tuple[bool, str]:
    """
    Check if embedding model is available and working.
    
    Args:
        base_url: Ollama server URL
        model: Embedding model name
        timeout: Request timeout in seconds
        
    Returns:
        Tuple of (success, message)
    """
    try:
        # Try to generate a test embedding
        response = requests.post(
            f"{base_url}/api/embeddings",
            json={
                "model": model,
                "prompt": "test"
            },
            timeout=timeout
        )
        
        if response.status_code == 200:
            data = response.json()
            if "embedding" in data:
                embedding_dim = len(data["embedding"])
                return True, f"Model '{model}' working (dim={embedding_dim})"
            else:
                return False, f"Model '{model}' returned invalid response"
        elif response.status_code == 404:
            return False, f"Model '{model}' not found on server"
        else:
            return False, f"Model '{model}' returned status {response.status_code}"
            
    except requests.exceptions.Timeout:
        return False, f"Model '{model}' request timeout after {timeout}s"
    except Exception as e:
        return False, f"Error testing model '{model}': {str(e)}"


def check_generation_model(base_url: str, model: str, timeout: int = 30) -> Tuple[bool, str]:
    """
    Check if generation model is available and working.
    
    Args:
        base_url: Ollama server URL
        model: Generation model name
        timeout: Request timeout in seconds
        
    Returns:
        Tuple of (success, message)
    """
    try:
        # Try to generate a test response
        response = requests.post(
            f"{base_url}/api/generate",
            json={
                "model": model,
                "prompt": "Say 'OK'",
                "stream": False
            },
            timeout=timeout
        )
        
        if response.status_code == 200:
            data = response.json()
            if "response" in data:
                return True, f"Model '{model}' working"
            else:
                return False, f"Model '{model}' returned invalid response"
        elif response.status_code == 404:
            return False, f"Model '{model}' not found on server"
        else:
            return False, f"Model '{model}' returned status {response.status_code}"
            
    except requests.exceptions.Timeout:
        return False, f"Model '{model}' request timeout after {timeout}s"
    except Exception as e:
        return False, f"Error testing model '{model}': {str(e)}"


def perform_health_check(config: Config, check_models: bool = True, show_available: bool = True) -> bool:
    """
    Perform comprehensive health check of Ollama server.
    
    Args:
        config: Configuration object
        check_models: Whether to check individual models (slower)
        show_available: Whether to display all available models
        
    Returns:
        True if all checks pass, False otherwise
    """
    logger.info("=" * 50)
    logger.info("Performing Ollama Health Check")
    logger.info("=" * 50)
    
    all_passed = True
    
    # Check server connectivity
    logger.info(f"Checking server: {config.ollama.base_url}")
    success, message = check_ollama_server(config.ollama.base_url)
    
    if success:
        logger.info(f"✓ Server check passed: {message}")
    else:
        logger.error(f"✗ Server check failed: {message}")
        all_passed = False
        return all_passed  # No point checking models if server is down
    
    # Show available models
    if show_available:
        logger.info("")
        logger.info("Available Models:")
        logger.info("-" * 50)
        models = list_available_models(config.ollama.base_url)
        
        if models:
            # Separate by type
            embedding_models = []
            generation_models = []
            
            for model in models:
                name = model.get("name", "unknown")
                size = model.get("size", 0)
                size_gb = size / (1024**3) if size else 0
                
                # Heuristic: embedding models typically have "embed" in name
                if "embed" in name.lower():
                    embedding_models.append((name, size_gb))
                else:
                    generation_models.append((name, size_gb))
            
            if embedding_models:
                logger.info("  Embedding Models:")
                for name, size in sorted(embedding_models):
                    logger.info(f"    • {name} ({size:.1f} GB)")
            
            if generation_models:
                logger.info("  Generation Models:")
                for name, size in sorted(generation_models):
                    logger.info(f"    • {name} ({size:.1f} GB)")
            
            logger.info(f"  Total: {len(models)} models")
        else:
            logger.warning("  Could not retrieve model list")
        
        logger.info("-" * 50)
        logger.info("")
    
    if check_models:
        # Check embedding model
        logger.info(f"Checking embedding model: {config.ollama.embedding_model}")
        success, message = check_embedding_model(
            config.ollama.base_url,
            config.ollama.embedding_model
        )
        
        if success:
            logger.info(f"✓ Embedding model check passed: {message}")
        else:
            logger.error(f"✗ Embedding model check failed: {message}")
            all_passed = False
        
        # Check generation model
        logger.info(f"Checking generation model: {config.ollama.generation_model}")
        success, message = check_generation_model(
            config.ollama.base_url,
            config.ollama.generation_model
        )
        
        if success:
            logger.info(f"✓ Generation model check passed: {message}")
        else:
            logger.error(f"✗ Generation model check failed: {message}")
            all_passed = False
    
    logger.info("=" * 50)
    if all_passed:
        logger.info("✓ All health checks passed!")
    else:
        logger.error("✗ Some health checks failed")
    logger.info("=" * 50)
    
    return all_passed


# CLI tool for testing
if __name__ == "__main__":
    import sys
    from v1.core.config import load_config
    
    try:
        config = load_config()
        success = perform_health_check(config, check_models=True)
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        sys.exit(1)
