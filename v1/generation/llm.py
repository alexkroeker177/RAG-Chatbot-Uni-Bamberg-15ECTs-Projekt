"""
LLM interface using Ollama.

Wraps Ollama LLM with streaming support.
"""

from langchain_community.llms import Ollama

from v1.core.config import OllamaConfig
from v1.core.logger import setup_logger


logger = setup_logger(__name__)


def create_llm(config: OllamaConfig, streaming: bool = True) -> Ollama:
    """
    Create Ollama LLM instance.
    
    Args:
        config: Ollama configuration
        streaming: Enable token streaming
        
    Returns:
        Ollama LLM instance
    """
    llm = Ollama(
        base_url=config.base_url,
        model=config.generation_model,
        temperature=config.temperature,
        num_predict=config.max_tokens
    )
    
    logger.info(
        f"Created LLM: {config.generation_model} "
        f"(temp={config.temperature}, streaming={streaming})"
    )
    
    return llm


# Example usage
if __name__ == "__main__":
    from v1.core.config import load_config
    
    config = load_config()
    
    try:
        llm = create_llm(config.ollama, streaming=False)
        
        test_prompt = "Sage 'Hallo' auf Deutsch."
        print(f"Testing LLM with prompt: {test_prompt}\n")
        
        response = llm.invoke(test_prompt)
        print(f"Response: {response}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you're connected to the Ollama server")
