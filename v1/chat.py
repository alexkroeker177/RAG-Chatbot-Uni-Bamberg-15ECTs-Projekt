#!/usr/bin/env python3
"""
CLI chat interface for RAG system.

Simple command-line chat with streaming responses.
"""

import sys

from v1.core.config import load_config
from v1.core.health_check import perform_health_check
from v1.core.logger import setup_logger
from v1.generation.rag_chain import RAGChain


logger = setup_logger(__name__)


def print_header():
    """Print welcome header."""
    print("\n" + "=" * 70)
    print("RAG CHATBOT - Prüfungsordnungen Universität Bamberg")
    print("=" * 70)
    print("\nStelle deine Fragen zu Prüfungsordnungen und Regelungen.")
    print("Befehle: 'exit' oder 'quit' zum Beenden\n")


def print_sources(sources: list):
    """Print source citations."""
    if not sources:
        return
    
    print("\n📚 Quellen:")
    for src in sources:
        page_info = f", Seite {src['page']}" if src['page'] != 'n/a' else ""
        print(f"  • {src['doc']}{page_info}")


def main():
    """Main CLI chat loop."""
    try:
        # Load configuration
        config = load_config()
        
        # Quick health check with model listing
        logger.info("Performing quick health check...")
        print("\n🔍 Checking Ollama server and available models...")
        if not perform_health_check(config, check_models=False, show_available=True):
            print("❌ Health check failed. Please check your configuration.")
            return 1
        
        # Initialize RAG chain
        print("\n🔄 Initializing RAG system...")
        rag = RAGChain(config)
        print("✅ System ready!\n")
        
        # Print header
        print_header()
        
        # Session tracking for intent classification
        session_id = "cli_session"
        conversation_history = []
        
        # Chat loop
        while True:
            try:
                # Get user input
                question = input("💬 Du: ").strip()
                
                if not question:
                    continue
                
                # Check for exit commands
                if question.lower() in ['exit', 'quit', 'bye', 'tschüss']:
                    print("\n👋 Auf Wiedersehen!\n")
                    break
                
                # Get answer with intent classification and streaming
                print("\n🤖 Assistent: ", end="", flush=True)
                
                # Use stream_with_intent for streaming responses
                answer_stream, metadata = rag.stream_with_intent(
                    question=question,
                    session_id=session_id,
                    conversation_history=conversation_history[-3:]  # Last 3 turns
                )
                
                # Stream the answer
                full_answer = ""
                for chunk in answer_stream:
                    print(chunk, end="", flush=True)
                    full_answer += chunk
                
                print()  # New line after streaming
                
                # Print sources if available
                print_sources(metadata['sources'])
                
                # Show intent in debug mode
                if config.logging.level == "DEBUG":
                    intent = metadata.get('intent', 'unknown')
                    confidence = metadata.get('confidence', 0.0)
                    print(f"\n[Debug: Intent={intent}, Confidence={confidence:.2f}]")
                
                # Update conversation history
                conversation_history.append({
                    "role": "user",
                    "content": question
                })
                conversation_history.append({
                    "role": "assistant",
                    "content": full_answer
                })
                
                print()  # Extra line for spacing
                
            except KeyboardInterrupt:
                print("\n\n👋 Auf Wiedersehen!\n")
                break
            except Exception as e:
                logger.error(f"Error processing question: {e}")
                print(f"\n❌ Fehler: {e}\n")
        
        return 0
        
    except Exception as e:
        logger.error(f"Failed to start chat: {e}")
        print(f"\n❌ Fehler beim Starten: {e}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
