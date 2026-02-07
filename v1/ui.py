#!/usr/bin/env python3
"""
Streamlit web UI for RAG chatbot.

Provides a ChatGPT-like interface with conversation history and source display.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import requests

from v1.core.config import load_config
from v1.core.logger import setup_logger
from v1.generation.rag_chain import RAGChain


logger = setup_logger(__name__)


# Page configuration
st.set_page_config(
    page_title="RAG Chatbot - Prüfungsordnungen",
    page_icon="🎓",
    layout="wide"
)


def get_available_models(base_url: str) -> list:
    """
    Fetch available models from Ollama server.
    
    Args:
        base_url: Ollama server URL
        
    Returns:
        List of model names
    """
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = [model["name"] for model in data.get("models", [])]
            return sorted(models)
        else:
            logger.warning(f"Failed to fetch models: {response.status_code}")
            return []
    except Exception as e:
        logger.error(f"Error fetching models: {e}")
        return []


@st.cache_resource
def initialize_rag(_config=None):
    """Initialize RAG chain (cached)."""
    if _config is None:
        _config = load_config()
    return RAGChain(_config)


def display_message(role: str, content: str, sources: list = None):
    """Display a chat message with optional sources."""
    with st.chat_message(role):
        st.markdown(content)
        
        if sources:
            with st.expander("📚 Quellen anzeigen"):
                for src in sources:
                    page_info = f", Seite {src['page']}" if src['page'] != 'n/a' else ""
                    st.markdown(f"• **{src['doc']}**{page_info}")


def main():
    """Main Streamlit app."""
    
    # Initialize session state FIRST
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "session_id" not in st.session_state:
        import uuid
        st.session_state.session_id = str(uuid.uuid4())
    
    if "user_context" not in st.session_state:
        st.session_state.user_context = {
            "degree_level": None,
            "program": None,
            "semester": None
        }
    
    if "profile_submitted" not in st.session_state:
        st.session_state.profile_submitted = False
    
    # Load config
    config = load_config()
    
    # Log available models at startup (only once)
    if "models_logged" not in st.session_state:
        logger.info("=" * 70)
        logger.info("STREAMLIT UI STARTUP")
        logger.info("=" * 70)
        logger.info(f"Ollama Server: {config.ollama.base_url}")
        
        available_models = get_available_models(config.ollama.base_url)
        if available_models:
            logger.info(f"\n📦 Available Models ({len(available_models)}):")
            
            # Separate by type
            embedding_models = [m for m in available_models if "embed" in m.lower()]
            generation_models = [m for m in available_models if "embed" not in m.lower()]
            
            if embedding_models:
                logger.info("\n  Embedding Models:")
                for model in sorted(embedding_models):
                    marker = "✓" if model == config.ollama.embedding_model else " "
                    logger.info(f"    [{marker}] {model}")
            
            if generation_models:
                logger.info("\n  Generation Models:")
                for model in sorted(generation_models):
                    marker = "✓" if model == config.ollama.generation_model else " "
                    logger.info(f"    [{marker}] {model}")
            
            logger.info(f"\n🎯 Active Configuration:")
            logger.info(f"  Generation: {config.ollama.generation_model}")
            logger.info(f"  Embedding: {config.ollama.embedding_model}")
            logger.info(f"  Intent + Conversational: {config.intent_classification.model}")
        else:
            logger.warning("⚠️  No models available from Ollama server")
        
        logger.info("=" * 70)
        st.session_state.models_logged = True
    
    # Initialize selected model in session state
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = config.ollama.generation_model
    
    # Title
    st.title("🎓 RAG Chatbot - Prüfungsordnungen")
    st.caption("Universität Bamberg - Fakultät WIAI")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Einstellungen")
        
        # Model selector
        st.subheader("🤖 Sprachmodell")
        available_models = get_available_models(config.ollama.base_url)
        
        if available_models:
            # Find current model index
            try:
                current_index = available_models.index(st.session_state.selected_model)
            except ValueError:
                current_index = 0
            
            selected_model = st.selectbox(
                "Wähle ein Modell:",
                available_models,
                index=current_index,
                help="Wähle ein Sprachmodell für die Antwortgenerierung"
            )
            
            # Update config if model changed
            if selected_model != st.session_state.selected_model:
                st.session_state.selected_model = selected_model
                config.ollama.generation_model = selected_model
                # Clear cache to reinitialize with new model
                st.cache_resource.clear()
                st.rerun()
        else:
            st.warning("⚠️ Keine Modelle verfügbar")
            st.caption(f"Aktuell: {config.ollama.generation_model}")
        
        st.divider()
        
        st.header("ℹ️ Information")
        st.markdown("""
        Dieser Chatbot beantwortet Fragen zu:
        - Prüfungsordnungen
        - Studienordnungen
        - FAQ der Universität
        - Zuständige Abteilungen
        
        **Hinweis:** Antworten basieren ausschließlich auf den 
        verfügbaren Dokumenten.
        """)
        
        st.divider()
        
        # User Context Display & Edit
        st.header("👤 Dein Profil")
        if st.session_state.user_context["degree_level"]:
            st.write(f"**Abschluss:** {st.session_state.user_context['degree_level']}")
        if st.session_state.user_context["program"]:
            st.write(f"**Studiengang:** {st.session_state.user_context['program']}")
        if st.session_state.user_context["semester"]:
            st.write(f"**Semester:** {st.session_state.user_context['semester']}")
        
        if not any(st.session_state.user_context.values()):
            st.caption("_Noch keine Angaben_")
        
        # Button to reset profile and show form again
        if st.session_state.profile_submitted:
            if st.button("✏️ Profil ändern"):
                st.session_state.profile_submitted = False
                st.rerun()
        
        st.divider()
        
        # Stats
        st.header("📊 System")
        st.metric("Modell", st.session_state.selected_model.split(":")[0])
        st.metric("Embeddings", config.ollama.embedding_model.split(":")[0])
        
        st.divider()
        
        # Clear chat button
        if st.button("🗑️ Chat löschen"):
            st.session_state.messages = []
            st.session_state.profile_submitted = False
            st.session_state.beliefs_initialized = False
            st.session_state.user_context = {
                "degree_level": None,
                "program": None,
                "semester": None
            }
            st.rerun()
    
    # Initialize RAG chain with selected model
    try:
        with st.spinner("Initialisiere RAG System..."):
            # Update config with selected model
            config.ollama.generation_model = st.session_state.selected_model
            rag = initialize_rag(config)
    except Exception as e:
        st.error(f"❌ Fehler beim Initialisieren: {e}")
        st.stop()
    
    # Show welcome message if first visit
    if len(st.session_state.messages) == 0:
        welcome_msg = """👋 Willkommen! Ich helfe dir bei Fragen zu Prüfungsordnungen und Regelungen.

Bitte gib zunächst deine Studieninformationen an, damit ich dir besser helfen kann."""

        st.session_state.messages.append({
            "role": "assistant",
            "content": welcome_msg
        })
    
    # Display chat history
    for message in st.session_state.messages:
        display_message(
            message["role"],
            message["content"],
            message.get("sources")
        )
    
    # Show profile selection form if not yet submitted (OBLIGATORY)
    if not st.session_state.profile_submitted:
        with st.chat_message("assistant"):
            st.markdown("**📋 Dein Studienprofil**")
            st.caption("Diese Angaben helfen mir, dir passende Informationen zu geben.")

            degree = st.selectbox(
                "Abschluss",
                ["Bachelor", "Master", "Andere/Weiß nicht"],
                key="form_degree"
            )

            program = st.selectbox(
                "Studiengang",
                [
                    "Angewandte Informatik",
                    "Wirtschaftsinformatik",
                    "Software Systems Science",
                    "Computing in the Humanities",
                    "International Information Systems Management (IISM)",
                    "International Software Systems Science (ISSS)",
                    "Andere/Weiß nicht"
                ],
                key="form_program"
            )

            semester = st.selectbox(
                "Semester",
                [str(i) for i in range(1, 13)] + ["Weiß nicht"],
                key="form_semester"
            )

            if st.button("✓ Weiter zum Chat", type="primary", use_container_width=True):
                # Update user context (None for "Andere/Weiß nicht" options)
                st.session_state.user_context["degree_level"] = degree if degree != "Andere/Weiß nicht" else None

                # Extract clean program name
                clean_program = program.split(" (")[0] if "(" in program else program
                st.session_state.user_context["program"] = clean_program if program != "Andere/Weiß nicht" else None

                st.session_state.user_context["semester"] = int(semester) if semester.isdigit() else None

                st.session_state.profile_submitted = True

                # Build confirmation message
                has_degree = degree != "Andere/Weiß nicht"
                has_program = program != "Andere/Weiß nicht"
                has_semester = semester != "Weiß nicht"

                if has_program and has_degree and has_semester:
                    confirm_msg = f"Alles klar, du studierst {clean_program} im {degree}, {semester}. Semester. Wie kann ich dir helfen?"
                elif has_program and has_degree:
                    confirm_msg = f"Alles klar, du studierst {clean_program} im {degree}. Wie kann ich dir helfen?"
                elif has_program and has_semester:
                    confirm_msg = f"Alles klar, du studierst {clean_program} im {semester}. Semester. Wie kann ich dir helfen?"
                elif has_degree and has_semester:
                    confirm_msg = f"Alles klar, du bist im {degree}, {semester}. Semester. Wie kann ich dir helfen?"
                elif has_program:
                    confirm_msg = f"Alles klar, du studierst {clean_program}. Wie kann ich dir helfen?"
                elif has_degree:
                    confirm_msg = f"Alles klar, du bist im {degree}. Wie kann ich dir helfen?"
                elif has_semester:
                    confirm_msg = f"Alles klar, du bist im {semester}. Semester. Wie kann ich dir helfen?"
                else:
                    confirm_msg = "Alles klar! Wie kann ich dir bei Fragen zu Prüfungsordnungen helfen?"

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": confirm_msg
                })
                st.rerun()
    
    # Chat input (only available after profile form is submitted)
    if not st.session_state.profile_submitted:
        st.chat_input("Bitte fülle zuerst das Studienprofil aus...", disabled=True)
        return  # Don't proceed until form is submitted

    # Initialize beliefs from form data (once per session)
    if "beliefs_initialized" not in st.session_state:
        rag.initialize_user_context(
            st.session_state.session_id,
            st.session_state.user_context
        )
        st.session_state.beliefs_initialized = True

    if prompt := st.chat_input("Stelle deine Frage..."):
        # Add user message to history
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })
        
        # Display user message
        display_message("user", prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            sources_placeholder = st.empty()
            debug_placeholder = st.empty()
            full_response = ""
            
            # Generate response with intent classification and streaming
            try:
                # Use stream_with_intent for streaming responses
                answer_stream, metadata = rag.stream_with_intent(
                    question=prompt,
                    session_id=st.session_state.session_id,
                    conversation_history=st.session_state.messages[-3:]  # Last 3 turns
                )
                
                # Stream the answer
                for chunk in answer_stream:
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")
                
                # Remove cursor
                message_placeholder.markdown(full_response)
                
                # Get metadata (Conv-BDI format)
                sources = metadata.get('sources', [])
                intent = metadata.get('intention', metadata.get('intent', 'unknown'))
                confidence = metadata.get('confidence', 0.0)
                
                # Handle both Conv-BDI beliefs (dict) and legacy user_context (object)
                beliefs = metadata.get('beliefs') or metadata.get('user_context')
                
                # Update user context in session state if extracted
                if beliefs:
                    # Handle dict format (Conv-BDI)
                    if isinstance(beliefs, dict):
                        if beliefs.get('degree'):
                            st.session_state.user_context["degree_level"] = beliefs['degree']
                        if beliefs.get('program'):
                            st.session_state.user_context["program"] = beliefs['program']
                        if beliefs.get('semester'):
                            st.session_state.user_context["semester"] = beliefs['semester']
                    # Handle object format (legacy)
                    elif hasattr(beliefs, 'degree'):
                        if beliefs.degree:
                            st.session_state.user_context["degree_level"] = beliefs.degree
                        if beliefs.program:
                            st.session_state.user_context["program"] = beliefs.program
                        if beliefs.semester:
                            st.session_state.user_context["semester"] = beliefs.semester
                
                # Display sources if available
                if sources:
                    with sources_placeholder.expander("📚 Quellen anzeigen"):
                        for src in sources:
                            page_info = f", Seite {src['page']}" if src['page'] != 'n/a' else ""
                            st.markdown(f"• **{src['doc']}**{page_info}")
                
                # Show debug info in expander
                try:
                    config = load_config()
                    if config.logging.level == "DEBUG":
                        with debug_placeholder.expander("🔍 Debug Info"):
                            st.write(f"**Intention:** {intent}")
                            st.write(f"**Confidence:** {confidence:.2f}")
                            if metadata.get('reasoning'):
                                st.write(f"**Reasoning:** {metadata['reasoning'][:100]}...")
                            if beliefs:
                                if isinstance(beliefs, dict):
                                    st.write(f"**Beliefs:** {beliefs}")
                                else:
                                    st.write(f"**Beliefs:** {beliefs.degree} {beliefs.program} Semester {beliefs.semester}")
                except:
                    pass
                
                # Add assistant message to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "sources": sources
                })
                
            except Exception as e:
                error_msg = f"❌ Fehler: {str(e)}"
                message_placeholder.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })


if __name__ == "__main__":
    main()
