"""
Conv-BDI Conversational Response Generator.

Based on "Conv-BDI: An Extension of the BDI Framework for Conversational Agents"
(Walker et al., 2025)

Generates natural, context-aware responses for non-retrieval Intentions:
- CONVERSATIONAL_RESPONSE: Greetings, acknowledgments, context-setting
- DEPARTMENT_ROUTING: Directing users to appropriate departments
- CLARIFICATION_REQUEST: Asking for more information
"""

from typing import Optional
from langchain_community.llms import Ollama

from v1.generation.conv_bdi import (
    Intention,
    DynamicBeliefs,
    PURPOSE,
    DEPARTMENTS  # Single source of truth for department info
)
from v1.generation.prompts import create_conversational_prompt
from v1.core.logger import setup_logger


logger = setup_logger(__name__)


class ConversationalResponses:
    """
    Conv-BDI Conversational Response Generator.
    
    Handles all non-retrieval Intentions by generating appropriate
    responses based on the current Beliefs and selected Intention.
    """
    
    def __init__(self, llm: Optional[Ollama] = None):
        """
        Initialize conversational response generator.
        
        Args:
            llm: Ollama LLM instance for generating responses
        """
        self.llm = llm
    
    def generate_response(
        self,
        message: str,
        beliefs: Optional[DynamicBeliefs] = None,
        intention: Intention = Intention.CONVERSATIONAL_RESPONSE,
        conversation_history: Optional[list] = None
    ) -> str:
        """
        Generate response based on Conv-BDI components.
        
        Args:
            message: User message
            beliefs: Current dynamic beliefs about user
            intention: Selected intention for this turn
            conversation_history: Recent conversation turns for context
            
        Returns:
            Generated response string
        """
        if not self.llm:
            logger.warning("No LLM available for conversational response, using fallback")
            return self._get_fallback_response(intention, beliefs)
        
        try:
            # Create Conv-BDI structured prompt WITH conversation history
            prompt = create_conversational_prompt(
                message=message,
                beliefs=beliefs,
                intention=intention.value,
                conversation_history=conversation_history
            )
            
            # Generate response
            logger.debug(f"Generating {intention.value} response for: {message[:50]}...")
            response = self.llm.invoke(prompt)
            
            # Clean up response
            response = self._clean_response(response)
            
            logger.debug(f"Generated response: {response[:100]}...")
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating conversational response: {e}")
            return self._get_fallback_response(intention, beliefs)
    
    def _clean_response(self, response: str) -> str:
        """
        Clean LLM response.
        
        Args:
            response: Raw LLM response
            
        Returns:
            Cleaned response string
        """
        response = response.strip()
        
        # Remove any thinking tags
        if "<think>" in response:
            parts = response.split("</think>")
            if len(parts) > 1:
                response = parts[1].strip()
        
        # Remove markdown formatting if present
        if response.startswith("```"):
            lines = response.split("\n")
            response = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
        
        return response.strip()
    
    def _get_fallback_response(
        self,
        intention: Intention,
        beliefs: Optional[DynamicBeliefs] = None
    ) -> str:
        """
        Get fallback response when LLM is unavailable.
        
        Args:
            intention: Selected intention
            beliefs: Current beliefs
            
        Returns:
            Appropriate fallback response with links
        """
        fallback_responses = {
            Intention.CONVERSATIONAL_RESPONSE: 
                "Wie kann ich dir bei Fragen zu Prüfungsordnungen helfen?",
            
            Intention.DEPARTMENT_ROUTING:
                "Für diese Anfrage wende dich bitte an die zuständige Stelle:\n"
                "- Prüfungsamt: https://www.uni-bamberg.de/pruefungsamt/\n"
                "- Studierendenkanzlei: https://www.uni-bamberg.de/studierendenkanzlei/\n"
                "- Fachstudienberatung: https://www.uni-bamberg.de/studienberatung/fachstudienberatung/",
            
            Intention.CLARIFICATION_REQUEST:
                "Bitte formuliere deine Frage zu Prüfungsordnungen genauer.",
        }
        
        return fallback_responses.get(
            intention,
            "Wie kann ich dir bei Fragen zu Prüfungsordnungen helfen?"
        )
    
    @staticmethod
    def get_response(
        message: str,
        context: Optional[DynamicBeliefs] = None,
        has_new_context: bool = False,
        llm: Optional[Ollama] = None,
        intention: Intention = Intention.CONVERSATIONAL_RESPONSE,
        conversation_history: Optional[list] = None
    ) -> str:
        """
        Static method for backward compatibility.
        
        Get appropriate conversational response based on message and context.
        
        Args:
            message: User message
            context: Current user context (DynamicBeliefs)
            has_new_context: Whether new context was extracted
            llm: Ollama LLM instance
            intention: Selected intention
            conversation_history: Recent conversation turns for context
            
        Returns:
            Appropriate conversational response
        """
        generator = ConversationalResponses(llm)
        return generator.generate_response(message, context, intention, conversation_history)


# =============================================================================
# DEPARTMENT ROUTING
# =============================================================================
# Note: DEPARTMENTS dict is imported from conv_bdi.py (single source of truth)


def get_department_info(topic: Optional[str] = None) -> str:
    """
    Get department contact information with website link.

    Args:
        topic: Optional topic to find relevant department

    Returns:
        Formatted department information with link
    """
    if topic:
        topic_lower = topic.lower()
        for dept_id, dept in DEPARTMENTS.items():
            for keyword in dept["keywords"]:
                if keyword.lower() in topic_lower or topic_lower in keyword.lower():
                    return f"Für {topic} ist die {dept['name']} zuständig: {dept['url']}"

    # Default to Prüfungsamt
    dept = DEPARTMENTS["pruefungsamt"]
    return f"Wende dich bitte an das {dept['name']}: {dept['url']}"


def get_all_departments_info() -> str:
    """
    Get formatted list of all departments with links.

    Returns:
        Formatted string with all departments and their links
    """
    lines = ["Zuständige Stellen:"]
    for dept_id, dept in DEPARTMENTS.items():
        lines.append(f"- {dept['name']}: {dept['url']}")
    return "\n".join(lines)


# =============================================================================
# BACKWARD COMPATIBILITY
# =============================================================================
# Support for old UserContext type

from v1.generation.user_context import UserContext


def _convert_user_context_to_beliefs(context: Optional[UserContext]) -> Optional[DynamicBeliefs]:
    """Convert legacy UserContext to DynamicBeliefs."""
    if not context:
        return None
    
    return DynamicBeliefs(
        degree=context.degree,
        program=context.program,
        semester=context.semester
    )
