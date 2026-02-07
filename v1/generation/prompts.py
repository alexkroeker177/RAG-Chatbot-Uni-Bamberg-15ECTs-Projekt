"""
Conv-BDI Structured Prompts for RAG Answer Generation.

Based on "Conv-BDI: An Extension of the BDI Framework for Conversational Agents"
(Walker et al., 2025)

Prompts are structured with explicit Conv-BDI components:
- PURPOSE: The agent's role and scope
- GUIDELINES: Behavioral constraints
- BELIEFS: Current knowledge state (context)
- INTENTION: Current goal
"""

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from typing import Optional, List, Dict

from v1.generation.conv_bdi import (
    format_departments_for_prompt,
    DynamicBeliefs
)


# =============================================================================
# RAG SYSTEM PROMPT (Simplified)
# =============================================================================

RAG_SYSTEM_PROMPT = """Du bist ein Informationsassistent für Prüfungsordnungen der Uni Bamberg (Fakultät WIAI).

REGELN:
1. Antworte NUR basierend auf den bereitgestellten Dokumenten
2. Wenn Info NICHT im Kontext: "Diese Information finde ich nicht in den Dokumenten."
3. KEINE Erfindungen, KEINE Spekulationen
4. Bei Unsicherheit: verweise an zuständige Stelle mit Link

ZUSTÄNDIGE STELLEN:
{departments}

FORMAT: Kurz, präzise, auf Deutsch. Keine Quellenangaben (automatisch angezeigt)."""


# =============================================================================
# RAG CONTEXT TEMPLATE (Simplified)
# =============================================================================

RAG_CONTEXT_TEMPLATE = """KONTEXT:
{context}

NUTZERINFO: {user_beliefs}

FRAGE: {question}

ANTWORT:"""


# =============================================================================
# INTENT CLASSIFICATION PROMPT (Simplified)
# =============================================================================
# Belief extraction removed - user info comes from obligatory form

INTENT_CLASSIFICATION_PROMPT = """Klassifiziere die Nachricht. Antworte NUR mit JSON.

INTENTIONEN:
- information_retrieval: Fragen zu Regeln, Fristen, ECTS, Modulen, Prüfungen
- conversational_response: Grüße, Danke, Verabschiedungen, Off-Topic
- department_routing: Explizite Frage nach Zuständigkeit/Kontakt ("An wen wende ich mich...")
- clarification_request: Unklare oder unvollständige Frage

IM ZWEIFEL: information_retrieval

BEISPIELE:
"Wie viele ECTS brauche ich?" → {{"intention": "information_retrieval", "confidence": 0.95}}
"Hallo" → {{"intention": "conversational_response", "confidence": 0.99}}
"An wen wende ich mich bei Problemen?" → {{"intention": "department_routing", "confidence": 0.95}}

Nachricht: "{message}"
JSON:"""


# =============================================================================
# CONVERSATIONAL RESPONSE PROMPT (Simplified)
# =============================================================================

CONVERSATIONAL_PROMPT = """Informationssystem für Prüfungsordnungen der Uni Bamberg. KEIN Smalltalk.

REAKTIONEN:
- Begrüßung: "Hallo! Wie kann ich bei Prüfungsordnungen helfen?"
- Danke: "Gerne! Bei weiteren Fragen bin ich da."
- Off-Topic: "Das kann ich nicht beantworten. Ich helfe bei Fragen zu Prüfungsordnungen."
- Verabschiedung: "Auf Wiedersehen!"

Nutzerkontext: {beliefs}
Nachricht: {message}

ANTWORT (1-2 Sätze):"""


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_rag_prompt() -> ChatPromptTemplate:
    """
    Create the main RAG prompt template.

    Returns:
        ChatPromptTemplate for RAG
    """
    system_content = RAG_SYSTEM_PROMPT.format(
        departments=format_departments_for_prompt()
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_content),
        ("human", RAG_CONTEXT_TEMPLATE)
    ])

    return prompt


def create_simple_prompt() -> PromptTemplate:
    """
    Create a simple prompt template (for non-chat models).

    Returns:
        PromptTemplate for RAG
    """
    system_content = RAG_SYSTEM_PROMPT.format(
        departments=format_departments_for_prompt()
    )

    full_template = f"{system_content}\n\n{RAG_CONTEXT_TEMPLATE}"

    prompt = PromptTemplate(
        template=full_template,
        input_variables=["user_beliefs", "context", "question"]
    )

    return prompt


def create_intent_classification_prompt(message: str) -> str:
    """
    Create intent classification prompt.

    Args:
        message: User message to classify

    Returns:
        Formatted prompt string
    """
    return INTENT_CLASSIFICATION_PROMPT.format(message=message)


def create_conversational_prompt(
    message: str,
    beliefs: Optional[DynamicBeliefs] = None,
    intention: str = "conversational_response",
    conversation_history: Optional[List[Dict]] = None
) -> str:
    """
    Create conversational response prompt.

    Args:
        message: User message
        beliefs: Current beliefs about user
        intention: Current intention type (unused, kept for compatibility)
        conversation_history: Recent conversation turns (unused, kept for compatibility)

    Returns:
        Formatted prompt string
    """
    beliefs_str = "Kein Nutzerkontext bekannt."
    if beliefs:
        beliefs_str = beliefs.format_for_prompt()

    return CONVERSATIONAL_PROMPT.format(
        beliefs=beliefs_str,
        message=message
    )


def format_context_with_sources(documents: list) -> str:
    """
    Format retrieved documents with source information.
    
    This represents the "static beliefs" from the knowledge base.
    
    Args:
        documents: List of retrieved Document objects
        
    Returns:
        Formatted context string
    """
    if not documents:
        return "Keine relevanten Dokumente gefunden."
    
    formatted_parts = []
    
    for i, doc in enumerate(documents, 1):
        meta = doc.metadata
        
        # Build source line
        source_parts = [f"[Dokument {i}]"]
        
        doc_name = meta.get('doc', 'unknown')
        source_parts.append(doc_name)
        
        page = meta.get('page', 'n/a')
        if page and page != 'n/a':
            source_parts.append(f"Seite {page}")
        
        section = meta.get('section_title')
        if section:
            source_parts.append(f"({section})")
        
        source_line = " | ".join(source_parts)
        
        # Add content
        formatted_parts.append(f"{source_line}\n{doc.page_content}\n")
    
    return "\n".join(formatted_parts)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    from langchain_core.documents import Document
    
    print("=== Conv-BDI RAG Prompt Template ===\n")
    
    # Test RAG prompt
    prompt = create_rag_prompt()
    print(f"Prompt type: {type(prompt)}")
    print(f"Input variables: {prompt.input_variables}")
    
    # Test context formatting
    print("\n=== Context Formatting (Static Beliefs) ===\n")
    
    test_docs = [
        Document(
            page_content="Der Bachelorstudiengang Informatik umfasst 180 ECTS-Punkte.",
            metadata={
                "doc": "Bachelor_Informatik.pdf",
                "page": "5",
                "section_title": "§ 2 Studienumfang",
                "source_type": "regulation"
            }
        )
    ]
    
    formatted_context = format_context_with_sources(test_docs)
    print(formatted_context)
    
    # Test intent classification prompt
    print("\n=== Intent Classification Prompt ===\n")

    intent_prompt = create_intent_classification_prompt("Wie viele ECTS brauche ich?")
    print(intent_prompt)

    # Test conversational prompt
    print("\n=== Conversational Prompt ===\n")

    beliefs = DynamicBeliefs(degree="Bachelor", program="Informatik")
    conv_prompt = create_conversational_prompt("Hallo!", beliefs)
    print(conv_prompt)
