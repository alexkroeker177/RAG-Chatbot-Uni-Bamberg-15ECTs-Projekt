"""
Conv-BDI Framework Implementation.

Based on "Conv-BDI: An Extension of the BDI Framework for Conversational Agents"
(Walker et al., 2025)

This module defines the core components of the Conv-BDI framework:
- Purpose: The high-level reason for the agent's existence
- Guidelines: Behavioral constraints for the agent
- Beliefs: The agent's knowledge state (static + dynamic)
- Desires: All potential goals congruent with the Purpose
- Intentions: The specific "Desire in Focus" for the current turn
- Actions: Discrete operations to fulfill Intentions
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any


# =============================================================================
# PURPOSE
# =============================================================================
# The high-level reason for the agent's existence and scope.
# This defines WHAT the agent is and WHY it exists.

PURPOSE = """Du bist ein Informationsassistent für Prüfungsordnungen und Studienregelungen des Prüfungsamt der Fakultät WIAI an der Universität Bamberg.

DEIN EINZIGER ZWECK:
- Fragen zu Prüfungsordnungen, Studienordnungen und offiziellen Regelungen der Uni Bamberg beantworten
- Bei Fragen außerhalb deines Wissens an die zuständige Stelle MIT LINK verweisen

DU BIST SPEZIALISIERT AUF:
- Prüfungsordnungen der Fakultät WIAI (Wirtschaftsinformatik und Angewandte Informatik)
- Bachelor- und Masterstudiengänge (siehe Abkürzungen unten)
- Allgemeine Prüfungsordnung (APO) der Universität Bamberg

STUDIENGÄNGE UND ABKÜRZUNGEN:
- AI = Angewandte Informatik (NICHT Artificial Intelligence!)
- WI, WInf = Wirtschaftsinformatik
- KI, DS, KI & DS = Künstliche Intelligenz & Data Science
- IISM = International Information Systems Management
- IRD = Interaction Research and Design
- ISSS = International Software Systems Science

DU BIST KEIN:
- Allgemeiner Chatbot für Smalltalk oder allgemeine Fragen
- Persönlicher Berater für Lebens- oder Karrierefragen
- Suchmaschine für allgemeine Informationen"""


# =============================================================================
# DEPARTMENTS (Single Source of Truth)
# =============================================================================
# All department contact information in one place.

DEPARTMENTS = {
    "pruefungsamt": {
        "name": "Prüfungsamt",
        "url": "https://www.uni-bamberg.de/pruefungsamt/",
        "keywords": ["Prüfungsanmeldung", "Zeugnis", "Urkunde", "Noten", "Notenverbuchung", "Modulverschiebungen", "technische Probleme bei Anmeldung"]
    },
    "studierendenkanzlei": {
        "name": "Studierendenkanzlei",
        "url": "https://www.uni-bamberg.de/studierendenkanzlei/",
        "keywords": ["Urlaubssemester", "Teilzeitstudium", "Einschreibung", "Exmatrikulation", "Studienbescheinigungen", "Beurlaubung"]
    },
    "fachstudienberatung": {
        "name": "Fachstudienberatung",
        "url": "https://www.uni-bamberg.de/studienberatung/fachstudienberatung/",
        "keywords": ["Modulwahl", "Studienplanung", "Studienverlaufsberatung", "Anrechnung", "Auslandsstudium", "Anerkennung"]
    },
    "pruefungsausschuss": {
        "name": "Prüfungsausschuss",
        "url": "https://www.uni-bamberg.de/wiai/pruefungsausschuesse/",
        "keywords": ["Härtefallregelungen", "Widersprüche", "Ausnahmegenehmigungen", "Auslegung der Prüfungsordnung"]
    }
}


def format_departments_for_prompt() -> str:
    """
    Generate department routing text from DEPARTMENTS.

    Returns:
        Formatted string for prompt inclusion
    """
    lines = []
    for dept in DEPARTMENTS.values():
        keywords_str = ", ".join(dept["keywords"])
        lines.append(f"- {keywords_str} → {dept['name']}: {dept['url']}")
    return "\n".join(lines)


# =============================================================================
# PROGRAM MAPPINGS (Single Source of Truth)
# =============================================================================
# Maps abbreviations and filename patterns to canonical program names.
# Used by ingestion (filename parsing) and retrieval (query augmentation, re-ranking).
#
# IMPORTANT: "AI" at Bamberg = "Angewandte Informatik" NOT "Artificial Intelligence"!

PROGRAM_MAPPINGS: Dict[str, Dict[str, Any]] = {
    "ai": {
        "canonical": "Angewandte Informatik",
        "short": "ai",
        "aliases": ["angewandte informatik", "applied informatics"],
        "filename_patterns": ["angewandte_informatik", "angewandte-informatik", "ai"]
    },
    "wi": {
        "canonical": "Wirtschaftsinformatik",
        "short": "wi",
        "aliases": ["winf", "wirtschaftsinformatik", "business informatics"],
        "filename_patterns": ["wirtschaftsinformatik", "wi"]
    },
    "inf": {
        "canonical": "Informatik",
        "short": "inf",
        "aliases": ["informatik", "computer science"],
        "filename_patterns": ["informatik"]
    },
    "ki_ds": {
        "canonical": "Künstliche Intelligenz und Data Science",
        "short": "ki_ds",
        "aliases": ["ki", "ds", "ki & ds", "data science", "künstliche intelligenz"],
        "filename_patterns": ["ki_ds", "data_science", "kuenstliche_intelligenz"]
    },
    "iism": {
        "canonical": "International Information Systems Management",
        "short": "iism",
        "aliases": ["international information systems management"],
        "filename_patterns": ["iism", "international_information_systems"]
    },
    "ird": {
        "canonical": "Interaction Research and Design",
        "short": "ird",
        "aliases": ["interaction research and design"],
        "filename_patterns": ["ird", "interaction_research", "interaction-research"]
    },
    "isss": {
        "canonical": "International Software Systems Science",
        "short": "isss",
        "aliases": ["international software systems science", "software systems science"],
        "filename_patterns": ["isss", "software_systems_science"]
    },
    "cith": {
        "canonical": "Computing in the Humanities",
        "short": "cith",
        "aliases": ["computing in the humanities", "digital humanities"],
        "filename_patterns": ["computing_in_the_humanities", "cith"]
    },
    "general": {
        "canonical": "Allgemein",
        "short": "general",
        "aliases": ["apo", "allgemeine prüfungsordnung", "allgemein"],
        "filename_patterns": ["apo", "allgemeine"]
    }
}


def get_program_from_name(name: str) -> Optional[Dict[str, Any]]:
    """
    Look up a program by any of its names/aliases.

    Args:
        name: Program name, abbreviation, or alias (case-insensitive)

    Returns:
        Program mapping dict or None if not found
    """
    name_lower = name.lower().strip()

    for key, mapping in PROGRAM_MAPPINGS.items():
        # Check short form
        if name_lower == mapping["short"]:
            return mapping
        # Check canonical name
        if name_lower == mapping["canonical"].lower():
            return mapping
        # Check aliases
        if name_lower in [a.lower() for a in mapping["aliases"]]:
            return mapping

    return None


def normalize_program_name(name: str) -> tuple:
    """
    Normalize a program name to canonical and short forms.

    Args:
        name: Program name in any form

    Returns:
        Tuple of (canonical_name, short_form) or (name, name.lower()) if not found
    """
    mapping = get_program_from_name(name)
    if mapping:
        return mapping["canonical"], mapping["short"]
    return name, name.lower().replace(" ", "_")


# =============================================================================
# GUIDELINES
# =============================================================================
# Behavioral constraints that apply regardless of the specific Intention.
# These define HOW the agent should behave.

class GuidelineCategory(Enum):
    """Categories of behavioral guidelines."""
    GROUNDING = "grounding"      # Answer based on retrieved context only
    TONE = "tone"                # Communication style
    FORMATTING = "formatting"    # Output format requirements
    SAFETY = "safety"            # Out-of-scope handling
    LANGUAGE = "language"        # Language requirements


@dataclass
class Guideline:
    """A single behavioral guideline."""
    category: GuidelineCategory
    rule: str
    priority: int = 1  # Higher = more important


GUIDELINES: List[Guideline] = [
    # Grounding Guidelines (highest priority)
    Guideline(
        category=GuidelineCategory.GROUNDING,
        rule="Beantworte Fragen AUSSCHLIESSLICH basierend auf dem bereitgestellten Kontext. Wenn die Antwort NICHT EINDEUTIG im Kontext steht, sage: 'Diese Information finde ich nicht in den verfügbaren Dokumenten.'",
        priority=10
    ),
    Guideline(
        category=GuidelineCategory.GROUNDING,
        rule="Erfinde KEINE Informationen. Spekuliere NICHT. Rate NICHT. Wenn du unsicher bist, sage es klar.",
        priority=10
    ),
    Guideline(
        category=GuidelineCategory.GROUNDING,
        rule="Sage 'Ich weiß es nicht' oder 'Das kann ich nicht beantworten' wenn: (1) Die Info nicht im Kontext ist, (2) Die Frage unklar ist, (3) Du dir nicht sicher bist.",
        priority=10
    ),
    
    # Scope Guidelines (very high priority)
    Guideline(
        category=GuidelineCategory.SAFETY,
        rule="KEIN Smalltalk, KEINE allgemeinen Gespräche. Antworte NUR auf Fragen zu Prüfungsordnungen, Studienregelungen und universitären Prozessen.",
        priority=9
    ),
    Guideline(
        category=GuidelineCategory.SAFETY,
        rule="Bei Fragen außerhalb deines Wissensbereichs: Verweise IMMER mit konkretem LINK an die zuständige Stelle.",
        priority=9
    ),
    Guideline(
        category=GuidelineCategory.SAFETY,
        rule="Bei sensiblen Themen (Härtefälle, Widersprüche) verweise an den Prüfungsausschuss.",
        priority=9
    ),
    # Note: Department routing info is now in DEPARTMENTS dict (single source of truth)

    # Tone Guidelines
    Guideline(
        category=GuidelineCategory.TONE,
        rule="Sei sachlich, präzise und professionell. Keine übertriebene Freundlichkeit oder Emojis.",
        priority=5
    ),
    Guideline(
        category=GuidelineCategory.TONE,
        rule="Verwende klare, verständliche Sprache ohne unnötigen Fachjargon.",
        priority=5
    ),
    
    # Formatting Guidelines
    Guideline(
        category=GuidelineCategory.FORMATTING,
        rule="Halte Antworten kurz und präzise. Keine langen Einleitungen.",
        priority=4
    ),
    Guideline(
        category=GuidelineCategory.FORMATTING,
        rule="Gib KEINE Quellenangaben in der Antwort - diese werden automatisch angezeigt.",
        priority=4
    ),
    
    # Language Guidelines
    Guideline(
        category=GuidelineCategory.LANGUAGE,
        rule="Antworte auf Deutsch, es sei denn der Nutzer schreibt auf Englisch.",
        priority=3
    ),
]


def format_guidelines(categories: Optional[List[GuidelineCategory]] = None) -> str:
    """
    Format guidelines for prompt inclusion.
    
    Args:
        categories: Optional filter for specific categories
        
    Returns:
        Formatted guidelines string
    """
    filtered = GUIDELINES
    if categories:
        filtered = [g for g in GUIDELINES if g.category in categories]
    
    # Sort by priority (descending)
    sorted_guidelines = sorted(filtered, key=lambda g: g.priority, reverse=True)
    
    lines = ["VERHALTENSREGELN:"]
    for i, g in enumerate(sorted_guidelines, 1):
        lines.append(f"{i}. {g.rule}")
    
    return "\n".join(lines)


# =============================================================================
# BELIEFS
# =============================================================================
# The agent's knowledge state: static (documents) + dynamic (user context)

@dataclass
class DynamicBeliefs:
    """
    Dynamic beliefs extracted from conversation.
    
    These are updated as the conversation progresses.
    """
    # User study context
    degree: Optional[str] = None          # Bachelor, Master
    program: Optional[str] = None         # Informatik, WI, etc.
    semester: Optional[int] = None        # Current semester
    
    # Conversation state
    current_topic: Optional[str] = None   # Current discussion topic
    clarification_needed: bool = False    # Whether clarification is needed
    
    # Metadata
    last_updated: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "degree": self.degree,
            "program": self.program,
            "semester": self.semester,
            "current_topic": self.current_topic,
            "clarification_needed": self.clarification_needed
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DynamicBeliefs':
        """Create from dictionary."""
        return cls(
            degree=data.get("degree"),
            program=data.get("program"),
            semester=data.get("semester"),
            current_topic=data.get("current_topic"),
            clarification_needed=data.get("clarification_needed", False)
        )
    
    def merge(self, other: 'DynamicBeliefs') -> 'DynamicBeliefs':
        """Merge with another beliefs object, preferring non-None values."""
        return DynamicBeliefs(
            degree=other.degree if other.degree is not None else self.degree,
            program=other.program if other.program is not None else self.program,
            semester=other.semester if other.semester is not None else self.semester,
            current_topic=other.current_topic if other.current_topic is not None else self.current_topic,
            clarification_needed=other.clarification_needed,
            last_updated=datetime.now()
        )
    
    def format_for_prompt(self) -> str:
        """Format beliefs for prompt inclusion."""
        parts = []
        
        if self.degree:
            parts.append(f"Abschluss: {self.degree}")
        if self.program:
            parts.append(f"Studiengang: {self.program}")
        if self.semester:
            parts.append(f"Semester: {self.semester}")
        if self.current_topic:
            parts.append(f"Aktuelles Thema: {self.current_topic}")
        
        if parts:
            return "Bekannter Nutzerkontext:\n" + "\n".join(f"- {p}" for p in parts)
        return "Kein Nutzerkontext bekannt."


# =============================================================================
# INTENTIONS
# =============================================================================
# The specific "Desire in Focus" committed to for the current turn.

class Intention(Enum):
    """
    Committed intentions that trigger specific action sequences.
    
    An Intention is a Desire that has been selected for execution.
    """
    # RAG-based intentions (require retrieval)
    INFORMATION_RETRIEVAL = "information_retrieval"
    
    # Direct response intentions (no retrieval)
    CONVERSATIONAL_RESPONSE = "conversational_response"
    
    # Routing intentions
    DEPARTMENT_ROUTING = "department_routing"
    
    # Clarification intentions
    CLARIFICATION_REQUEST = "clarification_request"


@dataclass
class IntentionResult:
    """
    Result of intention selection.
    """
    intention: Intention
    confidence: float
    updated_beliefs: Optional[DynamicBeliefs] = None
    reasoning: str = ""
    latency: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "intention": self.intention.value,
            "confidence": self.confidence,
            "updated_beliefs": self.updated_beliefs.to_dict() if self.updated_beliefs else None,
            "reasoning": self.reasoning,
            "latency": self.latency
        }


