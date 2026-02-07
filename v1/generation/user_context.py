"""
User context management for personalized responses.

Tracks user study information (degree, program, semester) across conversation turns.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict


@dataclass
class UserContext:
    """User study context information."""
    
    degree: Optional[str] = None  # Bachelor, Master
    program: Optional[str] = None  # Informatik, Angewandte Informatik, etc.
    semester: Optional[int] = None
    last_updated: datetime = field(default_factory=datetime.now)
    
    def is_complete(self) -> bool:
        """
        Check if all context fields are populated.
        
        Returns:
            True if degree, program, and semester are all set
        """
        return all([
            self.degree is not None,
            self.program is not None,
            self.semester is not None
        ])
    
    def merge(self, other: 'UserContext') -> 'UserContext':
        """
        Merge with another context, preferring non-None values.
        
        Args:
            other: Another UserContext to merge with
            
        Returns:
            New UserContext with merged values
        """
        return UserContext(
            degree=other.degree if other.degree is not None else self.degree,
            program=other.program if other.program is not None else self.program,
            semester=other.semester if other.semester is not None else self.semester,
            last_updated=datetime.now()
        )


class UserContextManager:
    """Manages user context storage and retrieval."""
    
    def __init__(self):
        """Initialize context storage (in-memory)."""
        self._contexts: Dict[str, UserContext] = {}
    
    def get_context(self, session_id: str) -> Optional[UserContext]:
        """
        Retrieve context for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            UserContext if exists, None otherwise
        """
        return self._contexts.get(session_id)
    
    def update_context(self, session_id: str, context: UserContext):
        """
        Update context for a session.
        
        Args:
            session_id: Session identifier
            context: New or updated context
        """
        existing = self._contexts.get(session_id)
        
        if existing:
            # Merge with existing context
            self._contexts[session_id] = existing.merge(context)
        else:
            # Store new context
            self._contexts[session_id] = context
    
    def format_context_prompt(self, context: Optional[UserContext]) -> str:
        """
        Format context for inclusion in prompts.
        
        Args:
            context: User context to format
            
        Returns:
            Formatted string for prompt enhancement
        """
        if not context:
            return ""
        
        parts = []
        
        if context.degree:
            parts.append(f"Abschluss: {context.degree}")
        
        if context.program:
            parts.append(f"Studiengang: {context.program}")
        
        if context.semester:
            parts.append(f"Semester: {context.semester}")
        
        if not parts:
            return ""
        
        return "Nutzerkontext: " + ", ".join(parts)
