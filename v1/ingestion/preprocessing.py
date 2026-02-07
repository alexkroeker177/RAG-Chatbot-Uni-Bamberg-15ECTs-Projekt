"""
Text preprocessing utilities for document ingestion.

Handles Unicode normalization, whitespace cleanup, and language detection.
"""

import re
import unicodedata
from typing import Optional

from langdetect import detect, LangDetectException

from v1.core.logger import setup_logger


logger = setup_logger(__name__)


def normalize_unicode(text: str) -> str:
    """
    Normalize text using Unicode NFKC (Compatibility Composition).
    
    This handles German umlauts and special characters properly:
    - ä, ö, ü, ß remain as single characters
    - Compatibility characters converted to canonical forms
    
    Args:
        text: Input text
        
    Returns:
        Normalized text
    """
    return unicodedata.normalize("NFKC", text)


def clean_whitespace(text: str) -> str:
    """
    Clean up excessive whitespace and newlines.
    
    - Collapses multiple spaces into single space
    - Collapses multiple newlines into double newline (paragraph break)
    - Strips leading/trailing whitespace
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    # Replace multiple spaces with single space
    text = re.sub(r" +", " ", text)
    
    # Replace multiple newlines with double newline (keep paragraph breaks)
    text = re.sub(r"\n{3,}", "\n\n", text)
    
    # Remove spaces at start/end of lines
    text = "\n".join(line.strip() for line in text.split("\n"))
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def remove_control_characters(text: str) -> str:
    """
    Remove control characters except newlines and tabs.
    
    Args:
        text: Input text
        
    Returns:
        Text without control characters
    """
    # Keep newlines (\n) and tabs (\t)
    return "".join(
        char for char in text
        if char in ("\n", "\t") or not unicodedata.category(char).startswith("C")
    )


def preprocess_text(text: str) -> str:
    """
    Apply full preprocessing pipeline to text.
    
    Steps:
    1. Unicode NFKC normalization
    2. Remove control characters
    3. Clean whitespace
    
    Args:
        text: Raw input text
        
    Returns:
        Preprocessed text
    """
    if not text:
        return ""
    
    # Apply preprocessing steps
    text = normalize_unicode(text)
    text = remove_control_characters(text)
    text = clean_whitespace(text)
    
    return text


def detect_language(text: str, default: str = "unknown") -> str:
    """
    Detect language of text using langdetect.
    
    Args:
        text: Input text
        default: Default language if detection fails
        
    Returns:
        ISO 639-1 language code (e.g., 'de', 'en') or default
    """
    if not text or len(text.strip()) < 10:
        # Too short for reliable detection
        return default
    
    try:
        lang = detect(text)
        logger.debug(f"Detected language: {lang}")
        return lang
    except LangDetectException as e:
        logger.debug(f"Language detection failed: {e}, using default '{default}'")
        return default


def extract_section_title(text: str) -> Optional[str]:
    """
    Extract section title from text (e.g., "§ 15 Prüfungsordnung").
    
    Looks for common patterns in German legal documents:
    - § followed by number
    - Artikel followed by number
    - Chapter/Kapitel headings
    
    Args:
        text: Text to search
        
    Returns:
        Section title if found, None otherwise
    """
    # Pattern for § sections
    section_pattern = r"§\s*\d+[a-z]?\s+[^\n]{1,100}"
    match = re.search(section_pattern, text, re.IGNORECASE)
    if match:
        return match.group(0).strip()
    
    # Pattern for Artikel
    article_pattern = r"Artikel\s+\d+[a-z]?\s+[^\n]{1,100}"
    match = re.search(article_pattern, text, re.IGNORECASE)
    if match:
        return match.group(0).strip()
    
    # Pattern for numbered chapters
    chapter_pattern = r"(?:Kapitel|Chapter)\s+\d+[:\.]?\s+[^\n]{1,100}"
    match = re.search(chapter_pattern, text, re.IGNORECASE)
    if match:
        return match.group(0).strip()
    
    return None


# Example usage and testing
if __name__ == "__main__":
    # Test Unicode normalization
    test_text = "Prüfungsordnung für Informatik\n\n\nMit  mehreren   Leerzeichen"
    print("Original:", repr(test_text))
    print("Preprocessed:", repr(preprocess_text(test_text)))
    print()
    
    # Test language detection
    german_text = "Dies ist ein deutscher Text über die Prüfungsordnung."
    english_text = "This is an English text about examination regulations."
    print(f"German text detected as: {detect_language(german_text)}")
    print(f"English text detected as: {detect_language(english_text)}")
    print()
    
    # Test section extraction
    section_text = "§ 15 Prüfungsordnung\nDieser Paragraph regelt..."
    print(f"Extracted section: {extract_section_title(section_text)}")
    
    # Test with German umlauts
    umlaut_text = "Über die Prüfung für Schüler"
    print(f"Umlauts preserved: {preprocess_text(umlaut_text)}")
