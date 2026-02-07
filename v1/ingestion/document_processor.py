"""
Document processing for PDFs, FAQs, and department information.

Extracts text, applies preprocessing, and creates LangChain Document objects.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from langchain_core.documents import Document
from pypdf import PdfReader

from v1.core.logger import setup_logger
from v1.ingestion.preprocessing import (
    preprocess_text,
    detect_language,
    extract_section_title
)
from v1.generation.conv_bdi import PROGRAM_MAPPINGS, normalize_program_name


logger = setup_logger(__name__)


def _extract_program_from_filename(filename: str) -> Tuple[str, str, str]:
    """
    Extract degree level and program from PDF filename.

    Filename patterns:
    - "Bachelor_Angewandte_Informatik_Studienordnung-..."
    - "Master_Wirtschaftsinformatik_Studienordnung-..."
    - "Bachelorstudiengang_International_Information_Systems..."
    - "Allgemeine_Prüfungsordnung..."

    Args:
        filename: PDF filename (without path)

    Returns:
        Tuple of (degree, program_short, program_canonical)
    """
    filename_lower = filename.lower()

    # Extract degree level
    degree = "general"
    if "bachelor" in filename_lower:
        degree = "Bachelor"
    elif "master" in filename_lower:
        degree = "Master"

    # Check APO/general FIRST (before other patterns that might match substrings)
    if "apo" in filename_lower or "allgemeine" in filename_lower:
        return degree, "general", "Allgemein"

    # Check for specific longer patterns first (to avoid false positives)
    # KI & DS: "Künstliche Intelligenz" and "Data Science"
    if ("künstliche intelligenz" in filename_lower or "kuenstliche intelligenz" in filename_lower
            or "data science" in filename_lower):
        return degree, "ki_ds", "Künstliche Intelligenz und Data Science"

    # ISSS: "Software Systems Science" or "isss"
    if "software systems science" in filename_lower or "isss" in filename_lower:
        return degree, "isss", "International Software Systems Science"

    # AI: "Angewandte Informatik" (check before generic "informatik")
    if "angewandte" in filename_lower and "informatik" in filename_lower:
        return degree, "ai", "Angewandte Informatik"

    # WI: "Wirtschaftsinformatik" (check before generic "informatik")
    if "wirtschaftsinformatik" in filename_lower:
        return degree, "wi", "Wirtschaftsinformatik"

    # Try to match other programs by filename patterns
    for prog_key, mapping in PROGRAM_MAPPINGS.items():
        # Skip already-handled cases
        if prog_key in ["general", "isss", "ai", "wi", "inf"]:
            continue

        for pattern in mapping["filename_patterns"]:
            # Normalize pattern for matching
            pattern_normalized = pattern.lower().replace(" ", "_").replace("-", "_")
            pattern_with_hyphens = pattern.lower().replace(" ", "-").replace("_", "-")
            pattern_with_spaces = pattern.lower()

            if (pattern_normalized in filename_lower or
                pattern_with_hyphens in filename_lower or
                pattern_with_spaces in filename_lower):
                return degree, mapping["short"], mapping["canonical"]

    # Generic "Informatik" (pure CS, not AI or WI)
    if "informatik" in filename_lower:
        if "angewandte" not in filename_lower and "wirtschafts" not in filename_lower:
            return degree, "inf", "Informatik"

    # Default to general
    return degree, "general", "Allgemein"


class DocumentProcessor:
    """Processes various document types into LangChain Documents."""
    
    def __init__(self):
        """Initialize document processor."""
        self.processed_count = 0
        self.error_count = 0
    
    def process_pdf(
        self,
        pdf_path: Path,
        program: Optional[str] = None
    ) -> List[Document]:
        """
        Extract text from PDF and create Document objects.

        Args:
            pdf_path: Path to PDF file
            program: Program category (optional, auto-detected from filename if None)

        Returns:
            List of Document objects (one per page)
        """
        documents = []

        try:
            logger.info(f"Processing PDF: {pdf_path.name}")

            # Auto-detect program and degree from filename if not provided
            if program is None:
                degree, program_short, program_canonical = _extract_program_from_filename(pdf_path.name)
                logger.info(f"  Auto-detected: {degree} {program_canonical} ({program_short})")
            else:
                # Normalize provided program name
                program_canonical, program_short = normalize_program_name(program)
                degree = "general"  # Cannot infer degree if program manually specified

            reader = PdfReader(str(pdf_path))

            for page_num, page in enumerate(reader.pages, start=1):
                try:
                    # Extract text
                    raw_text = page.extract_text()

                    if not raw_text or len(raw_text.strip()) < 50:
                        logger.debug(f"Skipping page {page_num} (too short or empty)")
                        continue

                    # Preprocess text
                    text = preprocess_text(raw_text)

                    # Detect language
                    lang = detect_language(text, default="de")

                    # Try to extract section title
                    section_title = extract_section_title(text) or f"Page {page_num}"

                    # Create metadata with program info for belief-aware retrieval
                    metadata = {
                        "program": program_short,              # Short form for filtering
                        "program_canonical": program_canonical, # Full name for display/matching
                        "degree": degree,                       # Bachelor/Master/general
                        "doc": pdf_path.name,
                        "page": str(page_num),
                        "section_title": section_title,
                        "source_type": "regulation",
                        "topic": "general",
                        "lang": lang
                    }

                    # Create Document
                    doc = Document(
                        page_content=text,
                        metadata=metadata
                    )
                    documents.append(doc)

                except Exception as e:
                    logger.error(f"Error processing page {page_num} of {pdf_path.name}: {e}")
                    self.error_count += 1

            logger.info(f"Extracted {len(documents)} pages from {pdf_path.name}")
            self.processed_count += 1

        except Exception as e:
            logger.error(f"Failed to process PDF {pdf_path.name}: {e}")
            self.error_count += 1

        return documents
    
    def process_faq(self, faq_file: Path) -> List[Document]:
        """
        Process FAQ JSON file into Document objects.
        
        Expected JSON structure:
        {
            "faq_data": [
                {
                    "id": 1,
                    "topic": "...",
                    "question": "...",
                    "answer": "..."
                }
            ]
        }
        
        Args:
            faq_file: Path to FAQ JSON file
            
        Returns:
            List of Document objects
        """
        documents = []
        
        try:
            logger.info(f"Processing FAQ file: {faq_file.name}")
            
            with open(faq_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            faq_entries = data.get("faq_data", [])
            
            for entry in faq_entries:
                try:
                    question = entry.get("question", "")
                    answer = entry.get("answer", "")
                    topic = entry.get("topic", "general")
                    faq_id = entry.get("id", "unknown")
                    
                    if not question or not answer:
                        logger.warning(f"Skipping FAQ entry {faq_id}: missing question or answer")
                        continue
                    
                    # Combine question and answer
                    content = f"Frage: {question}\n\nAntwort: {answer}"
                    
                    # Preprocess
                    content = preprocess_text(content)
                    
                    # Detect language
                    lang = detect_language(content, default="de")
                    
                    # Create metadata
                    metadata = {
                        "program": "general",
                        "doc": "university_faq",
                        "page": "n/a",
                        "section_title": topic,
                        "source_type": "faq",
                        "topic": topic,
                        "question_id": str(faq_id),
                        "lang": lang
                    }
                    
                    # Create Document
                    doc = Document(
                        page_content=content,
                        metadata=metadata
                    )
                    documents.append(doc)
                    
                except Exception as e:
                    logger.error(f"Error processing FAQ entry: {e}")
                    self.error_count += 1
            
            logger.info(f"Processed {len(documents)} FAQ entries")
            self.processed_count += 1
            
        except Exception as e:
            logger.error(f"Failed to process FAQ file {faq_file.name}: {e}")
            self.error_count += 1
        
        return documents
    
    def process_departments(self, departments_file: Path) -> List[Document]:
        """
        Process department routing JSON into Document objects.
        
        Expected JSON structure:
        {
            "departments": [
                {
                    "name": "...",
                    "name_en": "...",
                    "responsibilities": [...],
                    "example_questions": [...],
                    "website": "..."
                }
            ]
        }
        
        Args:
            departments_file: Path to departments JSON file
            
        Returns:
            List of Document objects
        """
        documents = []
        
        try:
            logger.info(f"Processing departments file: {departments_file.name}")
            
            with open(departments_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            departments = data.get("departments", [])
            
            for dept in departments:
                try:
                    name = dept.get("name", "")
                    name_en = dept.get("name_en", "")
                    responsibilities = dept.get("responsibilities", [])
                    examples = dept.get("example_questions", [])
                    website = dept.get("website", "")
                    
                    if not name or not responsibilities:
                        logger.warning(f"Skipping department: missing name or responsibilities")
                        continue
                    
                    # Create searchable content
                    content_parts = [
                        f"Abteilung: {name} ({name_en})",
                        f"Zuständig für: {', '.join(responsibilities)}",
                        f"Beispielfragen: {' | '.join(examples)}",
                        f"Website: {website}"
                    ]
                    content = "\n\n".join(content_parts)
                    
                    # Preprocess
                    content = preprocess_text(content)
                    
                    # Create metadata
                    metadata = {
                        "program": "general",
                        "doc": "departments.json",
                        "page": "n/a",
                        "section_title": name,
                        "source_type": "department_info",
                        "topic": "routing",
                        "department_name": name,
                        "lang": "de"
                    }
                    
                    # Create Document
                    doc = Document(
                        page_content=content,
                        metadata=metadata
                    )
                    documents.append(doc)
                    
                except Exception as e:
                    logger.error(f"Error processing department: {e}")
                    self.error_count += 1
            
            logger.info(f"Processed {len(documents)} departments")
            self.processed_count += 1
            
        except Exception as e:
            logger.error(f"Failed to process departments file {departments_file.name}: {e}")
            self.error_count += 1
        
        return documents
    
    def get_stats(self) -> Dict[str, int]:
        """Get processing statistics."""
        return {
            "processed": self.processed_count,
            "errors": self.error_count
        }


# Example usage
if __name__ == "__main__":
    processor = DocumentProcessor()
    
    # Test with FAQ file if it exists
    faq_path = Path("QA.json")
    if faq_path.exists():
        docs = processor.process_faq(faq_path)
        print(f"\nProcessed {len(docs)} FAQ documents")
        if docs:
            print(f"Sample FAQ document:")
            print(f"  Content: {docs[0].page_content[:200]}...")
            print(f"  Metadata: {docs[0].metadata}")
    
    # Test with departments file if it exists
    dept_path = Path("departments.json")
    if dept_path.exists():
        docs = processor.process_departments(dept_path)
        print(f"\nProcessed {len(docs)} department documents")
        if docs:
            print(f"Sample department document:")
            print(f"  Content: {docs[0].page_content[:200]}...")
            print(f"  Metadata: {docs[0].metadata}")
    
    print(f"\nStats: {processor.get_stats()}")
