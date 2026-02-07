"""
Question variation generator using Ollama.

Generates 6 variations of each FAQ question for evaluation.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import requests

from v1.core.config import load_config, Config
from v1.core.logger import setup_logger


logger = setup_logger(__name__)


VARIATION_PROMPT = """Du bist ein Experte für Sprachvariation. Erstelle 6 verschiedene Formulierungen der folgenden Frage, die alle dasselbe bedeuten aber unterschiedlich formuliert sind.

Original-Frage: {question}
Thema: {topic}

Erstelle genau 6 Variationen:
1. FORMAL: Höfliche, formelle Formulierung (z.B. "Könnten Sie mir bitte erläutern...")
2. INFORMAL: Lockere, umgangssprachliche Formulierung (z.B. "Hey, muss ich eigentlich...")
3. KEYWORD: Nur Schlüsselwörter, Telegramm-Stil (z.B. "Immatrikulation Abschlussarbeit?")
4. VERBOSE: Ausführlich mit persönlichem Kontext (z.B. "Ich bin gerade dabei... und frage mich...")
5. SYNONYM: Ersetze Schlüsselbegriffe durch Synonyme
6. INVERSE: Frage aus umgekehrter Perspektive (z.B. "Was passiert wenn ich NICHT...")

WICHTIG: Antworte NUR mit einem validen JSON-Objekt, keine Erklärungen davor oder danach.

Ausgabeformat:
{{"formal": "...", "informal": "...", "keyword": "...", "verbose": "...", "synonym": "...", "inverse": "..."}}"""


class QuestionGenerator:
    """Generates question variations using Ollama."""

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize question generator.

        Args:
            config: Optional config object, loads from file if not provided
        """
        self.config = config or load_config()
        self.base_url = self.config.ollama.base_url
        self.model = self.config.evaluation.evaluation_model
        self.output_dir = Path(self.config.evaluation.output_directory)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _call_ollama(self, prompt: str) -> str:
        """
        Call Ollama API with a prompt.

        Args:
            prompt: The prompt to send

        Returns:
            Response text
        """
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 1024
            }
        }

        response = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=120
        )

        if response.status_code != 200:
            raise RuntimeError(f"Ollama API error: {response.status_code} - {response.text}")

        result = response.json()
        return result["message"]["content"]

    def _parse_variations(self, response_text: str) -> Dict[str, str]:
        """
        Parse JSON variations from API response.

        Args:
            response_text: Raw API response

        Returns:
            Dictionary of variation type -> question text
        """
        # Try to extract JSON from response
        text = response_text.strip()

        # Handle markdown code blocks
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        # Find JSON object
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            text = text[start:end]

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse variations JSON: {e}")
            logger.debug(f"Raw response: {response_text}")
            return {}

    def generate_variations(self, question: str, topic: str) -> Dict[str, str]:
        """
        Generate 6 variations of a question.

        Args:
            question: Original FAQ question
            topic: Question topic for context

        Returns:
            Dictionary with variation types as keys
        """
        prompt = VARIATION_PROMPT.format(question=question, topic=topic)

        try:
            response = self._call_ollama(prompt)
            variations = self._parse_variations(response)

            # Validate we have all expected types
            expected_types = ["formal", "informal", "keyword", "verbose", "synonym", "inverse"]
            for var_type in expected_types:
                if var_type not in variations:
                    logger.warning(f"Missing variation type: {var_type}")
                    variations[var_type] = question  # Fallback to original

            return variations

        except Exception as e:
            logger.error(f"Failed to generate variations: {e}")
            # Return original question as fallback for all types
            return {t: question for t in ["formal", "informal", "keyword", "verbose", "synonym", "inverse"]}

    def process_faq_file(self, faq_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Process all FAQ entries and generate variations.

        Args:
            faq_path: Path to FAQ JSON file, uses config default if not provided

        Returns:
            Dictionary with metadata and all questions
        """
        faq_path = faq_path or Path(self.config.data.faq_file)

        logger.info(f"Loading FAQ from: {faq_path}")

        with open(faq_path, "r", encoding="utf-8") as f:
            faq_data = json.load(f)

        faq_entries = faq_data.get("faq_data", [])
        logger.info(f"Found {len(faq_entries)} FAQ entries")

        all_questions = []
        question_id = 1

        for i, entry in enumerate(faq_entries):
            faq_id = entry.get("id", i + 1)
            topic = entry.get("topic", "Unknown")
            question = entry.get("question", "")
            answer = entry.get("answer", "")

            if not question:
                logger.warning(f"Skipping FAQ {faq_id}: no question")
                continue

            logger.info(f"Processing FAQ {faq_id}/{len(faq_entries)}: {question[:50]}...")

            # Add original question
            all_questions.append({
                "id": question_id,
                "original_faq_id": faq_id,
                "variation_type": "original",
                "topic": topic,
                "question": question,
                "ground_truth_answer": answer
            })
            question_id += 1

            # Generate and add variations
            variations = self.generate_variations(question, topic)

            for var_type, var_question in variations.items():
                all_questions.append({
                    "id": question_id,
                    "original_faq_id": faq_id,
                    "variation_type": var_type,
                    "topic": topic,
                    "question": var_question,
                    "ground_truth_answer": answer
                })
                question_id += 1

            # Rate limiting - be nice to the API
            time.sleep(0.5)

        result = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "source": str(faq_path),
                "total_questions": len(all_questions),
                "original_count": len(faq_entries),
                "variations_per_question": 6,
                "variation_types": ["original", "formal", "informal", "keyword", "verbose", "synonym", "inverse"]
            },
            "questions": all_questions
        }

        return result

    def save_questions(self, questions_data: Dict[str, Any], output_path: Optional[Path] = None) -> Path:
        """
        Save generated questions to JSON file.

        Args:
            questions_data: Questions data dictionary
            output_path: Output file path, uses default if not provided

        Returns:
            Path to saved file
        """
        output_path = output_path or self.output_dir / "evaluation_questions.json"

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(questions_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved {questions_data['metadata']['total_questions']} questions to {output_path}")
        return output_path

    def run(self, faq_path: Optional[Path] = None) -> Path:
        """
        Run the full question generation pipeline.

        Args:
            faq_path: Optional path to FAQ file

        Returns:
            Path to generated questions file
        """
        logger.info("Starting question variation generation...")
        questions_data = self.process_faq_file(faq_path)
        output_path = self.save_questions(questions_data)
        logger.info(f"Question generation complete: {output_path}")
        return output_path


def main():
    """Run question generator as standalone script."""
    generator = QuestionGenerator()
    output_path = generator.run()
    print(f"Generated questions saved to: {output_path}")


if __name__ == "__main__":
    main()
