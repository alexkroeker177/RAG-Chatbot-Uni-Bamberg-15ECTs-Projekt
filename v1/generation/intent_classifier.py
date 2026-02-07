"""
Intent Classification Module with Hybrid Keyword/LLM approach.

Uses fast keyword patterns for obvious cases (~30% of messages),
falls back to LLM for complex classification.
"""

import json
import re
import time
from hashlib import md5
from typing import Optional, List, Dict

from langchain_community.llms import Ollama

from v1.core.config import Config
from v1.core.logger import setup_logger
from v1.generation.conv_bdi import (
    Intention,
    IntentionResult,
    DynamicBeliefs
)
from v1.generation.prompts import create_intent_classification_prompt


logger = setup_logger(__name__)


# =============================================================================
# KEYWORD PATTERNS FOR FAST CLASSIFICATION
# =============================================================================

# Greetings - match simple greetings at start of message
GREETING_PATTERN = re.compile(
    r'^(hallo|hi|hey|guten\s*(tag|morgen|abend)|servus|moin)[\s!.,?]*$',
    re.IGNORECASE
)

# Farewells
FAREWELL_PATTERN = re.compile(
    r'^(tschüss|tschüs|bye|auf\s*wiedersehen|ciao|bis\s*dann)[\s!.,?]*$',
    re.IGNORECASE
)

# Thanks/Acknowledgments
THANKS_PATTERN = re.compile(
    r'^(danke|vielen\s*dank|ok(ay)?|alles\s*klar|verstanden|super|perfekt|gut)[\s!.,?]*$',
    re.IGNORECASE
)

# Question indicators (for information retrieval)
QUESTION_PATTERN = re.compile(
    r'(\?|^wie\s|^was\s|^wann\s|^wo\s|^wer\s|^welche|muss\s+ich|kann\s+ich|darf\s+ich|brauche\s+ich)',
    re.IGNORECASE
)


class IntentClassifier:
    """
    Hybrid Intent Classifier with keyword fast-path and LLM fallback.

    Uses pattern matching for obvious cases (~30% faster),
    falls back to LLM for complex messages.
    """

    def __init__(self, config: Config):
        """Initialize with lightweight LLM for classification."""
        self.config = config
        self.timeout = config.intent_classification.timeout
        self.confidence_threshold = config.intent_classification.confidence_threshold
        self.fallback_to_query = config.intent_classification.fallback_to_query

        # LLM classification cache (avoid repeated calls for same messages)
        self._llm_cache: Dict[str, IntentionResult] = {}
        self._cache_max_size = 256

        # Initialize lightweight LLM for classification
        try:
            self.llm = Ollama(
                base_url=config.ollama.base_url,
                model=config.intent_classification.model,
                temperature=0.0  # Deterministic for classification
            )
            logger.info(f"✓ Intent classifier initialized with {config.intent_classification.model}")
        except Exception as e:
            logger.error(f"Failed to initialize intent classifier: {e}")
            raise

    def _fast_classify(self, message: str) -> Optional[IntentionResult]:
        """
        Fast keyword-based classification for obvious cases.

        Returns IntentionResult if pattern matches, None otherwise.
        """
        msg = message.strip()

        # Check for simple greetings
        if GREETING_PATTERN.match(msg):
            logger.debug(f"Fast classify: greeting pattern matched")
            return IntentionResult(
                intention=Intention.CONVERSATIONAL_RESPONSE,
                confidence=0.99,
                reasoning="Pattern: greeting"
            )

        # Check for farewells
        if FAREWELL_PATTERN.match(msg):
            logger.debug(f"Fast classify: farewell pattern matched")
            return IntentionResult(
                intention=Intention.CONVERSATIONAL_RESPONSE,
                confidence=0.99,
                reasoning="Pattern: farewell"
            )

        # Check for thanks/acknowledgments
        if THANKS_PATTERN.match(msg):
            logger.debug(f"Fast classify: thanks pattern matched")
            return IntentionResult(
                intention=Intention.CONVERSATIONAL_RESPONSE,
                confidence=0.99,
                reasoning="Pattern: thanks"
            )

        # Check for clear questions (must have question indicator AND be substantial)
        if QUESTION_PATTERN.search(msg) and len(msg) > 15:
            logger.debug(f"Fast classify: question pattern matched")
            return IntentionResult(
                intention=Intention.INFORMATION_RETRIEVAL,
                confidence=0.90,
                reasoning="Pattern: question"
            )

        # No pattern matched, need LLM
        return None
    
    def classify(
        self,
        message: str,
        conversation_history: Optional[List[Dict]] = None,
        current_beliefs: Optional[DynamicBeliefs] = None
    ) -> IntentionResult:
        """
        Classify user message using hybrid keyword/LLM approach.

        First tries fast keyword patterns, then falls back to LLM.
        """
        start_time = time.time()

        # Try fast keyword classification first
        fast_result = self._fast_classify(message)
        if fast_result:
            fast_result.latency = time.time() - start_time
            logger.info(
                f"Fast classify: {fast_result.intention.value} "
                f"(confidence: {fast_result.confidence:.2f}, latency: {fast_result.latency:.3f}s)"
            )
            return fast_result

        # Check LLM cache
        cache_key = md5(message.encode()).hexdigest()
        if cache_key in self._llm_cache:
            cached = self._llm_cache[cache_key]
            cached.latency = time.time() - start_time
            logger.info(
                f"Cache hit: {cached.intention.value} "
                f"(confidence: {cached.confidence:.2f}, latency: {cached.latency:.3f}s)"
            )
            return cached

        # Fall back to LLM classification
        try:
            prompt = create_intent_classification_prompt(message)
            response_text = self.llm.invoke(prompt)
            result = self._parse_response(response_text)

            latency = time.time() - start_time
            result.latency = latency

            logger.info(
                f"LLM classify: {result.intention.value} "
                f"(confidence: {result.confidence:.2f}, latency: {latency:.2f}s)"
            )

            # Check confidence threshold - fallback to retrieval if uncertain
            if result.confidence < self.confidence_threshold and self.fallback_to_query:
                logger.warning(
                    f"Low confidence ({result.confidence:.2f}), "
                    f"defaulting to information_retrieval intention"
                )
                result.intention = Intention.INFORMATION_RETRIEVAL

            # Cache the result (limit cache size)
            if len(self._llm_cache) >= self._cache_max_size:
                # Remove oldest entry
                oldest_key = next(iter(self._llm_cache))
                del self._llm_cache[oldest_key]
            self._llm_cache[cache_key] = result

            return result
            
        except TimeoutError:
            logger.error(f"Intent classification timeout after {self.timeout}s")
            return self._fallback_result(
                "Classification timeout",
                time.time() - start_time
            )
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse classification response: {e}")
            return self._fallback_result(
                "JSON parsing error",
                time.time() - start_time
            )
        except Exception as e:
            logger.error(f"Intent classification error: {e}")
            return self._fallback_result(
                f"Classification error: {str(e)}",
                time.time() - start_time
            )
    
    def _parse_response(self, response_text: str) -> IntentionResult:
        """
        Parse LLM JSON response into IntentionResult.

        Args:
            response_text: Raw LLM response

        Returns:
            IntentionResult parsed from response

        Raises:
            json.JSONDecodeError: If response is not valid JSON
        """
        # Clean response text
        response_text = self._clean_response(response_text)

        # Parse JSON
        data = json.loads(response_text)

        # Extract intention
        intention = self._parse_intention(data.get("intention", "information_retrieval"))

        # Extract confidence
        confidence = float(data.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]

        return IntentionResult(
            intention=intention,
            confidence=confidence,
            reasoning=f"Classified as {intention.value}"
        )
    
    def _clean_response(self, response_text: str) -> str:
        """
        Clean LLM response to extract JSON.
        
        Handles:
        - Thinking tags (deepseek-r1)
        - Markdown code blocks
        - Extra text before/after JSON
        
        Args:
            response_text: Raw response
            
        Returns:
            Cleaned JSON string
        """
        response_text = response_text.strip()
        
        # Remove thinking tags if present (deepseek-r1 specific)
        if "<think>" in response_text:
            parts = response_text.split("</think>")
            if len(parts) > 1:
                response_text = parts[1].strip()
        
        # Handle markdown code blocks
        if "```json" in response_text:
            start = response_text.find("```json") + 7
            end = response_text.find("```", start)
            if end != -1:
                response_text = response_text[start:end].strip()
        elif "```" in response_text:
            start = response_text.find("```") + 3
            end = response_text.find("```", start)
            if end != -1:
                response_text = response_text[start:end].strip()
        
        # Find JSON object
        if not response_text.startswith("{"):
            start_idx = response_text.find("{")
            if start_idx != -1:
                # Find matching closing brace
                brace_count = 0
                end_idx = start_idx
                for i, char in enumerate(response_text[start_idx:], start_idx):
                    if char == "{":
                        brace_count += 1
                    elif char == "}":
                        brace_count -= 1
                        if brace_count == 0:
                            end_idx = i + 1
                            break
                response_text = response_text[start_idx:end_idx]
        
        return response_text.strip()
    
    def _parse_intention(self, intention_str: str) -> Intention:
        """
        Parse intention string to Intention enum.
        
        Args:
            intention_str: Intention string from JSON
            
        Returns:
            Intention enum value
        """
        intention_str = intention_str.lower().strip()
        
        # Map string to Intention enum
        intention_map = {
            "information_retrieval": Intention.INFORMATION_RETRIEVAL,
            "query": Intention.INFORMATION_RETRIEVAL,  # Backward compatibility
            "conversational_response": Intention.CONVERSATIONAL_RESPONSE,
            "conversational": Intention.CONVERSATIONAL_RESPONSE,  # Backward compatibility
            "department_routing": Intention.DEPARTMENT_ROUTING,
            "routing": Intention.DEPARTMENT_ROUTING,
            "clarification_request": Intention.CLARIFICATION_REQUEST,
            "clarification": Intention.CLARIFICATION_REQUEST,
        }
        
        return intention_map.get(intention_str, Intention.INFORMATION_RETRIEVAL)

    def _fallback_result(self, reason: str, latency: float) -> IntentionResult:
        """
        Create fallback result when classification fails.

        Defaults to INFORMATION_RETRIEVAL as the safest option.
        """
        return IntentionResult(
            intention=Intention.INFORMATION_RETRIEVAL,
            confidence=0.0,
            reasoning=f"Fallback: {reason}",
            latency=latency
        )
