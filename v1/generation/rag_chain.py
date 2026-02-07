"""
Conv-BDI RAG Chain Orchestration.

Based on "Conv-BDI: An Extension of the BDI Framework for Conversational Agents"
(Walker et al., 2025)

This module orchestrates the complete RAG pipeline with Conv-BDI components:
- Beliefs: Retrieved documents + User context
- Intentions: Selected via IntentClassifier
- Actions: Retrieval, Generation, Routing
"""

import time
from typing import Iterator, Dict, Any, List, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.llms import Ollama

from v1.core.config import Config
from v1.core.logger import setup_logger
from v1.generation.prompts import (
    create_simple_prompt,
    format_context_with_sources
)
from v1.generation.llm import create_llm
from v1.retrieval.embeddings import create_embeddings
from v1.retrieval.vector_store import VectorStoreManager
from v1.retrieval.retriever import create_retriever

# Conv-BDI imports
from v1.generation.conv_bdi import (
    Intention,
    IntentionResult,
    DynamicBeliefs,
    normalize_program_name
)
from v1.generation.intent_classifier import IntentClassifier
from v1.generation.conversational_responses import ConversationalResponses


logger = setup_logger(__name__)


class BeliefStateManager:
    """
    Manages Dynamic Beliefs across conversation sessions.
    
    This is the Conv-BDI Beliefs component for tracking user context.
    """
    
    def __init__(self):
        """Initialize belief state storage (in-memory)."""
        self._beliefs: Dict[str, DynamicBeliefs] = {}
    
    def get_beliefs(self, session_id: str) -> Optional[DynamicBeliefs]:
        """Get beliefs for a session."""
        return self._beliefs.get(session_id)
    
    def update_beliefs(self, session_id: str, new_beliefs: DynamicBeliefs):
        """Update beliefs for a session, merging with existing."""
        existing = self._beliefs.get(session_id)
        
        if existing:
            self._beliefs[session_id] = existing.merge(new_beliefs)
        else:
            self._beliefs[session_id] = new_beliefs
        
        logger.debug(f"Updated beliefs for session {session_id}: {self._beliefs[session_id]}")
    
    def clear_beliefs(self, session_id: str):
        """Clear beliefs for a session."""
        if session_id in self._beliefs:
            del self._beliefs[session_id]

    def initialize_beliefs(self, session_id: str, user_context: Dict[str, Any]):
        """
        Initialize beliefs from user form data.

        Args:
            session_id: Session identifier
            user_context: Dict with degree_level, program, semester from form
        """
        beliefs = DynamicBeliefs(
            degree=user_context.get("degree_level"),
            program=user_context.get("program"),
            semester=user_context.get("semester"),
            current_topic=None,
            clarification_needed=False
        )
        self._beliefs[session_id] = beliefs
        logger.debug(f"Initialized beliefs for session {session_id}: {beliefs}")


class RAGChain:
    """
    Conv-BDI RAG Pipeline.
    
    Orchestrates the complete RAG system with:
    - Intent Classification (Desires → Intentions)
    - Belief State Management
    - Action Execution (Retrieval, Generation, Routing)
    """
    
    def __init__(self, config: Config):
        """
        Initialize RAG chain with Conv-BDI components.
        
        Args:
            config: System configuration
        """
        self.config = config
        
        logger.info("Initializing Conv-BDI RAG chain...")
        
        # === STATIC BELIEFS: Vector Store ===
        self.embeddings = create_embeddings(config.ollama)
        manager = VectorStoreManager(config.chroma, self.embeddings)
        self.vectorstore = manager.get_vectorstore()
        self.retriever = create_retriever(self.vectorstore, config.retrieval)
        
        # === ACTIONS: LLMs ===
        self.llm = create_llm(config.ollama, streaming=True)
        self.prompt = create_simple_prompt()
        self.chain = self._build_chain()
        
        # === INTENTIONS: Intent Classifier ===
        if config.intent_classification.enabled:
            try:
                self.intent_classifier = IntentClassifier(config)
                self.belief_manager = BeliefStateManager()

                # Reuse intent classifier's LLM for conversational responses (2 LLMs total)
                self.conversational_llm = self.intent_classifier.llm

                logger.info("✓ Intent classification enabled")
                logger.info(f"✓ Using {config.intent_classification.model} for intent + conversational")
            except Exception as e:
                logger.warning(f"Failed to initialize intent classifier: {e}")
                logger.warning("Continuing without intent classification")
                self.intent_classifier = None
                self.belief_manager = None
                self.conversational_llm = None
        else:
            self.intent_classifier = None
            self.belief_manager = None
            self.conversational_llm = None
            logger.info("Intent classification disabled")
        
        logger.info("✓ Conv-BDI RAG chain initialized")
    
    def _build_chain(self):
        """Build the LangChain pipeline for RAG (without beliefs - for backward compat)."""
        chain = (
            {
                "context": self.retriever | self._format_docs,
                "user_beliefs": lambda x: "Kein Nutzerkontext bekannt.",
                "question": RunnablePassthrough()
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        return chain

    def _generate_with_beliefs(
        self,
        question: str,
        docs: List,
        beliefs: Optional[DynamicBeliefs] = None
    ) -> Iterator[str]:
        """
        Generate response with proper beliefs injection.

        Uses raw question for retrieval, injects beliefs only in generation prompt.
        """
        # Format context from retrieved docs
        context = format_context_with_sources(docs)

        # Format beliefs
        beliefs_str = "Kein Nutzerkontext bekannt."
        if beliefs:
            beliefs_str = beliefs.format_for_prompt()

        # Build prompt with all components
        prompt_text = self.prompt.format(
            context=context,
            user_beliefs=beliefs_str,
            question=question
        )

        # Stream response
        for chunk in self.llm.stream(prompt_text):
            yield chunk

    def _augment_query_with_beliefs(
        self,
        query: str,
        beliefs: Optional[DynamicBeliefs]
    ) -> str:
        """
        Augment query with user beliefs for better semantic retrieval.

        Prepends degree and program context naturally to improve
        semantic similarity matching for ambiguous queries.

        Args:
            query: Original user query
            beliefs: Current user beliefs (degree, program, semester)

        Returns:
            Augmented query string, or original if no relevant beliefs
        """
        if not beliefs:
            return query

        # Build context prefix from available beliefs
        context_parts = []

        # Add degree if available
        if beliefs.degree:
            context_parts.append(beliefs.degree)

        # Add program if available (use canonical name for better matching)
        if beliefs.program:
            canonical, _ = normalize_program_name(beliefs.program)
            context_parts.append(canonical)

        # Only augment if we have meaningful context
        if not context_parts:
            return query

        # Create natural prefix
        # e.g., "Master Angewandte Informatik: Welche Modulgruppen gibt es?"
        context_prefix = " ".join(context_parts)
        augmented_query = f"{context_prefix}: {query}"

        logger.debug(f"Query augmented: '{query}' -> '{augmented_query}'")

        return augmented_query

    def _rerank_documents_by_beliefs(
        self,
        docs: List,
        beliefs: Optional[DynamicBeliefs],
        boost_factor: float = 1.5
    ) -> List:
        """
        Re-rank retrieved documents based on user beliefs.

        Boosts documents that match the user's program/degree while
        keeping all documents (doesn't filter out potentially relevant info).

        Strategy:
        1. Exact metadata match (program field) -> highest boost
        2. Content mention of program name -> medium boost
        3. Same degree level -> small boost
        4. General/APO documents -> no penalty (always relevant)

        Args:
            docs: List of retrieved documents
            beliefs: Current user beliefs
            boost_factor: Multiplier for matching documents (default 1.5)

        Returns:
            Re-ranked list of documents (same length, different order)
        """
        if not beliefs or not docs:
            return docs

        # Get normalized program info
        user_program_short = None
        user_program_canonical = None
        user_degree = beliefs.degree

        if beliefs.program:
            user_program_canonical, user_program_short = normalize_program_name(beliefs.program)

        # Score each document
        scored_docs = []

        for doc in docs:
            score = 1.0  # Base score

            metadata = doc.metadata
            doc_program = metadata.get("program", "general")
            doc_program_canonical = metadata.get("program_canonical", "")
            doc_degree = metadata.get("degree", "general")
            content_lower = doc.page_content.lower()

            # Boost 1: Exact program metadata match (highest priority)
            if user_program_short and doc_program == user_program_short:
                score *= boost_factor * 1.2  # 1.8x boost
                logger.debug(f"Metadata program match: {doc.metadata.get('doc')}")

            # Boost 2: Program name appears in content
            elif user_program_canonical:
                canonical_lower = user_program_canonical.lower()
                if canonical_lower in content_lower:
                    score *= boost_factor  # 1.5x boost
                    logger.debug(f"Content program match: {doc.metadata.get('doc')}")
                # Also check short form in content
                elif user_program_short and user_program_short in content_lower:
                    score *= boost_factor * 0.8  # 1.2x boost

            # Boost 3: Same degree level
            if user_degree and doc_degree == user_degree:
                score *= 1.1  # 10% boost for same degree

            # Boost 4: General/APO documents are always relevant (no penalty)
            # They contain rules that apply to all programs

            scored_docs.append((score, doc))

        # Sort by score (descending) while maintaining relative order for equal scores
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        # Log reranking result
        if logger.isEnabledFor(10):  # DEBUG level
            logger.debug("Reranking result:")
            for score, doc in scored_docs[:3]:
                logger.debug(f"  {score:.2f}: {doc.metadata.get('doc', 'unknown')}")

        return [doc for _, doc in scored_docs]

    def _format_docs(self, docs):
        """Format retrieved documents (Static Beliefs)."""
        return format_context_with_sources(docs)

    def initialize_user_context(self, session_id: str, user_context: Dict[str, Any]):
        """
        Initialize beliefs from user form data.

        Call this after the user submits the profile form.

        Args:
            session_id: Session identifier
            user_context: Dict with degree_level, program, semester
        """
        if self.belief_manager:
            self.belief_manager.initialize_beliefs(session_id, user_context)

    # =========================================================================
    # MAIN ENTRY POINTS
    # =========================================================================
    
    def invoke(self, question: str) -> Dict[str, Any]:
        """
        Run RAG pipeline without intent classification.
        
        Args:
            question: User question
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        start_time = time.time()
        
        try:
            # ACTION: Retrieve documents
            docs = self.retriever.invoke(question)
            
            # ACTION: Generate answer
            answer = self.chain.invoke(question)
            
            latency = time.time() - start_time
            
            # Extract sources
            sources = self._extract_sources(docs)
            
            logger.info(f"Generated answer in {latency:.2f}s")
            
            return {
                "answer": answer,
                "sources": sources,
                "retrieved_docs": docs,
                "latency": latency,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error in RAG chain: {e}")
            return {
                "answer": f"Fehler bei der Antwortgenerierung: {str(e)}",
                "sources": [],
                "retrieved_docs": [],
                "latency": time.time() - start_time,
                "success": False
            }
    
    def invoke_with_intent(
        self,
        question: str,
        session_id: str,
        conversation_history: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Run Conv-BDI pipeline with intent classification.
        
        This is the main entry point for the Conv-BDI system:
        1. Classify intent (Desires → Intentions)
        2. Update beliefs
        3. Execute planned actions
        
        Args:
            question: User message
            session_id: Session identifier for belief tracking
            conversation_history: Recent conversation turns
            
        Returns:
            Dictionary with answer, sources, intention, beliefs, and metadata
        """
        # Fallback if intent classification disabled
        if not self.intent_classifier:
            result = self.invoke(question)
            result["intention"] = "information_retrieval"
            result["beliefs"] = None
            return result
        
        # Get current beliefs
        current_beliefs = self.belief_manager.get_beliefs(session_id)
        
        # === INTENTION SELECTION ===
        intention_result = self.intent_classifier.classify(
            question,
            conversation_history or [],
            current_beliefs
        )
        
        # === UPDATE BELIEFS ===
        if intention_result.updated_beliefs:
            self.belief_manager.update_beliefs(
                session_id,
                intention_result.updated_beliefs
            )
        
        # Get updated beliefs
        beliefs = self.belief_manager.get_beliefs(session_id)
        
        # === EXECUTE ACTIONS based on Intention ===
        if intention_result.intention == Intention.INFORMATION_RETRIEVAL:
            return self._execute_retrieval(question, session_id, beliefs, intention_result)
        
        elif intention_result.intention == Intention.CONVERSATIONAL_RESPONSE:
            return self._execute_conversational(question, session_id, beliefs, intention_result, conversation_history)
        
        elif intention_result.intention == Intention.DEPARTMENT_ROUTING:
            return self._execute_routing(question, session_id, beliefs, intention_result, conversation_history)
        
        elif intention_result.intention == Intention.CLARIFICATION_REQUEST:
            return self._execute_clarification(question, session_id, beliefs, intention_result, conversation_history)
        
        else:
            # Default to retrieval
            return self._execute_retrieval(question, session_id, beliefs, intention_result)
    
    # =========================================================================
    # ACTION EXECUTORS
    # =========================================================================
    
    def _execute_retrieval(
        self,
        question: str,
        session_id: str,
        beliefs: Optional[DynamicBeliefs],
        intention_result: IntentionResult
    ) -> Dict[str, Any]:
        """
        Execute INFORMATION_RETRIEVAL intention.

        Uses belief-aware retrieval:
        1. Augment query with user context (degree, program)
        2. Retrieve documents
        3. Re-rank based on user beliefs
        4. Generate response with beliefs in prompt
        """
        start_time = time.time()

        # Step 1: Augment query with beliefs for better semantic matching
        augmented_query = self._augment_query_with_beliefs(question, beliefs)

        # Step 2: Retrieve documents using augmented query
        docs = self.retriever.invoke(augmented_query)

        # Step 3: Re-rank documents based on beliefs
        docs = self._rerank_documents_by_beliefs(docs, beliefs)

        # Step 4: Generate with beliefs properly injected in prompt
        context = format_context_with_sources(docs)
        beliefs_str = beliefs.format_for_prompt() if beliefs else "Kein Nutzerkontext bekannt."
        prompt_text = self.prompt.format(
            context=context,
            user_beliefs=beliefs_str,
            question=question  # Use original question in prompt (not augmented)
        )
        answer = self.llm.invoke(prompt_text)

        latency = time.time() - start_time

        query_was_augmented = augmented_query != question
        logger.info(
            f"Executed INFORMATION_RETRIEVAL in {latency:.2f}s "
            f"(query augmented: {query_was_augmented})"
        )

        return {
            "answer": answer,
            "sources": self._extract_sources(docs),
            "retrieved_docs": docs,
            "intention": intention_result.intention.value,
            "confidence": intention_result.confidence,
            "beliefs": beliefs.to_dict() if beliefs else None,
            "reasoning": intention_result.reasoning,
            "latency": latency,
            "success": True
        }
    
    def _execute_conversational(
        self,
        message: str,
        session_id: str,
        beliefs: Optional[DynamicBeliefs],
        intention_result: IntentionResult,
        conversation_history: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Execute CONVERSATIONAL_RESPONSE intention.
        
        Actions: update_user_context → generate_conversational_response
        """
        start_time = time.time()
        
        # ACTION: Generate conversational response WITH history
        response = ConversationalResponses.get_response(
            message=message,
            context=beliefs,
            has_new_context=intention_result.updated_beliefs is not None,
            llm=self.conversational_llm,
            intention=Intention.CONVERSATIONAL_RESPONSE,
            conversation_history=conversation_history
        )
        
        latency = time.time() - start_time
        
        logger.info(f"Executed CONVERSATIONAL_RESPONSE in {latency:.2f}s")
        
        return {
            "answer": response,
            "sources": [],
            "retrieved_docs": [],
            "intention": intention_result.intention.value,
            "confidence": intention_result.confidence,
            "beliefs": beliefs.to_dict() if beliefs else None,
            "reasoning": intention_result.reasoning,
            "latency": latency,
            "success": True
        }
    
    def _execute_routing(
        self,
        message: str,
        session_id: str,
        beliefs: Optional[DynamicBeliefs],
        intention_result: IntentionResult,
        conversation_history: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Execute DEPARTMENT_ROUTING intention.
        
        Actions: retrieve_documents → generate_rag_response (with routing focus)
        
        Uses RAG to find the correct department from the vector DB
        (departments.json is embedded in ChromaDB).
        """
        start_time = time.time()
        
        # Create a routing-focused query to retrieve department info
        routing_query = self._create_routing_query(message, beliefs)
        
        # ACTION: Retrieve documents (including department info from ChromaDB)
        docs = self.retriever.invoke(routing_query)
        
        # ACTION: Generate RAG response (grounded on retrieved department info)
        answer = self.chain.invoke(routing_query)
        
        latency = time.time() - start_time
        
        logger.info(f"Executed DEPARTMENT_ROUTING with RAG in {latency:.2f}s")
        
        return {
            "answer": answer,
            "sources": self._extract_sources(docs),
            "retrieved_docs": docs,
            "intention": intention_result.intention.value,
            "confidence": intention_result.confidence,
            "beliefs": beliefs.to_dict() if beliefs else None,
            "reasoning": intention_result.reasoning,
            "latency": latency,
            "success": True
        }
    
    def _create_routing_query(
        self,
        message: str,
        beliefs: Optional[DynamicBeliefs]
    ) -> str:
        """
        Create a query optimized for department/responsibility retrieval.
        
        Args:
            message: Original user message
            beliefs: Current user beliefs
            
        Returns:
            Enhanced query for department lookup
        """
        # Add context about looking for responsible department
        query_parts = [message]
        
        # Add belief context if available
        if beliefs:
            if beliefs.current_topic:
                query_parts.append(f"Thema: {beliefs.current_topic}")
        
        # Add routing keywords to improve retrieval
        query_parts.append("Zuständigkeit Abteilung Kontakt")
        
        return " ".join(query_parts)
    
    def _execute_clarification(
        self,
        message: str,
        session_id: str,
        beliefs: Optional[DynamicBeliefs],
        intention_result: IntentionResult,
        conversation_history: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Execute CLARIFICATION_REQUEST intention.
        
        Actions: generate_conversational_response (asking for clarification)
        """
        start_time = time.time()
        
        # ACTION: Generate clarification request WITH history
        response = ConversationalResponses.get_response(
            message=message,
            context=beliefs,
            has_new_context=False,
            llm=self.conversational_llm,
            intention=Intention.CLARIFICATION_REQUEST,
            conversation_history=conversation_history
        )
        
        latency = time.time() - start_time
        
        logger.info(f"Executed CLARIFICATION_REQUEST in {latency:.2f}s")
        
        return {
            "answer": response,
            "sources": [],
            "retrieved_docs": [],
            "intention": intention_result.intention.value,
            "confidence": intention_result.confidence,
            "beliefs": beliefs.to_dict() if beliefs else None,
            "reasoning": intention_result.reasoning,
            "latency": latency,
            "success": True
        }
    
    # =========================================================================
    # STREAMING METHODS
    # =========================================================================
    
    def stream(self, question: str) -> Iterator[str]:
        """Stream answer tokens."""
        try:
            for chunk in self.chain.stream(question):
                yield chunk
        except Exception as e:
            logger.error(f"Error streaming answer: {e}")
            yield f"\n\nFehler: {str(e)}"
    
    def stream_with_intent(
        self,
        question: str,
        session_id: str,
        conversation_history: Optional[List[Dict]] = None
    ) -> tuple[Iterator[str], Dict[str, Any]]:
        """
        Stream answer with Conv-BDI intent classification.
        
        Returns:
            Tuple of (answer_stream, metadata_dict)
        """
        # Fallback if intent classification disabled
        if not self.intent_classifier:
            docs = self.retriever.invoke(question)
            metadata = {
                "sources": self._extract_sources(docs),
                "intention": "information_retrieval",
                "confidence": 1.0,
                "beliefs": None
            }
            return self.stream(question), metadata
        
        # Get current beliefs
        current_beliefs = self.belief_manager.get_beliefs(session_id)
        
        # Classify intent
        intention_result = self.intent_classifier.classify(
            question,
            conversation_history or [],
            current_beliefs
        )
        
        # Update beliefs
        if intention_result.updated_beliefs:
            self.belief_manager.update_beliefs(
                session_id,
                intention_result.updated_beliefs
            )
        
        beliefs = self.belief_manager.get_beliefs(session_id)
        
        # Prepare metadata
        metadata = {
            "intention": intention_result.intention.value,
            "confidence": intention_result.confidence,
            "beliefs": beliefs.to_dict() if beliefs else None,
            "reasoning": intention_result.reasoning,
            "sources": []
        }
        
        # Route based on intention
        if intention_result.intention == Intention.INFORMATION_RETRIEVAL:
            # Step 1: Augment query with beliefs
            augmented_query = self._augment_query_with_beliefs(question, beliefs)

            # Step 2: Retrieve with augmented query
            docs = self.retriever.invoke(augmented_query)

            # Step 3: Re-rank based on beliefs
            docs = self._rerank_documents_by_beliefs(docs, beliefs)

            metadata["sources"] = self._extract_sources(docs)
            # Generate with beliefs in prompt (use original question)
            return self._generate_with_beliefs(question, docs, beliefs), metadata

        elif intention_result.intention == Intention.DEPARTMENT_ROUTING:
            # Retrieve with routing query
            routing_query = self._create_routing_query(question, beliefs)
            docs = self.retriever.invoke(routing_query)
            metadata["sources"] = self._extract_sources(docs)
            return self._generate_with_beliefs(routing_query, docs, beliefs), metadata
        
        else:
            # Instant response for conversational/clarification intentions
            response = ConversationalResponses.get_response(
                message=question,
                context=beliefs,
                has_new_context=intention_result.updated_beliefs is not None,
                llm=self.conversational_llm,
                intention=intention_result.intention,
                conversation_history=conversation_history
            )
            
            def instant_stream():
                yield response
            
            return instant_stream(), metadata
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _extract_sources(self, docs: List) -> List[Dict[str, Any]]:
        """Extract source metadata from documents."""
        return [
            {
                "doc": doc.metadata.get("doc", "unknown"),
                "page": doc.metadata.get("page", "n/a"),
                "section": doc.metadata.get("section_title", ""),
                "type": doc.metadata.get("source_type", "unknown")
            }
            for doc in docs
        ]


# =============================================================================
# BACKWARD COMPATIBILITY
# =============================================================================
# These aliases maintain compatibility with existing code

# Legacy imports that might be used elsewhere
from v1.generation.user_context import UserContextManager


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    from v1.core.config import load_config
    
    config = load_config()
    
    try:
        print("Initializing Conv-BDI RAG chain...")
        rag = RAGChain(config)
        
        session_id = "test-session"
        
        # Test conversation flow
        test_messages = [
            "Hallo, ich studiere Informatik im 3. Semester",
            "Wie viele ECTS brauche ich für den Bachelor?",
            "Danke, das hilft mir weiter!"
        ]
        
        for msg in test_messages:
            print(f"\n{'='*70}")
            print(f"User: {msg}")
            print('='*70)
            
            result = rag.invoke_with_intent(msg, session_id)
            
            print(f"\nIntention: {result['intention']}")
            print(f"Confidence: {result['confidence']:.2f}")
            print(f"Beliefs: {result.get('beliefs')}")
            print(f"Answer: {result['answer'][:200]}...")
            if result['sources']:
                print(f"Sources: {len(result['sources'])} documents")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
