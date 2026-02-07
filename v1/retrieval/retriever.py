"""
Retrieval engine using MMR (Maximal Marginal Relevance).

Retrieves relevant document chunks from the vector store.
"""

from typing import List

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from v1.core.config import RetrievalConfig
from v1.core.logger import setup_logger


logger = setup_logger(__name__)


def create_retriever(
    vectorstore: VectorStore,
    config: RetrievalConfig
):
    """
    Create a retriever with MMR search.
    
    Args:
        vectorstore: LangChain vector store
        config: Retrieval configuration
        
    Returns:
        Configured retriever
    """
    retriever = vectorstore.as_retriever(
        search_type=config.search_type,
        search_kwargs={
            "k": config.final_k,
            "fetch_k": config.fetch_k,
            "lambda_mult": config.lambda_mult
        }
    )
    
    logger.info(
        f"Created retriever: {config.search_type}, "
        f"fetch_k={config.fetch_k}, final_k={config.final_k}"
    )
    
    return retriever


def retrieve_documents(
    retriever,
    query: str
) -> List[Document]:
    """
    Retrieve relevant documents for a query.
    
    Args:
        retriever: LangChain retriever
        query: User query
        
    Returns:
        List of relevant documents
    """
    try:
        logger.debug(f"Retrieving documents for query: {query[:100]}...")
        
        documents = retriever.invoke(query)
        
        logger.info(f"Retrieved {len(documents)} documents")
        
        # Log document sources
        sources = [doc.metadata.get("doc", "unknown") for doc in documents]
        logger.debug(f"Sources: {set(sources)}")
        
        return documents
        
    except Exception as e:
        logger.error(f"Error retrieving documents: {e}")
        return []


def format_retrieved_docs(documents: List[Document]) -> str:
    """
    Format retrieved documents for display or prompt inclusion.
    
    Args:
        documents: List of retrieved documents
        
    Returns:
        Formatted string with document content and metadata
    """
    if not documents:
        return "Keine relevanten Dokumente gefunden."
    
    formatted_parts = []
    
    for i, doc in enumerate(documents, 1):
        meta = doc.metadata
        
        # Format source information
        source_info = f"[Quelle {i}] {meta.get('doc', 'unknown')}"
        
        if meta.get('page') and meta.get('page') != 'n/a':
            source_info += f", Seite {meta['page']}"
        
        if meta.get('section_title'):
            source_info += f" - {meta['section_title']}"
        
        # Add content
        formatted_parts.append(f"{source_info}\n{doc.page_content}\n")
    
    return "\n".join(formatted_parts)


# Example usage
if __name__ == "__main__":
    from v1.core.config import load_config
    from v1.retrieval.embeddings import create_embeddings
    from v1.retrieval.vector_store import VectorStoreManager
    
    # Load config
    config = load_config()
    
    try:
        # Create embeddings
        embeddings = create_embeddings(config.ollama)
        
        # Create vector store manager
        manager = VectorStoreManager(config.chroma, embeddings)
        vectorstore = manager.get_vectorstore()
        
        # Create retriever
        retriever = create_retriever(vectorstore, config.retrieval)
        
        # Test queries
        test_queries = [
            "Wie viele ECTS brauche ich für den Bachelor?",
            "Kann ich mich für eine Prüfung abmelden?",
            "Wie beantrage ich ein Urlaubssemester?"
        ]
        
        for query in test_queries:
            print(f"\n{'='*70}")
            print(f"Query: {query}")
            print('='*70)
            
            docs = retrieve_documents(retriever, query)
            
            print(f"\nRetrieved {len(docs)} documents:")
            for i, doc in enumerate(docs, 1):
                print(f"\n[{i}] {doc.metadata.get('doc', 'unknown')}")
                print(f"    Page: {doc.metadata.get('page', 'n/a')}")
                print(f"    Type: {doc.metadata.get('source_type', 'unknown')}")
                print(f"    Content: {doc.page_content[:150]}...")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure ingestion has been run and you're connected to Ollama")
