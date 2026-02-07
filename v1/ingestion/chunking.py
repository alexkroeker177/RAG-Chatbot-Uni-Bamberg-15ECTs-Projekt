"""
Text chunking utilities using LangChain's RecursiveCharacterTextSplitter.

Splits documents into semantic chunks with overlap for better retrieval.
"""

import hashlib
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from v1.core.config import ChunkingConfig
from v1.core.logger import setup_logger


logger = setup_logger(__name__)


def create_text_splitter(config: ChunkingConfig) -> RecursiveCharacterTextSplitter:
    """
    Create a RecursiveCharacterTextSplitter with configured parameters.
    
    Args:
        config: Chunking configuration
        
    Returns:
        Configured text splitter
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        length_function=len,
        is_separator_regex=False,
        separators=[
            "\n\n",  # Paragraph breaks
            "\n",    # Line breaks
            ". ",    # Sentence breaks
            ", ",    # Clause breaks
            " ",     # Word breaks
            ""       # Character breaks (fallback)
        ]
    )
    
    logger.debug(
        f"Created text splitter: chunk_size={config.chunk_size}, "
        f"overlap={config.chunk_overlap}"
    )
    
    return splitter


def generate_chunk_id(doc: Document, chunk_index: int) -> str:
    """
    Generate a unique ID for a chunk based on its content and metadata.
    
    This enables idempotent ingestion - we can check if a chunk already exists.
    
    Args:
        doc: Document chunk
        chunk_index: Index of this chunk in the original document
        
    Returns:
        Unique chunk ID (hash)
    """
    # Create a unique string from metadata and content
    id_string = (
        f"{doc.metadata.get('doc', 'unknown')}_"
        f"{doc.metadata.get('page', 'unknown')}_"
        f"{chunk_index}_"
        f"{doc.page_content[:100]}"  # First 100 chars for uniqueness
    )
    
    # Generate hash
    chunk_id = hashlib.md5(id_string.encode()).hexdigest()
    return chunk_id


def chunk_documents(
    documents: List[Document],
    config: ChunkingConfig
) -> List[Document]:
    """
    Split documents into chunks with overlap.
    
    Preserves metadata from original documents and adds chunk-specific info.
    
    Args:
        documents: List of documents to chunk
        config: Chunking configuration
        
    Returns:
        List of chunked documents with metadata
    """
    if not documents:
        logger.warning("No documents to chunk")
        return []
    
    splitter = create_text_splitter(config)
    
    all_chunks = []
    total_original = len(documents)
    
    for doc_idx, doc in enumerate(documents, 1):
        try:
            # Split the document
            chunks = splitter.split_documents([doc])
            
            # Add chunk-specific metadata
            for chunk_idx, chunk in enumerate(chunks):
                # Generate unique chunk ID
                chunk_id = generate_chunk_id(chunk, chunk_idx)
                
                # Add chunk metadata
                chunk.metadata["chunk_id"] = chunk_id
                chunk.metadata["chunk_index"] = chunk_idx
                chunk.metadata["total_chunks"] = len(chunks)
                
                all_chunks.append(chunk)
            
            logger.debug(
                f"Chunked document [{doc_idx}/{total_original}]: "
                f"{len(chunks)} chunks from {doc.metadata.get('doc', 'unknown')}"
            )
            
        except Exception as e:
            logger.error(
                f"Error chunking document {doc.metadata.get('doc', 'unknown')}: {e}"
            )
            continue
    
    logger.info(
        f"Chunking complete: {len(documents)} documents → {len(all_chunks)} chunks"
    )
    
    return all_chunks


def get_chunk_stats(chunks: List[Document]) -> dict:
    """
    Get statistics about chunked documents.
    
    Args:
        chunks: List of chunked documents
        
    Returns:
        Dictionary with statistics
    """
    if not chunks:
        return {
            "total_chunks": 0,
            "avg_chunk_length": 0,
            "min_chunk_length": 0,
            "max_chunk_length": 0,
            "unique_sources": 0
        }
    
    chunk_lengths = [len(chunk.page_content) for chunk in chunks]
    unique_sources = len(set(chunk.metadata.get("doc", "unknown") for chunk in chunks))
    
    stats = {
        "total_chunks": len(chunks),
        "avg_chunk_length": sum(chunk_lengths) / len(chunk_lengths),
        "min_chunk_length": min(chunk_lengths),
        "max_chunk_length": max(chunk_lengths),
        "unique_sources": unique_sources
    }
    
    return stats


# Example usage
if __name__ == "__main__":
    from v1.core.config import load_config, ChunkingConfig
    
    # Create test documents
    test_docs = [
        Document(
            page_content="Dies ist ein langer Text über Prüfungsordnungen. " * 50,
            metadata={"doc": "test.pdf", "page": "1", "source_type": "regulation"}
        ),
        Document(
            page_content="Ein weiterer Text mit Informationen. " * 30,
            metadata={"doc": "test.pdf", "page": "2", "source_type": "regulation"}
        )
    ]
    
    # Load config
    config = load_config()
    
    # Chunk documents
    chunks = chunk_documents(test_docs, config.chunking)
    
    print(f"\nChunked {len(test_docs)} documents into {len(chunks)} chunks")
    
    # Show stats
    stats = get_chunk_stats(chunks)
    print("\nChunk Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Show sample chunk
    if chunks:
        print(f"\nSample chunk:")
        print(f"  Content length: {len(chunks[0].page_content)}")
        print(f"  Metadata: {chunks[0].metadata}")
        print(f"  Content preview: {chunks[0].page_content[:200]}...")
