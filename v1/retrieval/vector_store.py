"""
Chroma vector store management with idempotent operations.

Handles creation, persistence, and querying of the vector database.
"""

from pathlib import Path
from typing import List, Optional, Set

import chromadb
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from tqdm import tqdm

from v1.core.config import ChromaConfig
from v1.core.logger import setup_logger


logger = setup_logger(__name__)


class VectorStoreManager:
    """Manages Chroma vector database operations."""
    
    def __init__(
        self,
        config: ChromaConfig,
        embeddings
    ):
        """
        Initialize vector store manager.
        
        Args:
            config: Chroma configuration
            embeddings: Embedding function (OllamaEmbeddings)
        """
        self.config = config
        self.embeddings = embeddings
        self.persist_dir = Path(config.persist_directory)
        
        # Create persist directory if it doesn't exist
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Chroma client
        self._init_client()
        
        logger.info(f"Initialized vector store at {self.persist_dir}")
    
    def _init_client(self):
        """Initialize Chroma client and collection."""
        # Create persistent client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(
                name=self.config.collection_name
            )
            logger.info(f"Loaded existing collection: {self.config.collection_name}")
        except Exception:
            self.collection = self.client.create_collection(
                name=self.config.collection_name,
                metadata={"hnsw:space": self.config.distance_metric}
            )
            logger.info(f"Created new collection: {self.config.collection_name}")
    
    def get_vectorstore(self) -> Chroma:
        """
        Get LangChain Chroma vectorstore instance.
        
        Returns:
            Chroma vectorstore
        """
        vectorstore = Chroma(
            client=self.client,
            collection_name=self.config.collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(self.persist_dir)
        )
        
        return vectorstore
    
    def get_existing_chunk_ids(self) -> Set[str]:
        """
        Get set of chunk IDs already in the collection.
        
        Returns:
            Set of existing chunk IDs
        """
        try:
            # Get all documents from collection
            results = self.collection.get(
                include=["metadatas"]
            )
            
            # Extract chunk IDs from metadata
            chunk_ids = set()
            if results and "metadatas" in results:
                for metadata in results["metadatas"]:
                    if metadata and "chunk_id" in metadata:
                        chunk_ids.add(metadata["chunk_id"])
            
            logger.debug(f"Found {len(chunk_ids)} existing chunks in collection")
            return chunk_ids
            
        except Exception as e:
            logger.warning(f"Could not get existing chunk IDs: {e}")
            return set()
    
    def add_documents(
        self,
        documents: List[Document],
        batch_size: int = 100,
        show_progress: bool = True
    ) -> int:
        """
        Add documents to vector store with idempotency.
        
        Only adds documents that don't already exist (based on chunk_id).
        
        Args:
            documents: List of documents to add
            batch_size: Number of documents to process at once
            show_progress: Whether to show progress bar
            
        Returns:
            Number of documents actually added
        """
        if not documents:
            logger.warning("No documents to add")
            return 0
        
        # Get existing chunk IDs
        existing_ids = self.get_existing_chunk_ids()
        
        # Filter out documents that already exist
        new_documents = []
        for doc in documents:
            chunk_id = doc.metadata.get("chunk_id")
            if chunk_id and chunk_id not in existing_ids:
                new_documents.append(doc)
        
        if not new_documents:
            logger.info("All documents already exist in collection, skipping")
            return 0
        
        logger.info(
            f"Adding {len(new_documents)} new documents "
            f"(skipped {len(documents) - len(new_documents)} existing)"
        )
        
        # Get vectorstore
        vectorstore = self.get_vectorstore()
        
        # Add documents in batches with progress bar
        added_count = 0
        
        iterator = range(0, len(new_documents), batch_size)
        if show_progress:
            iterator = tqdm(
                iterator,
                desc="Adding documents",
                unit="batch"
            )
        
        for i in iterator:
            batch = new_documents[i:i + batch_size]
            
            try:
                # Add batch to vectorstore
                vectorstore.add_documents(batch)
                added_count += len(batch)
                
                logger.debug(f"Added batch of {len(batch)} documents")
                
            except Exception as e:
                logger.error(f"Error adding batch: {e}")
                continue
        
        logger.info(f"Successfully added {added_count} documents to vector store")
        
        return added_count
    
    def get_collection_stats(self) -> dict:
        """
        Get statistics about the collection.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            count = self.collection.count()
            
            # Get sample of metadata to analyze
            sample = self.collection.get(
                limit=min(100, count),
                include=["metadatas"]
            )
            
            # Count by source type
            source_types = {}
            if sample and "metadatas" in sample:
                for metadata in sample["metadatas"]:
                    if metadata:
                        source_type = metadata.get("source_type", "unknown")
                        source_types[source_type] = source_types.get(source_type, 0) + 1
            
            stats = {
                "total_chunks": count,
                "collection_name": self.config.collection_name,
                "persist_directory": str(self.persist_dir),
                "distance_metric": self.config.distance_metric,
                "source_types": source_types
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"error": str(e)}
    
    def delete_collection(self):
        """Delete the entire collection."""
        try:
            self.client.delete_collection(name=self.config.collection_name)
            logger.info(f"Deleted collection: {self.config.collection_name}")
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
    
    def reset_collection(self):
        """Reset collection (delete and recreate)."""
        self.delete_collection()
        self._init_client()
        logger.info("Collection reset complete")


# Example usage
if __name__ == "__main__":
    from v1.core.config import load_config
    from v1.retrieval.embeddings import create_embeddings
    
    # Load config
    config = load_config()
    
    # Create embeddings (this will fail if not connected to Ollama)
    try:
        embeddings = create_embeddings(config.ollama)
        
        # Create vector store manager
        manager = VectorStoreManager(config.chroma, embeddings)
        
        # Get stats
        stats = manager.get_collection_stats()
        print("\nVector Store Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Test with sample documents
        test_docs = [
            Document(
                page_content="Test document 1",
                metadata={
                    "chunk_id": "test_1",
                    "doc": "test.pdf",
                    "source_type": "test"
                }
            ),
            Document(
                page_content="Test document 2",
                metadata={
                    "chunk_id": "test_2",
                    "doc": "test.pdf",
                    "source_type": "test"
                }
            )
        ]
        
        print(f"\nAdding {len(test_docs)} test documents...")
        added = manager.add_documents(test_docs, show_progress=False)
        print(f"Added {added} documents")
        
        # Get updated stats
        stats = manager.get_collection_stats()
        print(f"\nUpdated total chunks: {stats['total_chunks']}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you're connected to the Ollama server")
