#!/usr/bin/env python3
"""
Main ingestion script for RAG system.

Processes PDFs, FAQs, and department information, then stores in Chroma.
"""

import sys
import time
from pathlib import Path

from v1.core.config import load_config
from v1.core.health_check import perform_health_check
from v1.core.logger import setup_logger, log_ingestion_start, log_ingestion_complete
from v1.ingestion.document_processor import DocumentProcessor
from v1.ingestion.chunking import chunk_documents, get_chunk_stats
from v1.retrieval.embeddings import create_embeddings
from v1.retrieval.vector_store import VectorStoreManager


logger = setup_logger(__name__)


def main():
    """Main ingestion pipeline."""
    logger.info("=" * 70)
    logger.info("RAG SYSTEM INGESTION PIPELINE")
    logger.info("=" * 70)
    
    start_time = time.time()
    
    try:
        # Load configuration
        logger.info("Loading configuration...")
        config = load_config()
        logger.info("✓ Configuration loaded")
        
        # Perform health check with model listing
        logger.info("\nPerforming health check...")
        if not perform_health_check(config, check_models=True, show_available=True):
            logger.error("Health check failed. Please fix issues before continuing.")
            return 1
        
        # Initialize components
        logger.info("\nInitializing components...")
        
        # Create embeddings
        embeddings = create_embeddings(config.ollama)
        logger.info("✓ Embeddings initialized")
        
        # Create vector store manager
        vector_store = VectorStoreManager(config.chroma, embeddings)
        logger.info("✓ Vector store initialized")
        
        # Create document processor
        processor = DocumentProcessor()
        logger.info("✓ Document processor initialized")
        
        # Process documents
        logger.info("\n" + "=" * 70)
        logger.info("DOCUMENT PROCESSING")
        logger.info("=" * 70)
        
        all_documents = []
        
        # 1. Process PDFs
        pdf_dir = Path(config.data.pdf_directory)
        if pdf_dir.exists():
            pdf_files = list(pdf_dir.glob("*.pdf"))
            
            if pdf_files:
                log_ingestion_start(logger, "PDFs", len(pdf_files))
                
                for pdf_file in pdf_files:
                    docs = processor.process_pdf(pdf_file)
                    all_documents.extend(docs)
                    logger.info(f"  ✓ {pdf_file.name}: {len(docs)} pages")
                
                logger.info(f"✓ Processed {len(pdf_files)} PDF files")
            else:
                logger.warning(f"No PDF files found in {pdf_dir}")
        else:
            logger.warning(f"PDF directory not found: {pdf_dir}")
        
        # 2. Process FAQs
        faq_file = Path(config.data.faq_file)
        if faq_file.exists():
            log_ingestion_start(logger, "FAQ", 1)
            
            docs = processor.process_faq(faq_file)
            all_documents.extend(docs)
            
            logger.info(f"✓ Processed FAQ file: {len(docs)} entries")
        else:
            logger.warning(f"FAQ file not found: {faq_file}")
        
        # 3. Process departments
        dept_file = Path(config.data.departments_file)
        if dept_file.exists():
            log_ingestion_start(logger, "Departments", 1)
            
            docs = processor.process_departments(dept_file)
            all_documents.extend(docs)
            
            logger.info(f"✓ Processed departments file: {len(docs)} entries")
        else:
            logger.warning(f"Departments file not found: {dept_file}")
        
        # Summary of document processing
        logger.info("\n" + "-" * 70)
        logger.info(f"Total documents processed: {len(all_documents)}")
        logger.info(f"Processing stats: {processor.get_stats()}")
        
        if not all_documents:
            logger.error("No documents were processed. Please check your data files.")
            return 1
        
        # Chunk documents
        logger.info("\n" + "=" * 70)
        logger.info("TEXT CHUNKING")
        logger.info("=" * 70)
        
        chunks = chunk_documents(all_documents, config.chunking)
        
        # Show chunk statistics
        stats = get_chunk_stats(chunks)
        logger.info("\nChunk Statistics:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")
        
        if not chunks:
            logger.error("No chunks were created. Please check chunking configuration.")
            return 1
        
        # Add to vector store
        logger.info("\n" + "=" * 70)
        logger.info("VECTOR STORE INGESTION")
        logger.info("=" * 70)
        
        added_count = vector_store.add_documents(
            chunks,
            batch_size=100,
            show_progress=True
        )
        
        logger.info(f"\n✓ Added {added_count} chunks to vector store")
        
        # Show final statistics
        logger.info("\n" + "=" * 70)
        logger.info("FINAL STATISTICS")
        logger.info("=" * 70)
        
        vs_stats = vector_store.get_collection_stats()
        logger.info("\nVector Store:")
        for key, value in vs_stats.items():
            logger.info(f"  {key}: {value}")
        
        # Calculate duration
        duration = time.time() - start_time
        
        logger.info("\n" + "=" * 70)
        logger.info(f"✓ INGESTION COMPLETE in {duration:.2f}s")
        logger.info("=" * 70)
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("\n\nIngestion interrupted by user")
        return 1
        
    except Exception as e:
        logger.error(f"\n\nIngestion failed with error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
