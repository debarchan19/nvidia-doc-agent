#!/usr/bin/env python3
"""
Ingestion script for NVIDIA documentation RAG system.
Activates virtual environment and runs the document ingestion pipeline.
"""

from __future__ import annotations

import logging
import sys
import time


def setup_logging():
    """Configure logging for the ingestion process."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("ingestion.log", mode="w"),
        ],
    )


def main():
    """Main ingestion function."""
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("Starting NVIDIA Documentation Ingestion Pipeline")
    logger.info("=" * 60)

    start_time = time.time()

    try:
        # Import after activating virtual environment
        from rag.config import config
        from rag.embeddings import get_embeddings
        from rag.pipeline.ingest import IngestPipeline

        logger.info("Configuration loaded:")
        logger.info(f"  - Docs root: {config.docs_root}")
        logger.info(f"  - Vector store path: {config.vector_store_path()}")
        logger.info(f"  - Chunk size: {config.chunk_size}")
        logger.info(f"  - Chunk overlap: {config.chunk_overlap}")
        logger.info(f"  - Collection name: {config.chroma_collection}")

        # Check if docs directory exists
        if not config.docs_root.exists():
            logger.error(f"Documentation directory not found: {config.docs_root}")
            logger.info("Please ensure the NVIDIA documentation files are available.")
            sys.exit(1)

        # Count available markdown files
        file_count = config.count_markdown_files()
        logger.info(f"Found {file_count} markdown files to process")

        if file_count == 0:
            logger.warning("No markdown files found to process")
            sys.exit(1)

        # Initialize embeddings
        logger.info("Initializing local ModernBERT embeddings model...")
        embeddings = get_embeddings()  # Uses local ModernBERT-base model
        logger.info("Local ModernBERT embeddings model initialized successfully")

        # Run ingestion pipeline
        logger.info("Starting document ingestion...")
        vector_store = IngestPipeline.run(
            config=config,
            embeddings=embeddings,
            persist_directory=None,  # Use default from config
        )

        # Log success metrics
        elapsed_time = time.time() - start_time
        logger.info("=" * 60)
        logger.info("Ingestion completed successfully!")
        logger.info(f"  - Total processing time: {elapsed_time:.2f} seconds")
        logger.info(f"  - Files processed: {file_count}")
        logger.info(f"  - Vector store location: {config.vector_store_path()}")
        logger.info(f"  - Collection name: {config.chroma_collection}")
        logger.info("=" * 60)

        # Test the vector store
        logger.info("Testing vector store...")
        try:
            # Try a simple similarity search to verify the store works
            results = vector_store.similarity_search("NVIDIA GPU", k=1)
            if results:
                logger.info(
                    f"✓ Vector store test successful - found {len(results)} result(s)"
                )
                logger.info(
                    f"  Sample result: {results[0].metadata.get('source', 'unknown')}"
                )
            else:
                logger.warning("⚠ Vector store test returned no results")
        except Exception as e:
            logger.error(f"✗ Vector store test failed: {e}")

    except ImportError as e:
        logger.error(f"Import error - make sure you're in the correct environment: {e}")
        logger.info(
            "Ensure you've activated the virtual environment and installed dependencies"
        )
        sys.exit(1)
    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
