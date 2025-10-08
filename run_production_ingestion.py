#!/usr/bin/env python3
"""
Production Ingestion Script for Linux Servers
Optimized for large-scale document processing with proper resource management.
"""

import argparse
import logging
import os
import platform
import psutil
import time
from pathlib import Path
from typing import Optional

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Configure comprehensive logging."""
    level = getattr(logging, log_level.upper())
    
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file, mode="w"))
    
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )

def check_system_resources():
    """Check and log system resources."""
    logger = logging.getLogger(__name__)
    
    # CPU information
    cpu_count = psutil.cpu_count()
    cpu_freq = psutil.cpu_freq()
    
    # Memory information
    memory = psutil.virtual_memory()
    memory_gb = memory.total / (1024**3)
    
    # Disk information
    disk = psutil.disk_usage('/')
    disk_free_gb = disk.free / (1024**3)
    
    logger.info("=" * 60)
    logger.info("SYSTEM RESOURCES")
    logger.info("=" * 60)
    logger.info(f"Platform: {platform.system()} {platform.release()}")
    logger.info(f"Architecture: {platform.machine()}")
    logger.info(f"CPU: {cpu_count} cores")
    if cpu_freq:
        logger.info(f"CPU Frequency: {cpu_freq.current:.0f} MHz")
    logger.info(f"Memory: {memory_gb:.1f} GB total, {memory.percent}% used")
    logger.info(f"Disk: {disk_free_gb:.1f} GB free")
    logger.info("=" * 60)
    
    # Resource recommendations
    if memory_gb < 8:
        logger.warning("⚠️  Low memory detected. Consider reducing batch size.")
    if disk_free_gb < 10:
        logger.warning("⚠️  Low disk space. Ensure sufficient space for vector store.")

def setup_environment():
    """Setup environment variables and paths."""
    logger = logging.getLogger(__name__)
    
    # Set environment variables for optimal performance
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Avoid tokenizer warnings
    os.environ['OMP_NUM_THREADS'] = str(min(8, psutil.cpu_count()))  # Limit OpenMP threads
    
    # CUDA settings if available
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            logger.info(f"🚀 CUDA available with {gpu_count} GPU(s)")
            
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                logger.info(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            logger.info("💻 Using CPU for inference")
    except ImportError:
        logger.info("💻 PyTorch not available, using CPU")

def run_production_ingestion(
    docs_root: Path,
    vector_store_dir: Path,
    modernbert_path: Optional[Path] = None,
    batch_size: int = 32,
    max_workers: Optional[int] = None
):
    """Run production ingestion with optimizations."""
    logger = logging.getLogger(__name__)
    
    start_time = time.time()
    
    try:
        # Import after environment setup
        from rag.config import Config
        from rag.embeddings import get_embeddings
        from rag.pipeline.ingest import IngestPipeline
        
        # Create configuration
        config = Config(
            docs_root=docs_root,
            vector_store_dir=vector_store_dir,
            modernbert_model_path=str(modernbert_path) if modernbert_path else None
        )
        
        logger.info("PRODUCTION INGESTION CONFIGURATION")
        logger.info("=" * 60)
        logger.info(f"Documentation root: {config.docs_root}")
        logger.info(f"Vector store path: {config.vector_store_path()}")
        logger.info(f"Chunk size: {config.chunk_size}")
        logger.info(f"Chunk overlap: {config.chunk_overlap}")
        logger.info(f"Collection name: {config.chroma_collection}")
        if modernbert_path:
            logger.info(f"ModernBERT path: {modernbert_path}")
        logger.info("=" * 60)
        
        # Check documents
        file_count = config.count_markdown_files()
        if file_count == 0:
            logger.error("❌ No markdown files found to process")
            return False
        
        logger.info(f"📚 Found {file_count} markdown files to process")
        
        # Estimate processing time and resources
        estimated_chunks = file_count * 5  # Rough estimate
        estimated_time_min = estimated_chunks / 100  # Very rough estimate
        logger.info(f"📊 Estimated {estimated_chunks} chunks (~{estimated_time_min:.1f} min processing)")
        
        # Initialize embeddings
        logger.info("🤖 Initializing ModernBERT embeddings...")
        embeddings_start = time.time()
        
        embeddings = get_embeddings(
            model_name=str(modernbert_path) if modernbert_path else None,
            embedding_dim=768
        )
        
        embeddings_time = time.time() - embeddings_start
        logger.info(f"✅ Embeddings initialized in {embeddings_time:.2f} seconds")
        
        # Run ingestion pipeline
        logger.info("🔄 Starting production ingestion pipeline...")
        ingestion_start = time.time()
        
        vector_store = IngestPipeline.run(
            config=config,
            embeddings=embeddings,
            persist_directory=None,  # Use config default
        )
        
        ingestion_time = time.time() - ingestion_start
        total_time = time.time() - start_time
        
        # Success metrics
        logger.info("=" * 60)
        logger.info("INGESTION COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info(f"📊 Files processed: {file_count}")
        logger.info(f"⏱️  Embeddings setup: {embeddings_time:.2f} seconds")
        logger.info(f"⏱️  Ingestion time: {ingestion_time:.2f} seconds")
        logger.info(f"⏱️  Total time: {total_time:.2f} seconds")
        logger.info(f"📁 Vector store: {config.vector_store_path()}")
        logger.info(f"📚 Collection: {config.chroma_collection}")
        
        # Performance metrics
        docs_per_second = file_count / ingestion_time if ingestion_time > 0 else 0
        logger.info(f"🚀 Processing rate: {docs_per_second:.2f} docs/second")
        
        # Test the vector store
        logger.info("🧪 Testing vector store...")
        try:
            results = vector_store.similarity_search("NVIDIA GPU", k=3)
            if results:
                logger.info(f"✅ Vector store test successful - found {len(results)} results")
                for i, result in enumerate(results):
                    source = result.metadata.get('source', 'unknown')
                    logger.info(f"   {i+1}. {source}")
            else:
                logger.warning("⚠️  Vector store test returned no results")
        except Exception as e:
            logger.error(f"❌ Vector store test failed: {e}")
        
        logger.info("=" * 60)
        return True
        
    except Exception as e:
        logger.error(f"❌ Production ingestion failed: {e}", exc_info=True)
        return False

def main():
    """Main production ingestion function."""
    parser = argparse.ArgumentParser(description="Production NVIDIA Docs Ingestion")
    
    parser.add_argument("--docs-root", type=Path, required=True,
                       help="Path to documentation root directory")
    parser.add_argument("--vector-store", type=Path, required=True,
                       help="Path to vector store directory")
    parser.add_argument("--modernbert-path", type=Path,
                       help="Path to local ModernBERT model")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for processing")
    parser.add_argument("--max-workers", type=int,
                       help="Maximum worker threads")
    parser.add_argument("--log-level", default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--log-file", type=Path,
                       help="Log file path")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 Starting NVIDIA Documentation Production Ingestion")
    
    # Check system resources
    check_system_resources()
    
    # Setup environment
    setup_environment()
    
    # Validate inputs
    if not args.docs_root.exists():
        logger.error(f"❌ Documentation root not found: {args.docs_root}")
        return 1
    
    if args.modernbert_path and not args.modernbert_path.exists():
        logger.error(f"❌ ModernBERT path not found: {args.modernbert_path}")
        return 1
    
    # Run ingestion
    success = run_production_ingestion(
        docs_root=args.docs_root,
        vector_store_dir=args.vector_store,
        modernbert_path=args.modernbert_path,
        batch_size=args.batch_size,
        max_workers=args.max_workers
    )
    
    if success:
        logger.info("🎉 Production ingestion completed successfully!")
        
        # Create migration package
        logger.info("📦 Creating migration package...")
        try:
            from migrate_vector_db import create_vector_db_package
            
            package_path = create_vector_db_package(
                args.vector_store / "vector_store",
                args.vector_store.parent / "packages",
                include_metadata=True
            )
            
            logger.info(f"📦 Migration package ready: {package_path}")
            logger.info("💡 Transfer this package to your Mac for local use")
            
        except Exception as e:
            logger.warning(f"⚠️  Failed to create migration package: {e}")
        
        return 0
    else:
        logger.error("❌ Production ingestion failed!")
        return 1

if __name__ == "__main__":
    exit(main())