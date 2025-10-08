#!/usr/bin/env python3
"""
Comprehensive Test Suite for NVIDIA Docs RAG System
Tests embeddings, ingestion, and cross-platform compatibility.
"""

import logging
import tempfile
import time
import unittest
from pathlib import Path
from typing import List, Dict, Any

class TestModernBERT(unittest.TestCase):
    """Test ModernBERT embeddings implementation."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        logging.basicConfig(level=logging.INFO)
        cls.logger = logging.getLogger(__name__)
    
    def test_embeddings_initialization(self):
        """Test embeddings can be initialized."""
        from rag.embeddings import get_embeddings
        
        self.logger.info("Testing embeddings initialization...")
        embeddings = get_embeddings()
        self.assertIsNotNone(embeddings)
        self.logger.info("✅ Embeddings initialized successfully")
    
    def test_embeddings_dimensions(self):
        """Test embedding dimensions are correct."""
        from rag.embeddings import get_embeddings
        
        self.logger.info("Testing embedding dimensions...")
        embeddings = get_embeddings()
        
        test_text = "NVIDIA GPUs accelerate AI workloads."
        embedding = embeddings.embed_query(test_text)
        
        self.assertIsInstance(embedding, list)
        self.assertEqual(len(embedding), 768)  # ModernBERT-base dimension
        self.logger.info(f"✅ Embedding dimension correct: {len(embedding)}")
    
    def test_embeddings_consistency(self):
        """Test embeddings are consistent across calls."""
        from rag.embeddings import get_embeddings
        
        self.logger.info("Testing embedding consistency...")
        embeddings = get_embeddings()
        
        test_text = "CUDA programming enables parallel computing."
        
        embedding1 = embeddings.embed_query(test_text)
        embedding2 = embeddings.embed_query(test_text)
        
        # Should be identical for the same text
        for i, (v1, v2) in enumerate(zip(embedding1, embedding2)):
            self.assertAlmostEqual(v1, v2, places=6, 
                                 msg=f"Inconsistent embedding at index {i}")
        
        self.logger.info("✅ Embeddings are consistent")
    
    def test_embeddings_similarity(self):
        """Test semantic similarity works correctly."""
        from rag.embeddings import get_embeddings
        import numpy as np
        
        self.logger.info("Testing semantic similarity...")
        embeddings = get_embeddings()
        
        # Related texts
        text1 = "NVIDIA GPUs are used for machine learning."
        text2 = "Machine learning benefits from NVIDIA GPU acceleration."
        text3 = "The weather is sunny today."
        
        emb1 = np.array(embeddings.embed_query(text1))
        emb2 = np.array(embeddings.embed_query(text2))
        emb3 = np.array(embeddings.embed_query(text3))
        
        # Calculate cosine similarities
        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        
        sim_related = cosine_similarity(emb1, emb2)
        sim_unrelated = cosine_similarity(emb1, emb3)
        
        # Related texts should be more similar
        self.assertGreater(sim_related, sim_unrelated)
        self.logger.info(f"✅ Semantic similarity working: related={sim_related:.3f}, unrelated={sim_unrelated:.3f}")

class TestIngestionPipeline(unittest.TestCase):
    """Test ingestion pipeline functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        logging.basicConfig(level=logging.INFO)
        cls.logger = logging.getLogger(__name__)
    
    def test_config_creation(self):
        """Test configuration creation."""
        from rag.config import Config
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config = Config(docs_root=temp_path)
            
            self.assertEqual(config.docs_root, temp_path)
            self.assertEqual(config.chunk_size, 1200)
            self.assertEqual(config.chunk_overlap, 200)
            self.logger.info("✅ Configuration creation works")
    
    def test_document_processing(self):
        """Test document loading and chunking."""
        from rag.config import Config
        from rag.pipeline.ingest import IngestPipeline
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test documents
            test_docs = [
                ("doc1.md", "# NVIDIA GPU\n\nNVIDIA GPUs are powerful processors for AI workloads."),
                ("doc2.md", "# CUDA Programming\n\nCUDA enables parallel computing on NVIDIA GPUs."),
                ("doc3.md", "# DGX Systems\n\nDGX systems are purpose-built for enterprise AI.")
            ]
            
            for filename, content in test_docs:
                (temp_path / filename).write_text(content)
            
            config = Config(docs_root=temp_path)
            
            # Test document loading
            files = list(config.iter_markdown_files())
            self.assertEqual(len(files), 3)
            
            # Test document processing
            documents = list(IngestPipeline._load_documents(files, config))
            self.assertEqual(len(documents), 3)
            
            # Test chunking
            chunked_docs = IngestPipeline._chunk_documents(config=config, documents=documents)
            self.assertGreater(len(chunked_docs), 0)
            
            self.logger.info(f"✅ Document processing works: {len(files)} files, {len(chunked_docs)} chunks")

class TestCrossPlatformCompatibility(unittest.TestCase):
    """Test cross-platform compatibility."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        logging.basicConfig(level=logging.INFO)
        cls.logger = logging.getLogger(__name__)
    
    def test_vector_store_creation(self):
        """Test vector store can be created and persisted."""
        from rag.config import Config
        from rag.embeddings import get_embeddings
        from rag.pipeline.ingest import IngestPipeline
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            docs_path = temp_path / "docs"
            vector_path = temp_path / "vector_store"
            
            docs_path.mkdir()
            
            # Create test document
            (docs_path / "test.md").write_text("""
# Test Document

This is a test document for NVIDIA GPU documentation.

## GPU Computing

NVIDIA GPUs accelerate machine learning and AI workloads.

## CUDA Programming

CUDA is a parallel computing platform and programming model.
""")
            
            config = Config(docs_root=docs_path, vector_store_dir=vector_path)
            embeddings = get_embeddings()
            
            # Run ingestion
            vector_store = IngestPipeline.run(
                config=config,
                embeddings=embeddings,
                persist_directory=vector_path
            )
            
            # Test vector store exists
            self.assertTrue(vector_path.exists())
            self.assertTrue((vector_path / "chroma.sqlite3").exists())
            
            # Test similarity search
            results = vector_store.similarity_search("GPU computing", k=2)
            self.assertGreater(len(results), 0)
            
            self.logger.info(f"✅ Vector store creation works: {len(results)} results found")
    
    def test_migration_package(self):
        """Test migration package creation."""
        from migrate_vector_db import create_vector_db_package
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            vector_store = temp_path / "vector_store"
            output_dir = temp_path / "packages"
            
            # Create mock vector store
            vector_store.mkdir()
            (vector_store / "chroma.sqlite3").write_text("mock database")
            (vector_store / "test_file").write_text("test content")
            
            # Create package
            package_path = create_vector_db_package(vector_store, output_dir)
            
            self.assertTrue(package_path.exists())
            self.assertTrue(package_path.name.endswith(".tar.gz"))
            
            self.logger.info(f"✅ Migration package creation works: {package_path.name}")

def run_performance_benchmark():
    """Run performance benchmark."""
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 Running Performance Benchmark")
    logger.info("=" * 50)
    
    from rag.embeddings import get_embeddings
    import time
    
    # Test different batch sizes
    test_texts = [
        f"This is test document number {i} about NVIDIA GPU computing and machine learning applications."
        for i in range(100)
    ]
    
    embeddings = get_embeddings()
    
    # Single document timing
    start_time = time.time()
    single_embedding = embeddings.embed_query(test_texts[0])
    single_time = time.time() - start_time
    
    # Batch timing
    start_time = time.time()
    batch_embeddings = embeddings.embed_documents(test_texts)
    batch_time = time.time() - start_time
    
    logger.info(f"📊 Single document: {single_time:.3f} seconds")
    logger.info(f"📊 Batch (100 docs): {batch_time:.3f} seconds")
    logger.info(f"📊 Batch efficiency: {(single_time * 100) / batch_time:.2f}x faster")
    logger.info(f"📊 Throughput: {len(test_texts) / batch_time:.1f} docs/second")
    logger.info(f"📊 Embedding dimension: {len(single_embedding)}")

def main():
    """Run all tests."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    logger = logging.getLogger(__name__)
    
    logger.info("🧪 Starting NVIDIA Docs RAG Test Suite")
    logger.info("=" * 60)
    
    # Run unit tests
    test_classes = [
        TestModernBERT,
        TestIngestionPipeline,
        TestCrossPlatformCompatibility
    ]
    
    suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestClass(test_class)
        suite.addTests(tests)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Run performance benchmark
    run_performance_benchmark()
    
    # Summary
    logger.info("=" * 60)
    if result.wasSuccessful():
        logger.info("🎉 All tests passed!")
        logger.info("✅ System is ready for production deployment")
        return 0
    else:
        logger.error("❌ Some tests failed!")
        logger.error(f"Failures: {len(result.failures)}")
        logger.error(f"Errors: {len(result.errors)}")
        return 1

if __name__ == "__main__":
    exit(main())