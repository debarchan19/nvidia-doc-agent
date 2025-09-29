#!/usr/bin/env python3
"""
Tests for the RAG pipeline components.
"""
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from rag.pipeline.retrieve import Retriever
from rag.agent.tools import search_nvidia_docs, get_retrieval_stats
from rag.config import config


class TestRetriever:
    """Test cases for the Retriever class."""
    
    def test_retriever_initialization(self):
        """Test that retriever initializes properly."""
        retriever = Retriever()
        assert retriever.collection_name == config.chroma_collection
        assert retriever.top_k == 5
        assert retriever.max_distance == 2.0
        assert retriever.embeddings is not None
    
    def test_retriever_with_custom_params(self):
        """Test retriever with custom parameters."""
        retriever = Retriever(
            collection_name="test_collection",
            top_k=10,
            max_distance=1.5
        )
        assert retriever.collection_name == "test_collection"
        assert retriever.top_k == 10
        assert retriever.max_distance == 1.5
    
    def test_search_with_no_vector_store(self):
        """Test search behavior when vector store is not available."""
        retriever = Retriever()
        retriever._vector_store = None  # Simulate failed initialization
        
        results = retriever.search("test query")
        assert results == []
    
    @patch('rag.pipeline.retrieve.Chroma')
    def test_search_with_mock_vector_store(self, mock_chroma):
        """Test search functionality with mocked vector store."""
        # Mock the vector store
        mock_vector_store = MagicMock()
        mock_chroma.return_value = mock_vector_store
        
        # Mock search results
        mock_doc = MagicMock()
        mock_doc.page_content = "Test content about NVIDIA GPU"
        mock_doc.metadata = {"source": "test.md"}
        mock_vector_store.similarity_search_with_score.return_value = [
            (mock_doc, 0.2)  # Low distance = high similarity
        ]
        
        retriever = Retriever()
        results = retriever.search("GPU")
        
        assert len(results) == 1
        assert results[0].page_content == "Test content about NVIDIA GPU"
    
    def test_search_with_metadata(self):
        """Test search with metadata extraction."""
        retriever = Retriever()
        retriever._vector_store = None  # Will return empty results
        
        results = retriever.search_with_metadata("test query")
        assert results == []
    
    def test_get_collection_info_no_store(self):
        """Test collection info when no vector store is available."""
        retriever = Retriever()
        retriever._vector_store = None
        
        info = retriever.get_collection_info()
        assert info["status"] == "not_initialized"
        assert info["count"] == 0
    
    def test_health_check_no_store(self):
        """Test health check when no vector store is available."""
        retriever = Retriever()
        retriever._vector_store = None
        
        assert not retriever.health_check()


class TestAgentTools:
    """Test cases for agent tools."""
    
    @patch('rag.agent.tools.Retriever')
    def test_search_nvidia_docs_success(self, mock_retriever_class):
        """Test successful document search."""
        # Mock retriever instance
        mock_retriever = MagicMock()
        mock_retriever_class.return_value = mock_retriever
        mock_retriever.health_check.return_value = True
        mock_retriever.search_with_metadata.return_value = [
            {
                'content': 'NVIDIA GPU information',
                'source': 'gpu_guide.md',
                'metadata': {'type': 'guide'},
                'relevance_score': 0.9
            }
        ]
        
        result = search_nvidia_docs.invoke({"query": "GPU performance"})
        
        assert len(result) == 1
        assert result[0]['content'] == 'NVIDIA GPU information'
        assert result[0]['source'] == 'gpu_guide.md'
    
    @patch('rag.agent.tools.Retriever')
    def test_search_nvidia_docs_unhealthy_db(self, mock_retriever_class):
        """Test search when vector database is unhealthy."""
        mock_retriever = MagicMock()
        mock_retriever_class.return_value = mock_retriever
        mock_retriever.health_check.return_value = False
        
        result = search_nvidia_docs.invoke({"query": "GPU performance"})
        
        assert len(result) == 1
        assert "Vector database is not available" in result[0]['content']
        assert result[0]['source'] == 'system_error'
    
    @patch('rag.agent.tools.Retriever')
    def test_search_nvidia_docs_no_results(self, mock_retriever_class):
        """Test search when no results are found."""
        mock_retriever = MagicMock()
        mock_retriever_class.return_value = mock_retriever
        mock_retriever.health_check.return_value = True
        mock_retriever.search_with_metadata.return_value = []
        
        result = search_nvidia_docs.invoke({"query": "nonexistent topic"})
        
        assert len(result) == 1
        assert "No relevant documents found" in result[0]['content']
        assert result[0]['source'] == 'no_results'
    
    @patch('rag.agent.tools.Retriever')
    def test_search_nvidia_docs_exception(self, mock_retriever_class):
        """Test search when an exception occurs."""
        mock_retriever_class.side_effect = Exception("Database error")
        
        result = search_nvidia_docs.invoke({"query": "GPU performance"})
        
        assert len(result) == 1
        assert "Error occurred while searching" in result[0]['content']
        assert result[0]['source'] == 'system_error'
    
    @patch('rag.agent.tools.Retriever')
    def test_get_retrieval_stats_success(self, mock_retriever_class):
        """Test successful retrieval stats."""
        mock_retriever = MagicMock()
        mock_retriever_class.return_value = mock_retriever
        mock_retriever.get_collection_info.return_value = {
            'status': 'ready',
            'document_count': 100
        }
        mock_retriever.health_check.return_value = True
        
        result = get_retrieval_stats.invoke({})
        
        assert result['status'] == 'ready'
        assert result['document_count'] == 100
        assert result['health_check'] is True
    
    @patch('rag.agent.tools.Retriever')
    def test_get_retrieval_stats_exception(self, mock_retriever_class):
        """Test retrieval stats when an exception occurs."""
        mock_retriever_class.side_effect = Exception("Stats error")
        
        result = get_retrieval_stats.invoke({})
        
        assert result['status'] == 'error'
        assert 'Stats error' in result['error']
        assert result['health_check'] is False


class TestPipeline:
    """Integration tests for the complete pipeline."""
    
    def test_pipeline_components_import(self):
        """Test that all pipeline components can be imported."""
        from rag.pipeline.retrieve import Retriever
        from rag.agent.tools import search_nvidia_docs, get_retrieval_stats, AGENT_TOOLS
        
        assert Retriever is not None
        assert search_nvidia_docs is not None
        assert get_retrieval_stats is not None
        assert len(AGENT_TOOLS) == 2
