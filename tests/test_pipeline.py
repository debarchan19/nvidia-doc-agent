"""Tests for the RAG pipeline components."""
import pytest
from unittest.mock import patch, MagicMock

from rag.pipeline.retrieve import Retriever
from rag.agent.tools import search_nvidia_docs, get_retrieval_stats


def test_retriever_initialization():
    """Test that retriever initializes properly."""
    retriever = Retriever()
    assert retriever.top_k == 5
    assert retriever.max_distance == 2.0

def test_search_with_no_vector_store():
    """Test search behavior when vector store is not available."""
    retriever = Retriever()
    retriever._vector_store = None
    assert retriever.search("test query") == []

@patch('rag.agent.tools.Retriever')
def test_search_nvidia_docs_success(mock_retriever_class):
    """Test successful document search."""
    mock_retriever = MagicMock()
    mock_retriever_class.return_value = mock_retriever
    mock_retriever.health_check.return_value = True
    mock_retriever.search_with_metadata.return_value = [
        {'content': 'NVIDIA GPU info', 'source': 'gpu_guide.md'}
    ]
    
    result = search_nvidia_docs.invoke({"query": "GPU performance"})
    assert len(result) == 1
    assert result[0]['source'] == 'gpu_guide.md'

def test_pipeline_components_import():
    """Test that all pipeline components can be imported."""
    from rag.pipeline.retrieve import Retriever
    from rag.agent.tools import AGENT_TOOLS
    
    assert Retriever is not None
    assert len(AGENT_TOOLS) == 2
