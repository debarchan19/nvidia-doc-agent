#!/usr/bin/env python3
"""
Agent tools for the RAG system, including NVIDIA docs retrieval.
"""
from __future__ import annotations

import logging
from typing import Dict, Any, List

from langchain_core.tools import tool

from ..pipeline.retrieve import Retriever

logger = logging.getLogger(__name__)


class AgentTools:
    """Agent tools for the RAG system."""
    
    def __init__(self):
        """Initialize agent tools with retriever."""
        self.retriever = Retriever()
        logger.info("Initialized agent tools with retriever")
    
    def get_retrieval_tool(self):
        """Get the document retrieval tool for LangGraph."""
        return search_nvidia_docs


# Define the LangGraph tool function
@tool
def search_nvidia_docs(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Search the NVIDIA documentation vector database for relevant information.
    
    This tool searches through NVIDIA technical documentation to find relevant 
    information based on the provided query. It returns the most similar documents
    with their content and metadata.
    
    Args:
        query: The search query to find relevant documentation
        max_results: Maximum number of results to return (default: 5)
    
    Returns:
        List of dictionaries containing:
        - content: The document content/text
        - source: The source file name
        - metadata: Additional document metadata
        - relevance_score: How relevant the document is to the query
    """
    logger.info(f"Searching NVIDIA docs for: {query}")
    
    try:
        # Initialize retriever
        retriever = Retriever(top_k=max_results)
        
        # Check if retriever is healthy
        if not retriever.health_check():
            return [{
                "content": "Vector database is not available. Please ensure the NVIDIA docs have been indexed.",
                "source": "system_error",
                "metadata": {"error": "vector_db_unavailable"},
                "relevance_score": 0.0
            }]
        
        # Perform search
        results = retriever.search_with_metadata(query, top_k=max_results)
        
        if not results:
            return [{
                "content": f"No relevant documents found for query: {query}",
                "source": "no_results",
                "metadata": {"query": query},
                "relevance_score": 0.0
            }]
        
        logger.info(f"Found {len(results)} results for query")
        return results
        
    except Exception as e:
        logger.error(f"Error in search_nvidia_docs: {e}")
        return [{
            "content": f"Error occurred while searching: {str(e)}",
            "source": "system_error", 
            "metadata": {"error": str(e)},
            "relevance_score": 0.0
        }]


@tool  
def get_retrieval_stats() -> Dict[str, Any]:
    """
    Get statistics about the NVIDIA docs vector database.
    
    Returns information about the indexed documents including count,
    collection status, and health check results.
    
    Returns:
        Dictionary with database statistics and status information
    """
    try:
        retriever = Retriever()
        info = retriever.get_collection_info()
        info["health_check"] = retriever.health_check()
        return info
        
    except Exception as e:
        logger.error(f"Error getting retrieval stats: {e}")
        return {
            "status": "error",
            "error": str(e),
            "health_check": False
        }


# Tool registry for easy access
AGENT_TOOLS = [
    search_nvidia_docs,
    get_retrieval_stats
]
