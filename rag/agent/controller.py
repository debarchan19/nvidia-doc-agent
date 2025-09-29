#!/usr/bin/env python3
"""
Agent controller for orchestrating RAG operations.
"""
from __future__ import annotations

import logging
from typing import Dict, Any, List, Optional

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from ..pipeline.retrieve import Retriever
from .tools import AGENT_TOOLS, search_nvidia_docs, get_retrieval_stats

logger = logging.getLogger(__name__)


class AgentController:
    """Controller for orchestrating RAG agent operations."""
    
    def __init__(self, model_name: str = "llama3.2:3b"):
        """
        Initialize the agent controller.
        
        Args:
            model_name: Ollama model name to use
        """
        self.model_name = model_name
        self.llm = ChatOllama(model=model_name)
        self.retriever = Retriever()
        
        logger.info(f"Initialized agent controller with model: {model_name}")
    
    def search_documents(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search for documents using the retriever.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            List of search results
        """
        try:
            results = search_nvidia_docs.invoke({
                "query": query,
                "max_results": max_results
            })
            logger.info(f"Search for '{query}' returned {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get vector database statistics."""
        try:
            return get_retrieval_stats.invoke({})
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {"status": "error", "error": str(e)}
    
    def generate_response(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """
        Generate a response using the LLM with retrieved context.
        
        Args:
            query: User's question
            context_docs: Retrieved documents for context
            
        Returns:
            Generated response
        """
        try:
            # Prepare context from retrieved documents
            context_parts = []
            for i, doc in enumerate(context_docs[:5]):  # Limit to top 5 docs
                if doc.get('source') != 'no_results':
                    source = doc.get('source', 'unknown')
                    content = doc.get('content', '')[:500]  # Limit content length
                    context_parts.append(f"Document {i+1} ({source}):\n{content}")
            
            context = "\n\n".join(context_parts) if context_parts else "No relevant documents found."
            
            # Create prompt with context
            system_prompt = """
You are an AI assistant specialized in NVIDIA technology and documentation.
Use the provided document context to answer the user's question accurately.
Always cite the document sources in your response.
If the context doesn't contain relevant information, say so clearly.
"""
            
            user_prompt = f"""
Context from NVIDIA Documentation:
{context}

Question: {query}

Please provide a detailed answer based on the context above. Cite the document sources you use.
"""
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = self.llm.invoke(messages)
            return response.content
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error generating response: {str(e)}"
    
    def query(self, question: str, max_results: int = 5) -> Dict[str, Any]:
        """
        Process a complete query with search and generation.
        
        Args:
            question: User's question
            max_results: Maximum search results to use
            
        Returns:
            Dictionary with response and metadata
        """
        logger.info(f"Processing query: {question}")
        
        # Search for relevant documents
        search_results = self.search_documents(question, max_results)
        
        # Generate response using retrieved context
        response = self.generate_response(question, search_results)
        
        # Prepare result
        result = {
            "question": question,
            "response": response,
            "sources": [
                {
                    "source": doc.get('source', 'unknown'),
                    "relevance_score": doc.get('relevance_score', 0.0)
                }
                for doc in search_results
                if doc.get('source') != 'no_results'
            ],
            "search_results_count": len(search_results),
            "model_used": self.model_name
        }
        
        logger.info(f"Query processed successfully with {len(result['sources'])} sources")
        return result
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check of all system components.
        
        Returns:
            Health status information
        """
        health_info = {
            "controller": "healthy",
            "model": self.model_name,
            "timestamp": None
        }
        
        try:
            # Check database health
            db_stats = self.get_database_stats()
            health_info["database"] = db_stats
            
            # Check retriever health
            health_info["retriever"] = self.retriever.health_check()
            
            # Test LLM
            test_response = self.llm.invoke([HumanMessage(content="Hello")])
            health_info["llm"] = "healthy" if test_response else "error"
            
            # Test search
            test_search = self.search_documents("test", max_results=1)
            health_info["search"] = "healthy" if isinstance(test_search, list) else "error"
            
        except Exception as e:
            health_info["error"] = str(e)
            health_info["controller"] = "error"
        
        return health_info
