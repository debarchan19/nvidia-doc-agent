#!/usr/bin/env python3
"""
Retriever module for searching NVIDIA documentation in vector database.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document

from ..config import config
from ..embeddings import get_embeddings

logger = logging.getLogger(__name__)


class Retriever:
    """
    Retriever for searching NVIDIA docs in ChromaDB vector store.
    
    This class handles similarity search and document retrieval from the vector database.
    """
    
    def __init__(
        self,
        collection_name: str = None,
        vector_store_dir: Path = None,
        top_k: int = 5,
        max_distance: float = 2.0  # Direct distance threshold, lower = more similar
    ):
        """
        Initialize the retriever.
        
        Args:
            collection_name: Name of the ChromaDB collection
            vector_store_dir: Path to vector store directory
            top_k: Number of top results to return
            max_distance: Maximum distance threshold for results (lower = more similar)
        """
        self.collection_name = collection_name or config.chroma_collection
        self.vector_store_dir = vector_store_dir or config.vector_store_path()
        self.top_k = top_k
        self.max_distance = max_distance
        
        # Initialize embeddings
        self.embeddings = get_embeddings()
        logger.info(f"Initialized embeddings: {self.embeddings}")
        
        # Initialize ChromaDB vector store
        self._vector_store = None
        self._initialize_vector_store()
    
    def _initialize_vector_store(self) -> None:
        """Initialize the ChromaDB vector store."""
        try:
            self._vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=str(self.vector_store_dir)
            )
            logger.info(f"Initialized vector store at {self.vector_store_dir}")
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            self._vector_store = None
    
    def search(self, query: str, top_k: Optional[int] = None) -> List[Document]:
        """Search for documents similar to the query."""
        if not self._vector_store:
            return []
        
        try:
            results = self._vector_store.similarity_search_with_score(
                query=query, k=top_k or self.top_k
            )
            return [doc for doc, score in results if score <= self.max_distance]
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []
    
    def search_with_metadata(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Search for documents and return structured results with metadata.
        
        Args:
            query: Search query string
            top_k: Number of results to return
        
        Returns:
            List of dictionaries containing document content and metadata
        """
        documents = self.search(query, top_k)
        
        results = []
        for doc in documents:
            result = {
                'content': doc.page_content,
                'metadata': doc.metadata,
                'source': doc.metadata.get('source', 'unknown'),
                'relevance_score': doc.metadata.get('score', 0.0)
            }
            results.append(result)
        
        return results
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the vector store collection."""
        if not self._vector_store:
            return {"status": "not_initialized", "count": 0}
        
        try:
            # Get collection statistics
            collection = self._vector_store._collection
            count = collection.count()
            
            return {
                "status": "ready",
                "collection_name": self.collection_name,
                "document_count": count,
                "vector_store_dir": str(self.vector_store_dir)
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {"status": "error", "error": str(e)}
    
    def health_check(self) -> bool:
        """Check if the retriever is working properly."""
        try:
            if not self._vector_store:
                return False
            
            # Try a simple search
            test_results = self.search("test query", top_k=1)
            return True  # If no exception, it's working
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
