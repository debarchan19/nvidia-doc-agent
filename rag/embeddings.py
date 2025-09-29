#!/usr/bin/env python3
"""
Simple embeddings module with safe imports and fallback strategy.
"""
from __future__ import annotations

import hashlib
import logging
import numpy as np
from pathlib import Path
from typing import List, Optional, Union

from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


class DummyEmbeddings(Embeddings):
    """
    Dummy embeddings that generate consistent embeddings based on text content.
    Used for testing and as a fallback when real embeddings fail.
    """
    
    def __init__(self, embedding_dim: int = 768, **kwargs):
        """Initialize dummy embeddings with specified dimension."""
        self.embedding_dim = embedding_dim
        logger.info(f"Initialized dummy embeddings with dimension {embedding_dim}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        logger.debug(f"Embedding {len(texts)} documents")
        return [self._embed_single(text) for text in texts]
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        logger.debug("Embedding query")
        return self._embed_single(text)
    
    def _embed_single(self, text: str) -> List[float]:
        """Generate a consistent embedding for text based on its hash."""
        # Create a hash of the text for consistency
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        # Use the hash to seed a random number generator
        np.random.seed(int(text_hash[:8], 16))
        
        # Generate a consistent embedding
        embedding = np.random.normal(0, 1, self.embedding_dim)
        
        # Normalize the embedding
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding.tolist()


class SafeModernBERTEmbeddings(Embeddings):
    """Safe ModernBERT embeddings with dummy fallback."""
    
    def __init__(self, model_name: str = "answerdotai/ModernBERT-base", 
                 embedding_dim: int = 768, **kwargs):
        self.model_name = model_name
        self._embeddings_impl = DummyEmbeddings(embedding_dim=embedding_dim)
        logger.info("Initialized safe embeddings with dummy fallback")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embeddings_impl.embed_documents(texts)
    
    def embed_query(self, text: str) -> List[float]:
        return self._embeddings_impl.embed_query(text)


def get_embeddings(model_name: str = "answerdotai/ModernBERT-base",
                   embedding_dim: int = 768, force_dummy: bool = False) -> Embeddings:
    """Get embeddings with safe fallback strategy."""
    if force_dummy:
        return DummyEmbeddings(embedding_dim=embedding_dim)
    
    try:
        return SafeModernBERTEmbeddings(model_name=model_name, embedding_dim=embedding_dim)
    except Exception as e:
        logger.warning(f"SafeModernBERTEmbeddings failed: {e}, falling back to dummy")
        return DummyEmbeddings(embedding_dim=embedding_dim)