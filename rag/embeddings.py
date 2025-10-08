#!/usr/bin/env python3
"""
Embeddings module with proper ModernBERT implementation for RAG system.
"""
from __future__ import annotations

import logging
import warnings
from typing import List, Optional

from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

# Suppress warnings from transformers/torch during loading
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class ModernBERTEmbeddings(Embeddings):
    """
    ModernBERT embeddings using sentence-transformers.
    Provides high-quality 768-dimensional embeddings for RAG applications.
    """
    
    def __init__(self, model_name: str = "answerdotai/ModernBERT-base", **kwargs):
        """Initialize ModernBERT embeddings."""
        self.model_name = model_name
        self._model = None
        logger.info(f"Initializing ModernBERT embeddings with model: {model_name}")
        
        # Load the model
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model."""
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading {self.model_name}...")
            self._model = SentenceTransformer(self.model_name)
            logger.info(f"✓ Successfully loaded {self.model_name}")
            logger.info(f"  - Embedding dimension: {self._model.get_sentence_embedding_dimension()}")
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for ModernBERT embeddings. "
                "Install with: pip install sentence-transformers"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load ModernBERT model {self.model_name}: {e}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        if not texts:
            return []
        
        logger.debug(f"Embedding {len(texts)} documents")
        
        # Filter out empty texts
        non_empty_texts = [text if text.strip() else " " for text in texts]
        
        try:
            embeddings = self._model.encode(
                non_empty_texts,
                convert_to_numpy=True,
                show_progress_bar=len(non_empty_texts) > 100,
                batch_size=32
            )
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error embedding documents: {e}")
            raise
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        if not text.strip():
            text = " "  # Handle empty queries
        
        logger.debug("Embedding query")
        try:
            embedding = self._model.encode([text], convert_to_numpy=True)
            return embedding[0].tolist()
        except Exception as e:
            logger.error(f"Error embedding query: {e}")
            raise


class HuggingFaceEmbeddings(Embeddings):
    """
    Alternative implementation using HuggingFace transformers directly.
    More control over the model but requires more setup.
    """
    
    def __init__(self, model_name: str = "answerdotai/ModernBERT-base", **kwargs):
        """Initialize HuggingFace embeddings."""
        self.model_name = model_name
        self._tokenizer = None
        self._model = None
        logger.info(f"Initializing HuggingFace embeddings with model: {model_name}")
        
        # Load the model
        self._load_model()
    
    def _load_model(self):
        """Load the HuggingFace model and tokenizer."""
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
            
            logger.info(f"Loading tokenizer and model for {self.model_name}...")
            
            # Handle local model path
            if self.model_name.startswith("/"):
                logger.info(f"Loading local model from: {self.model_name}")
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=True)
                self._model = AutoModel.from_pretrained(self.model_name, local_files_only=True)
            else:
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoModel.from_pretrained(self.model_name)
            
            self._model.eval()  # Set to evaluation mode
            
            # Check if GPU/MPS is available
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
                logger.info("Using Apple Metal Performance Shaders (MPS)")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
                logger.info("Using CUDA GPU")
            else:
                self.device = torch.device("cpu")
                logger.info("Using CPU")
            
            self._model.to(self.device)
            
            logger.info(f"✓ Successfully loaded {self.model_name}")
            logger.info(f"  - Device: {self.device}")
            logger.info(f"  - Hidden size: {self._model.config.hidden_size}")
            logger.info(f"  - Model type: {self._model.config.model_type}")
            
        except ImportError:
            raise ImportError(
                "transformers and torch are required for HuggingFace embeddings. "
                "Install with: pip install transformers torch"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load HuggingFace model {self.model_name}: {e}")
    
    def _mean_pooling(self, model_output, attention_mask):
        """Apply mean pooling to get sentence embeddings."""
        import torch
        
        token_embeddings = model_output[0]  # First element contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        if not texts:
            return []
        
        logger.debug(f"Embedding {len(texts)} documents")
        
        try:
            import torch
            
            # Handle empty or whitespace-only texts
            processed_texts = [text.strip() if text.strip() else "[EMPTY]" for text in texts]
            
            # Tokenize with proper settings for ModernBERT
            encoded_input = self._tokenizer(
                processed_texts, 
                padding=True, 
                truncation=True, 
                max_length=512,  # Reasonable limit for document chunks
                return_tensors='pt'
            )
            encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
            
            # Generate embeddings
            with torch.no_grad():
                model_output = self._model(**encoded_input)
                embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
                
                # Normalize embeddings for better similarity computation
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                
            return embeddings.cpu().numpy().tolist()
            
        except Exception as e:
            logger.error(f"Error embedding documents: {e}")
            raise
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        return self.embed_documents([text])[0]


def get_embeddings(
    model_name: str = None,
    embedding_dim: int = 768,
    use_huggingface: bool = True,  # Use HuggingFace for local model
    **kwargs
) -> Embeddings:
    """
    Get embeddings with cross-platform ModernBERT implementation.
    
    Args:
        model_name: Path to the local ModernBERT model or HuggingFace model name.
                   If None, will auto-detect local ModernBERT-base directory.
        embedding_dim: Expected embedding dimension (for validation)
        use_huggingface: Whether to use HuggingFace transformers directly
        **kwargs: Additional arguments passed to the embeddings class
    
    Returns:
        Embeddings instance
        
    Raises:
        ImportError: If required dependencies are not installed
        RuntimeError: If model loading fails
    """
    # Auto-detect local ModernBERT model path
    if model_name is None:
        from pathlib import Path
        project_root = Path(__file__).parent.parent
        possible_paths = [
            project_root / "ModernBERT-base",  # In project root
            Path("./ModernBERT-base"),        # Current directory
            Path("../ModernBERT-base"),       # Parent directory
        ]
        
        for path in possible_paths:
            if path.exists() and (path / "config.json").exists():
                model_name = str(path.resolve())
                logger.info(f"Found local ModernBERT model at: {model_name}")
                break
        else:
            # Fallback to HuggingFace model
            model_name = "answerdotai/ModernBERT-base"
            logger.info("No local ModernBERT found, using HuggingFace model")
    
    logger.info(f"Initializing embeddings with model: {model_name}")
    
    try:
        if use_huggingface:
            embeddings = HuggingFaceEmbeddings(model_name=model_name, **kwargs)
        else:
            embeddings = ModernBERTEmbeddings(model_name=model_name, **kwargs)
        
        logger.info("✓ Embeddings initialized successfully")
        return embeddings
        
    except ImportError as e:
        logger.error(f"Missing dependencies for embeddings: {e}")
        logger.info("To install required dependencies:")
        logger.info("  pip install sentence-transformers  # For sentence-transformers")
        logger.info("  pip install transformers torch     # For HuggingFace transformers")
        raise
    
    except Exception as e:
        logger.error(f"Failed to initialize embeddings: {e}")
        raise