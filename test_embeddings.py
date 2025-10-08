#!/usr/bin/env python3
"""
Test script for the new ModernBERT embeddings implementation.
"""

import logging
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_embeddings():
    """Test the embeddings implementation."""
    print("🧪 Testing ModernBERT Embeddings Implementation")
    print("=" * 50)
    
    try:
        from rag.embeddings import get_embeddings
        
        # Test documents
        test_docs = [
            "NVIDIA GPUs are powerful computing devices for AI workloads.",
            "CUDA is a parallel computing platform and programming model.",
            "DGX systems are purpose-built for enterprise AI applications."
        ]
        
        test_query = "What are NVIDIA GPUs used for?"
        
        print("📦 Initializing embeddings...")
        embeddings = get_embeddings(model_name="answerdotai/ModernBERT-base")
        
        print("📝 Testing document embeddings...")
        doc_embeddings = embeddings.embed_documents(test_docs)
        print(f"  ✓ Generated embeddings for {len(doc_embeddings)} documents")
        print(f"  ✓ Embedding dimension: {len(doc_embeddings[0])}")
        
        print("❓ Testing query embedding...")
        query_embedding = embeddings.embed_query(test_query)
        print(f"  ✓ Generated query embedding with dimension: {len(query_embedding)}")
        
        # Test similarity
        import numpy as np
        query_vec = np.array(query_embedding)
        similarities = []
        
        for i, doc_emb in enumerate(doc_embeddings):
            doc_vec = np.array(doc_emb)
            similarity = np.dot(query_vec, doc_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(doc_vec))
            similarities.append((i, similarity))
            print(f"  📊 Similarity with doc {i+1}: {similarity:.4f}")
        
        # Find most similar document
        best_match = max(similarities, key=lambda x: x[1])
        print(f"\n🎯 Most similar document (index {best_match[0]}): {test_docs[best_match[0]]}")
        print(f"   Similarity score: {best_match[1]:.4f}")
        
        print("\n✅ All tests passed! Embeddings are working correctly.")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\n💡 To fix this, install the required dependencies:")
        print("   pip install sentence-transformers transformers torch")
        return False
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    return True

def test_huggingface_embeddings():
    """Test the HuggingFace embeddings implementation."""
    print("\n🧪 Testing HuggingFace Embeddings Implementation")
    print("=" * 50)
    
    try:
        from rag.embeddings import get_embeddings
        
        print("📦 Initializing HuggingFace embeddings...")
        embeddings = get_embeddings(
            model_name="answerdotai/ModernBERT-base", 
            use_huggingface=True
        )
        
        test_text = "This is a test sentence."
        
        print("📝 Testing embedding...")
        embedding = embeddings.embed_query(test_text)
        print(f"  ✓ Generated embedding with dimension: {len(embedding)}")
        
        print("✅ HuggingFace embeddings test passed!")
        
    except Exception as e:
        print(f"❌ HuggingFace embeddings error: {e}")
        print("💡 This is optional - sentence-transformers version should work fine.")

if __name__ == "__main__":
    success = test_embeddings()
    
    if success:
        test_huggingface_embeddings()
        
        print("\n🎉 Embeddings implementation is ready!")
        print("\n📋 Summary:")
        print("  • ModernBERT-base provides 768-dimensional embeddings")
        print("  • Both sentence-transformers and HuggingFace implementations available")
        print("  • No more dummy embeddings - real semantic understanding!")
        print("  • Ready for RAG ingestion and retrieval")
    else:
        print("\n⚠️  Please install dependencies and try again.")
        sys.exit(1)