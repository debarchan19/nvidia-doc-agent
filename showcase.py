#!/usr/bin/env python3
"""
NVIDIA Docs RAG System - Repository Showcase
Demonstrates the key features and capabilities of the system.
"""

import sys
from pathlib import Path

def print_header(title):
    print(f"\n{'='*60}")
    print(f"🚀 {title}")
    print(f"{'='*60}")

def main():
    print_header("NVIDIA Documentation RAG System")
    
    print("""
🎯 This repository contains a production-ready, cross-platform RAG system
   specifically designed for NVIDIA technical documentation.

✨ Key Features:
   • ModernBERT embeddings (768-dimensional semantic understanding)
   • Cross-platform deployment (Linux servers + Mac clients)
   • Seamless vector database migration
   • Production-optimized ingestion pipeline
   • Comprehensive test suite and validation
   • CUDA acceleration on Linux, Apple Silicon on Mac
    """)
    
    print_header("Repository Structure")
    
    project_root = Path(__file__).parent
    important_files = [
        ("🚀 deploy_linux.sh", "Linux server deployment script"),
        ("🍎 setup_mac.sh", "Mac client setup with Apple Silicon optimization"),
        ("⚡ run_production_ingestion.py", "Production-scale document processing"),
        ("📦 migrate_vector_db.py", "Vector database migration utilities"),
        ("🧪 test_rag_system.py", "Comprehensive test suite"),
        ("🧠 rag/embeddings.py", "ModernBERT implementation"),
        ("⚙️ rag/config.py", "Cross-platform configuration"),
        ("📚 DEPLOYMENT_GUIDE.md", "Step-by-step deployment guide"),
        ("📋 README.md", "Project documentation"),
    ]
    
    for file_desc, description in important_files:
        file_path = file_desc.split()[1]
        if (project_root / file_path).exists():
            print(f"   ✅ {file_desc:<30} - {description}")
        else:
            print(f"   ❌ {file_desc:<30} - {description}")
    
    print_header("Quick Start Commands")
    
    print("""
📋 Linux Server Deployment:
   ./deploy_linux.sh
   python3 run_production_ingestion.py --docs-root ./docs --vector-store ./vector_store

📋 Mac Client Setup:
   ./setup_mac.sh
   python3 migrate_vector_db.py extract --package vector_db.tar.gz --target ./local_store

📋 Testing:
   python3 test_rag_system.py
    """)
    
    print_header("System Validation")
    
    try:
        # Test embeddings import
        sys.path.append(str(project_root))
        from rag.embeddings import get_embeddings
        print("   ✅ Embeddings module - Ready")
        
        # Test configuration
        from rag.config import Config
        print("   ✅ Configuration system - Ready")
        
        # Test ModernBERT local model detection
        embeddings = get_embeddings()
        test_embedding = embeddings.embed_query("Test NVIDIA GPU query")
        print(f"   ✅ ModernBERT embeddings - {len(test_embedding)} dimensions")
        
        print("\n🎉 System validation successful!")
        print("   Ready for production deployment!")
        
    except Exception as e:
        print(f"\n⚠️  System validation warning: {e}")
        print("   Run setup scripts to resolve dependencies")
    
    print_header("Next Steps")
    
    print("""
1. 🐧 Linux Server: Run ./deploy_linux.sh for production setup
2. 📚 Documentation: Read DEPLOYMENT_GUIDE.md for detailed instructions  
3. 🧪 Testing: Execute python3 test_rag_system.py to validate setup
4. 🚀 Production: Use run_production_ingestion.py for large-scale processing
5. 🍎 Mac Client: Run ./setup_mac.sh for local development environment

🔗 Repository: https://github.com/debarchan19/nvidia-doc-agent
📖 Issues: Report bugs and feature requests on GitHub
🤝 Contribute: Fork, branch, and submit pull requests
    """)
    
    print(f"\n{'='*60}")
    print("🎉 Welcome to the NVIDIA Documentation RAG System!")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()