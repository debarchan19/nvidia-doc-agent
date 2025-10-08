# 🚀 NVIDIA Docs RAG - Cross-Platform Deployment Guide

## 📋 Overview

This guide covers the complete workflow for deploying the NVIDIA Documentation RAG system across Linux servers and Mac clients with seamless vector database migration.

## 🎯 Workflow Summary

1. **Linux Server**: Run production ingestion with full dataset
2. **Migration**: Package and transfer vector database
3. **Mac Client**: Extract and use vector database locally
4. **Testing**: Comprehensive validation at each step

## 🖥️ Linux Server Deployment

### Prerequisites
- Linux server with sufficient resources (8GB+ RAM recommended)
- Python 3.10+
- NVIDIA GPU (optional, but recommended for large datasets)

### Setup Steps

1. **Clone and Navigate**
   ```bash
   git clone <your-repo-url>
   cd nvidia-doc-agent
   ```

2. **Run Deployment Script**
   ```bash
   ./deploy_linux.sh
   ```
   This script will:
   - Check system resources
   - Detect NVIDIA GPUs
   - Setup conda/virtual environment
   - Install dependencies with CUDA support
   - Run system tests

3. **Prepare Documentation**
   ```bash
   # Create documentation directory
   mkdir nvidia_docs_md
   # Copy your NVIDIA documentation files here
   cp -r /path/to/nvidia/docs/* nvidia_docs_md/
   ```

4. **Run Production Ingestion**
   ```bash
   python3 run_production_ingestion.py \
     --docs-root ./nvidia_docs_md \
     --vector-store ./production_vector_store \
     --modernbert-path ./ModernBERT-base \
     --log-level INFO \
     --log-file ingestion.log
   ```

5. **Create Migration Package**
   ```bash
   python3 migrate_vector_db.py package \
     --vector-store ./production_vector_store/vector_store \
     --output ./packages
   ```

## 💻 Mac Client Setup

### Prerequisites
- macOS (Intel or Apple Silicon)
- Miniconda/Anaconda installed
- 8GB+ RAM

### Setup Steps

1. **Clone Repository** (if not already done)
   ```bash
   git clone <your-repo-url>
   cd nvidia-doc-agent
   ```

2. **Run Mac Setup Script**
   ```bash
   ./setup_mac.sh
   ```
   This script will:
   - Detect Apple Silicon vs Intel
   - Setup conda environment
   - Install optimized dependencies
   - Run validation tests

3. **Transfer Migration Package**
   ```bash
   # Copy package from Linux server
   scp user@linux-server:/path/to/packages/nvidia_vector_db_*.tar.gz ./
   ```

4. **Extract Vector Database**
   ```bash
   python3 migrate_vector_db.py extract \
     --package ./nvidia_vector_db_*.tar.gz \
     --target ./local_vector_store \
     --overwrite
   ```

5. **Validate Installation**
   ```bash
   python3 migrate_vector_db.py validate \
     --vector-store ./local_vector_store/vector_store
   ```

## 🔧 Configuration Options

### Environment Variables
```bash
# Optional: Override default paths
export NVIDIA_DOCS_ROOT=/path/to/docs
export NVIDIA_VECTOR_STORE_DIR=/path/to/vector/store  
export NVIDIA_MODERNBERT_PATH=/path/to/modernbert/model
```

### Production Ingestion Options
```bash
python3 run_production_ingestion.py \
  --docs-root /path/to/docs \              # Required: Documentation directory
  --vector-store /path/to/vector/store \   # Required: Vector store output
  --modernbert-path /path/to/model \       # Optional: Local ModernBERT model
  --batch-size 32 \                       # Optional: Processing batch size
  --max-workers 4 \                       # Optional: Parallel workers
  --log-level INFO \                      # Optional: Logging level
  --log-file ingestion.log                # Optional: Log file path
```

## 🧪 Testing and Validation

### Run Complete Test Suite
```bash
python3 test_rag_system.py
```

### Test Individual Components

1. **Test Embeddings**
   ```python
   from rag.embeddings import get_embeddings
   embeddings = get_embeddings()
   result = embeddings.embed_query("Test query")
   print(f"Embedding dimension: {len(result)}")
   ```

2. **Test Vector Database**
   ```python
   from rag.config import Config
   from langchain_community.vectorstores import Chroma
   from rag.embeddings import get_embeddings
   
   config = Config(docs_root=Path("./nvidia_docs_md"))
   embeddings = get_embeddings()
   vector_store = Chroma(
       persist_directory=str(config.vector_store_path()),
       embedding_function=embeddings
   )
   results = vector_store.similarity_search("NVIDIA GPU", k=3)
   ```

3. **Performance Benchmark**
   ```bash
   python3 -c "
   from test_rag_system import run_performance_benchmark
   run_performance_benchmark()
   "
   ```

## 📊 Performance Optimization

### Linux Server Optimizations
- **GPU Acceleration**: Use NVIDIA GPUs for faster embedding generation
- **Batch Processing**: Increase batch size for better throughput
- **Parallel Processing**: Use multiple CPU cores for document processing
- **Memory Management**: Monitor RAM usage during large ingestion

### Mac Client Optimizations
- **Apple Silicon**: Automatic MPS acceleration on M1/M2/M3 Macs
- **Memory Mapping**: Efficient vector database loading
- **Local Caching**: ModernBERT model caching for faster startup

## 🚨 Troubleshooting

### Common Issues

1. **ModernBERT Model Not Found**
   ```bash
   # Download model manually
   git lfs clone https://huggingface.co/answerdotai/ModernBERT-base
   ```

2. **CUDA Issues on Linux**
   ```bash
   # Check CUDA installation
   nvidia-smi
   python3 -c "import torch; print(torch.cuda.is_available())"
   ```

3. **Memory Issues**
   ```bash
   # Reduce batch size
   python3 run_production_ingestion.py --batch-size 16 ...
   ```

4. **Vector Database Corruption**
   ```bash
   # Validate and recreate if needed
   python3 migrate_vector_db.py validate --vector-store /path/to/store
   ```

### Log Files
- **Linux**: Check `ingestion.log` for detailed processing logs
- **Mac**: Check `migration.log` for transfer/extraction logs
- **Tests**: Test output includes performance metrics

## 📁 Directory Structure

```
nvidia-doc-agent/
├── ModernBERT-base/              # Local ModernBERT model
├── nvidia_docs_md/               # Source documentation
├── production_vector_store/      # Linux: Generated vector DB
├── local_vector_store/          # Mac: Extracted vector DB
├── packages/                    # Migration packages
├── rag/                        # Core RAG system
│   ├── embeddings.py           # Embedding implementations
│   ├── config.py              # Configuration
│   └── pipeline/              # Ingestion pipeline
├── deploy_linux.sh            # Linux deployment script
├── setup_mac.sh              # Mac setup script
├── run_production_ingestion.py # Production ingestion
├── migrate_vector_db.py       # Migration utilities
└── test_rag_system.py         # Test suite
```

## 🎯 Production Checklist

### Before Linux Deployment
- [ ] System resources checked (CPU, RAM, GPU)
- [ ] Documentation files prepared
- [ ] ModernBERT model available (local or HuggingFace)
- [ ] Sufficient disk space (estimate 2-5GB per 10k documents)

### During Ingestion
- [ ] Monitor system resources
- [ ] Check ingestion logs for errors
- [ ] Validate processing rate (docs/second)
- [ ] Test vector database after completion

### Before Migration
- [ ] Vector database validated
- [ ] Migration package created successfully
- [ ] Package integrity verified
- [ ] Transfer method prepared (scp, rsync, etc.)

### After Mac Setup
- [ ] Dependencies installed correctly
- [ ] Vector database extracted successfully
- [ ] Similarity search working
- [ ] Performance acceptable for local use

## 🚀 Next Steps

1. **Scale Up**: Use this setup for full NVIDIA documentation corpus
2. **API Integration**: Build REST API for RAG queries
3. **User Interface**: Create web interface for document search
4. **Monitoring**: Add metrics and monitoring for production use
5. **CI/CD**: Automate testing and deployment pipeline

## 📚 Additional Resources

- [ModernBERT Paper](https://arxiv.org/abs/2412.13663)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [LangChain RAG Guide](https://python.langchain.com/docs/use_cases/question_answering/)
- [NVIDIA Developer Documentation](https://developer.nvidia.com/)

---

🎉 **Your NVIDIA Documentation RAG system is now ready for production deployment!**