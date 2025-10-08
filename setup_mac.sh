#!/bin/bash

# NVIDIA Docs RAG - Mac Client Setup Script
# Optimized for using migrated vector databases on Mac M1

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_message() {
    echo -e "${2}${1}${NC}"
}

print_header() {
    echo ""
    echo "=========================================="
    print_message "$1" $BLUE
    echo "=========================================="
}

print_header "💻 NVIDIA Docs RAG - Mac Client Setup"

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    print_message "⚠️  This script is optimized for macOS" $YELLOW
    print_message "Current OS: $OSTYPE" $YELLOW
fi

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

print_message "📂 Working directory: $SCRIPT_DIR" $BLUE

# Check system info
print_header "🔍 System Information"

# Mac system info
SYSTEM_VERSION=$(sw_vers -productVersion)
HARDWARE=$(uname -m)

print_message "macOS Version: $SYSTEM_VERSION" $GREEN
print_message "Hardware: $HARDWARE" $GREEN

# Check for Apple Silicon
if [[ "$HARDWARE" == "arm64" ]]; then
    print_message "✅ Apple Silicon (M1/M2/M3) detected" $GREEN
    APPLE_SILICON=true
else
    print_message "Intel Mac detected" $BLUE
    APPLE_SILICON=false
fi

# Memory info
MEMORY_GB=$(system_profiler SPHardwareDataType | grep "Memory:" | awk '{print $2 $3}')
print_message "Memory: $MEMORY_GB" $GREEN

# Check for conda
print_header "🐍 Python Environment Setup"

if command -v conda &> /dev/null; then
    print_message "Conda found" $GREEN
    
    # Check if lang environment exists and activate it
    if conda env list | grep -q "lang"; then
        print_message "Activating existing 'lang' environment" $BLUE
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate lang
    else
        print_message "Creating new conda environment: nvidia_rag_mac" $BLUE
        conda create -n nvidia_rag_mac python=3.10 -y
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate nvidia_rag_mac
    fi
    
    # Install PyTorch for Mac
    if [[ "$APPLE_SILICON" == true ]]; then
        print_message "Installing PyTorch with Apple Silicon optimization..." $BLUE
        conda install pytorch torchvision torchaudio -c pytorch -y
    else
        print_message "Installing PyTorch for Intel Mac..." $BLUE
        conda install pytorch torchvision torchaudio -c pytorch -y
    fi
    
else
    print_message "❌ Conda not found!" $RED
    print_message "Please install Miniconda or Anaconda first:" $YELLOW
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Install dependencies
print_header "📦 Installing Dependencies"
pip install -r requirements.txt

# Install additional Mac-specific optimizations
if [[ "$APPLE_SILICON" == true ]]; then
    print_message "Installing Apple Silicon optimizations..." $BLUE
    # Install optimized packages for Apple Silicon
    pip install --upgrade numpy scipy
fi

# Verify installation
print_header "✅ Installation Verification"
python3 -c "
import torch
import platform
print(f'Python version: {platform.python_version()}')
print(f'PyTorch version: {torch.__version__}')
print(f'MPS available: {torch.backends.mps.is_available()}')
print(f'MPS built: {torch.backends.mps.is_built()}')

try:
    import transformers
    print(f'Transformers version: {transformers.__version__}')
except ImportError as e:
    print(f'Transformers error: {e}')

try:
    import chromadb
    print(f'ChromaDB available')
except ImportError as e:
    print(f'ChromaDB error: {e}')

try:
    import sentence_transformers
    print(f'Sentence Transformers available')
except ImportError as e:
    print(f'Sentence Transformers error: {e}')
"

# Check for ModernBERT model
print_header "🤖 ModernBERT Model Check"
if [ -d "ModernBERT-base" ]; then
    print_message "✅ Local ModernBERT model found" $GREEN
    MODEL_SIZE=$(du -sh ModernBERT-base | cut -f1)
    print_message "   Path: $(pwd)/ModernBERT-base" $GREEN
    print_message "   Size: $MODEL_SIZE" $GREEN
else
    print_message "⚠️  Local ModernBERT model not found" $YELLOW
    print_message "   The system will use a fallback model" $YELLOW
    print_message "   To use ModernBERT, copy the model directory here" $BLUE
fi

# Test embeddings
print_header "🧪 Testing Embeddings"
python3 -c "
import sys
sys.path.append('$(pwd)')

try:
    from rag.embeddings import get_embeddings
    print('✅ Embeddings module loaded')
    
    embeddings = get_embeddings()
    print('✅ Embeddings initialized')
    
    test_embedding = embeddings.embed_query('Test query for NVIDIA GPU')
    print(f'✅ Test embedding generated: {len(test_embedding)} dimensions')
    
    print('🎉 Embeddings working correctly!')
    
except Exception as e:
    print(f'❌ Embeddings test failed: {e}')
    import traceback
    traceback.print_exc()
"

# Run system tests
print_header "🧪 Running System Tests"
if python3 test_rag_system.py; then
    print_message "✅ All tests passed!" $GREEN
    TESTS_PASSED=true
else
    print_message "⚠️  Some tests failed - but system may still work" $YELLOW
    TESTS_PASSED=false
fi

# Final setup
print_header "🎯 Mac Client Setup Complete"

if [[ "$TESTS_PASSED" == true ]]; then
    print_message "✅ System is ready for use!" $GREEN
else
    print_message "⚠️  System setup with warnings" $YELLOW
fi

echo ""
print_message "Usage instructions:" $BLUE
echo ""
echo "1. 📥 Extract vector database from Linux server:"
echo "   python3 migrate_vector_db.py extract \\"
echo "     --package /path/to/package.tar.gz \\"
echo "     --target ./local_vector_store"
echo ""
echo "2. ✅ Validate extracted database:"
echo "   python3 migrate_vector_db.py validate \\"
echo "     --vector-store ./local_vector_store/vector_store"
echo ""
echo "3. 🚀 Use the RAG system:"
echo "   python3 -m rag.cli chat"
echo "   # (assuming you have a CLI module)"
echo ""
echo "4. 🧪 Test similarity search:"
echo "   python3 -c \""
echo "   from rag.config import Config"
echo "   from langchain_community.vectorstores import Chroma"
echo "   from rag.embeddings import get_embeddings"
echo "   "
echo "   config = Config(docs_root=Path('./nvidia_docs_md'), vector_store_dir=Path('./local_vector_store'))"
echo "   embeddings = get_embeddings()"
echo "   vector_store = Chroma(persist_directory=str(config.vector_store_path()), embedding_function=embeddings)"
echo "   results = vector_store.similarity_search('NVIDIA GPU', k=3)"
echo "   for r in results: print(f'- {r.metadata.get(\\\"source\\\", \\\"unknown\\\")}')"
echo "   \""
echo ""

print_message "🎉 Mac client setup completed!" $GREEN

if [[ "$APPLE_SILICON" == true ]]; then
    print_message "💡 Apple Silicon optimizations enabled" $BLUE
fi