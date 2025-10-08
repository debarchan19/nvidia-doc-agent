#!/bin/bash

# NVIDIA Docs RAG - Linux Server Deployment Script
# Optimized for production ingestion on Linux servers

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

print_header "🚀 NVIDIA Docs RAG - Linux Server Deployment"

# Check if running on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    print_message "⚠️  This script is optimized for Linux servers" $YELLOW
    print_message "Current OS: $OSTYPE" $YELLOW
fi

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

print_message "📂 Working directory: $SCRIPT_DIR" $BLUE

# Check system resources
print_header "🔍 System Resource Check"

# CPU info
CPU_CORES=$(nproc)
print_message "CPU Cores: $CPU_CORES" $GREEN

# Memory info
MEMORY_GB=$(free -g | awk '/^Mem:/{print $2}')
print_message "Memory: ${MEMORY_GB}GB" $GREEN

# Disk space
DISK_FREE_GB=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
print_message "Disk Free: ${DISK_FREE_GB}GB" $GREEN

# Resource warnings
if [ "$MEMORY_GB" -lt 8 ]; then
    print_message "⚠️  Warning: Low memory detected. Consider using smaller batch sizes." $YELLOW
fi

if [ "$DISK_FREE_GB" -lt 50 ]; then
    print_message "⚠️  Warning: Low disk space. Ensure sufficient space for processing." $YELLOW
fi

# Check for GPU
print_header "🚀 GPU Detection"
if command -v nvidia-smi &> /dev/null; then
    print_message "NVIDIA GPU detected:" $GREEN
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | while IFS=, read name memory; do
        print_message "  - $name (${memory}MB)" $GREEN
    done
else
    print_message "No NVIDIA GPU detected - using CPU" $YELLOW
fi

# Python environment setup
print_header "🐍 Python Environment Setup"

# Check for conda
if command -v conda &> /dev/null; then
    print_message "Conda found - using conda environment" $GREEN
    
    # Create or activate environment
    ENV_NAME="nvidia_rag"
    if conda env list | grep -q "^$ENV_NAME "; then
        print_message "Activating existing environment: $ENV_NAME" $BLUE
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate $ENV_NAME
    else
        print_message "Creating new conda environment: $ENV_NAME" $BLUE
        conda create -n $ENV_NAME python=3.10 -y
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate $ENV_NAME
    fi
    
    # Install PyTorch with CUDA if available
    if command -v nvidia-smi &> /dev/null; then
        print_message "Installing PyTorch with CUDA support..." $BLUE
        conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
    else
        print_message "Installing PyTorch CPU version..." $BLUE
        conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
    fi
    
else
    print_message "Conda not found - using pip with virtual environment" $YELLOW
    
    # Create virtual environment
    if [ ! -d ".venv" ]; then
        print_message "Creating virtual environment..." $BLUE
        python3 -m venv .venv
    fi
    
    print_message "Activating virtual environment..." $BLUE
    source .venv/bin/activate
    
    # Install PyTorch
    if command -v nvidia-smi &> /dev/null; then
        print_message "Installing PyTorch with CUDA support..." $BLUE
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    else
        print_message "Installing PyTorch CPU version..." $BLUE
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
fi

# Install dependencies
print_header "📦 Installing Dependencies"
pip install -r requirements.txt

# Verify installation
print_header "✅ Installation Verification"
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA devices: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  Device {i}: {torch.cuda.get_device_name(i)}')

try:
    import transformers
    print(f'Transformers version: {transformers.__version__}')
except ImportError:
    print('Transformers not available')

try:
    import chromadb
    print(f'ChromaDB version: {chromadb.__version__}')
except ImportError:
    print('ChromaDB not available')
"

# Check for ModernBERT model
print_header "🤖 ModernBERT Model Check"
if [ -d "ModernBERT-base" ]; then
    print_message "✅ Local ModernBERT model found" $GREEN
    print_message "   Path: $(pwd)/ModernBERT-base" $GREEN
else
    print_message "⚠️  Local ModernBERT model not found" $YELLOW
    print_message "   The system will download from HuggingFace on first use" $YELLOW
    print_message "   To use a local model, place it in: $(pwd)/ModernBERT-base" $BLUE
fi

# Run tests
print_header "🧪 Running System Tests"
if python3 test_rag_system.py; then
    print_message "✅ All tests passed!" $GREEN
else
    print_message "❌ Some tests failed - check logs" $RED
    print_message "You may still proceed with caution" $YELLOW
fi

# Final setup
print_header "🎯 Deployment Complete"
print_message "System is ready for production ingestion!" $GREEN
echo ""
print_message "Next steps:" $BLUE
echo "  1. Prepare your documentation files in a directory"
echo "  2. Run production ingestion:"
echo "     python3 run_production_ingestion.py \\"
echo "       --docs-root /path/to/docs \\"
echo "       --vector-store /path/to/vector/store \\"
echo "       --modernbert-path $(pwd)/ModernBERT-base"
echo ""
echo "  3. Create migration package:"
echo "     python3 migrate_vector_db.py package \\"
echo "       --vector-store /path/to/vector/store \\"
echo "       --output /path/to/packages"
echo ""
echo "  4. Transfer package to Mac and extract:"
echo "     python3 migrate_vector_db.py extract \\"
echo "       --package /path/to/package.tar.gz \\"
echo "       --target /path/to/local/vector/store"
echo ""

print_message "🎉 Linux server deployment completed successfully!" $GREEN