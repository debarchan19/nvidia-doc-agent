#!/bin/bash

# NVIDIA Documentation Ingestion Script
# This script activates the virtual environment and runs the ingestion pipeline

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored messages
print_message() {
    echo -e "${2}${1}${NC}"
}

print_message "🚀 Starting NVIDIA Documentation Ingestion" $BLUE
echo "=========================================="

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

print_message "📂 Working directory: $SCRIPT_DIR" $BLUE

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    print_message "❌ Virtual environment not found at .venv" $RED
    print_message "Please create a virtual environment first:" $YELLOW
    echo "  python3 -m venv .venv"
    echo "  source .venv/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
print_message "🐍 Activating virtual environment..." $BLUE
source .venv/bin/activate

# Verify Python environment
if [ -z "$VIRTUAL_ENV" ]; then
    print_message "❌ Failed to activate virtual environment" $RED
    exit 1
fi

print_message "✓ Virtual environment activated: $VIRTUAL_ENV" $GREEN

# Check if required packages are installed
print_message "📦 Checking required packages..." $BLUE
python3 -c "
try:
    import langchain
    import chromadb
    import numpy as np
    print('✓ Required packages are available')
except ImportError as e:
    print(f'❌ Missing required package: {e}')
    exit(1)
"

if [ $? -ne 0 ]; then
    print_message "Installing required packages..." $YELLOW
    pip install -r requirements.txt
fi

# Check if nvidia_docs_md directory exists
if [ ! -d "nvidia_docs_md" ]; then
    print_message "⚠️  nvidia_docs_md directory not found" $YELLOW
    print_message "The ingestion will still run but may not find documents to process" $YELLOW
fi

# Set environment variables if needed
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# Run the ingestion
print_message "🔄 Starting ingestion pipeline..." $BLUE
echo "This may take several minutes depending on the number of documents..."
echo ""

# Run the Python ingestion script
python3 run_ingestion.py

# Check if ingestion was successful
if [ $? -eq 0 ]; then
    print_message "✅ Ingestion completed successfully!" $GREEN
    echo ""
    print_message "📊 Next steps:" $BLUE
    echo "  • Vector store created in: rag/vector_store/"
    echo "  • You can now run the RAG system with: python3 -m rag.cli chat"
    echo "  • Check ingestion.log for detailed logs"
else
    print_message "❌ Ingestion failed!" $RED
    print_message "Check ingestion.log for error details" $YELLOW
    exit 1
fi

print_message "🎉 All done!" $GREEN