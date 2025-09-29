#!/bin/bash
# NVIDIA Documentation RAG System Setup Script

set -e  # Exit on any error

echo "🚀 Setting up NVIDIA Documentation RAG System..."

# Check Python version
python_version=$(python3 --version 2>&1 | grep -o '[0-9]\+\.[0-9]\+')
required_version="3.10"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python $required_version or higher is required. Found: $python_version"
    exit 1
fi

echo "✅ Python version check passed"

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "📦 Installing Ollama..."
    curl -fsSL https://ollama.ai/install.sh | sh
else
    echo "✅ Ollama is already installed"
fi

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Install the package in development mode
echo "📦 Installing NVIDIA RAG system..."
pip install -e .

# Start Ollama service (if not running)
if ! pgrep ollama > /dev/null; then
    echo "🚀 Starting Ollama service..."
    ollama serve &
    sleep 3
fi

# Pull the default model
echo "📥 Downloading default language model (llama3.2:3b)..."
ollama pull llama3.2:3b

# Run system test
echo "🧪 Running system test..."
nvidia-rag test

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "📖 Quick usage:"
echo "  nvidia-rag chat          # Interactive mode"
echo "  nvidia-rag status        # Check system status"
echo "  nvidia-rag test          # Run system test"
echo ""
echo "📚 For more information, see README.md"