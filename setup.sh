#!/bin/bash
# NVIDIA Documentation RAG System Setup Script

set -e

echo "Setting up NVIDIA Documentation RAG System..."

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "Installing Ollama..."
    curl -fsSL https://ollama.ai/install.sh | sh
fi

# Install Python dependencies
echo "Installing dependencies..."
pip install -r requirements.txt
pip install -e .

# Start Ollama service (if not running)
if ! pgrep ollama > /dev/null; then
    ollama serve &
    sleep 3
fi

# Pull the default model
echo "Downloading model..."
ollama pull llama3.2:3b

echo "Setup complete! Use 'nvidia-rag chat' to start."