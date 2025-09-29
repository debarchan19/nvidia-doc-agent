FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.ai/install.sh | sh

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application
COPY . .

# Install the package
RUN pip install -e .

# Create a volume for vector store persistence
VOLUME ["/app/rag/vector_store"]

# Expose Ollama port
EXPOSE 11434

# Create entrypoint script
RUN echo '#!/bin/bash\n\
    # Start Ollama in background\n\
    ollama serve &\n\
    \n\
    # Wait for Ollama to start\n\
    sleep 5\n\
    \n\
    # Pull the default model if not exists\n\
    ollama pull llama3.2:3b || echo "Model already exists or failed to download"\n\
    \n\
    # Run the RAG system\n\
    exec "$@"' > /entrypoint.sh && chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["nvidia-rag", "chat"]