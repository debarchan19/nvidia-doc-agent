# 🚀 NVIDIA Documentation RAG System

<div align="center">

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![LangGraph](https://img.shields.io/badge/LangGraph-0.6+-green.svg)
![Ollama](https://img.shields.io/badge/Ollama-Compatible-orange.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

**Intelligent AI Assistant for NVIDIA Technical Documentation**

*Ask questions about NVIDIA GPUs, CUDA, DGX systems, and more - get instant, accurate answers from official documentation*

[🎯 Quick Start](#-quick-start) • [🔧 Installation](#-installation) • [💡 Examples](#-examples) • [📚 Documentation](#-documentation)

</div>

---

## ✨ What is this?

The **NVIDIA Documentation RAG System** is an intelligent AI assistant that helps you quickly find information from NVIDIA's extensive technical documentation. Using advanced **Retrieval-Augmented Generation (RAG)** with **LangGraph** and **Ollama**, it provides accurate, source-cited answers to your NVIDIA technology questions.

### 🎯 Key Features

- 🔍 **Intelligent Search** - Query 4,800+ NVIDIA technical documents instantly
- 🤖 **AI-Powered Answers** - Get comprehensive responses using Ollama LLMs
- 📚 **Source Citations** - Every answer includes references to official docs
- 🔄 **Conversational** - Multi-turn conversations with context awareness
- ⚡ **Fast & Local** - Runs entirely on your machine, no API keys needed
- 🛠️ **Production Ready** - Built with LangGraph for robust workflows

### 💡 Perfect For

- **Developers** working with NVIDIA GPUs and CUDA
- **System Administrators** managing DGX systems
- **Researchers** exploring AI acceleration technologies
- **Engineers** implementing NVIDIA solutions
- **Anyone** needing quick access to NVIDIA documentation

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.ai/) installed and running
- 4GB+ RAM (for vector database)

### 1. Install Ollama Model
```bash
# Install the default model
ollama pull llama3.2:3b

# Or use a different model
ollama pull llama3.2:1b  # Smaller, faster
ollama pull llama3.1:8b  # Larger, more capable
```

### 2. Install the RAG System
```bash
# Clone the repository
git clone https://github.com/yourusername/nvidia-docs-rag.git
cd nvidia-docs-rag

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

### 3. Start Asking Questions!
```bash
# Interactive chat mode
nvidia-rag chat

# Single query
nvidia-rag query "What is CUDA programming?"

# Check system status
nvidia-rag status
```

---

## 💡 Examples

### Interactive Chat
```
🤖 NVIDIA Documentation RAG Assistant
Ask questions about NVIDIA technology, GPUs, CUDA, DGX systems, and more!

🧑 Your question: What are the key features of NVIDIA DGX systems?

🔍 Searching NVIDIA docs...

🤖 Answer:
NVIDIA DGX systems are purpose-built AI infrastructure platforms designed for enterprise 
AI workloads. Key features include:

• **High-Performance Computing**: Multiple NVIDIA GPUs (A100, H100) for parallel processing
• **Optimized Software Stack**: Pre-installed AI frameworks and NVIDIA software
• **Scalable Architecture**: Support for multi-node configurations and clustering
• **Enterprise Features**: Advanced networking, storage, and management capabilities

Sources: dgx-superpod-administration-guide-dgx-a100.pdf.md, enterprise-support-services-user-guide.pdf.md
```

### Programmatic Usage
```python
from rag.main import RagAgent

# Initialize the agent
agent = RagAgent(model_name="llama3.2:3b")

# Ask a question
response = agent.chat("How do I optimize GPU memory usage in CUDA?")
print(response)

# Multiple questions with context
agent.chat("What is CUDA?")
agent.chat("Show me examples of CUDA programming")  # Maintains context
```

---

## 🔧 Installation

### Option 1: Quick Install
```bash
pip install nvidia-docs-rag
```

### Option 2: From Source
```bash
# Clone repository
git clone https://github.com/yourusername/nvidia-docs-rag.git
cd nvidia-docs-rag

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Option 3: Using Docker
```bash
# Build the image
docker build -t nvidia-rag .

# Run the container
docker run -it --rm nvidia-rag chat
```

---

## 📖 Usage

### Command Line Interface

```bash
# Start interactive chat
nvidia-rag chat

# Process single query
nvidia-rag query "What is NVIDIA Tensor Core?"

# Check system status
nvidia-rag status

# Run system test
nvidia-rag test

# Use different model
nvidia-rag --model llama3.1:8b chat
```

### Python API

```python
from rag.main import RagAgent
from rag.agent.controller import AgentController

# Method 1: Using RagAgent (LangGraph-based)
agent = RagAgent()
response = agent.chat("What are NVIDIA GPUs used for?")

# Method 2: Using AgentController (Direct control)
controller = AgentController()
result = controller.query("Explain CUDA architecture")
print(result['response'])

# Method 3: Direct tool usage
from rag.agent.tools import search_nvidia_docs

results = search_nvidia_docs.invoke({
    "query": "GPU memory optimization",
    "max_results": 5
})
```

---

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Query    │───▶│   LangGraph      │───▶│   Ollama LLM    │
└─────────────────┘    │   Workflow       │    └─────────────────┘
                       └──────────────────┘             │
                                │                       │
                                ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Final Answer   │◀───│  RAG Pipeline    │◀───│  Retrieved Docs │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                       ▲
                                ▼                       │
                       ┌──────────────────┐    ┌─────────────────┐
                       │   Search Tools   │───▶│  ChromaDB       │
                       └──────────────────┘    │  Vector Store   │
                                               └─────────────────┘
```

### Components

- **LangGraph Workflow**: Orchestrates the RAG pipeline with conditional edges
- **Ollama Integration**: Local LLM inference without API dependencies  
- **ChromaDB Vector Store**: Efficient similarity search over 4,800+ documents
- **Smart Retrieval**: Context-aware document retrieval with relevance scoring
- **Citation System**: Automatic source attribution for all answers



## 🛠️ Configuration

### Environment Variables
```bash
# Optional: Set custom paths
export NVIDIA_DOCS_ROOT="/path/to/nvidia/docs"
export NVIDIA_VECTOR_STORE_DIR="/path/to/vector/store"
```

### Custom Models
```python
# Use different Ollama models
agent = RagAgent(model_name="llama3.1:8b")  # More capable
agent = RagAgent(model_name="llama3.2:1b")  # Faster/smaller
```

### Advanced Configuration
```python
from rag.pipeline.retrieve import Retriever

# Custom retriever settings
retriever = Retriever(
    top_k=10,           # More results
    max_distance=1.5    # Stricter relevance
)
```

---

## 📚 Documentation Topics Covered

The system includes comprehensive documentation on:

- **GPU Architecture**: NVIDIA GPU designs, specifications, and capabilities
- **CUDA Programming**: Development guides, APIs, and best practices  
- **DGX Systems**: Administration, deployment, and management
- **Networking**: InfiniBand, Ethernet, and interconnect technologies
- **AI Frameworks**: TensorRT, cuDNN, and acceleration libraries
- **Driver & Firmware**: Installation, configuration, and updates
- **Enterprise Solutions**: Deployment guides and enterprise features


## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature-name`
3. **Make** your changes and add tests
4. **Run** tests: `pytest`
5. **Submit** a pull request

### Development Setup
```bash
# Clone and setup development environment
git clone https://github.com/yourusername/nvidia-docs-rag.git
cd nvidia-docs-rag

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black rag/
```


## 🙋‍♂️ Support

- 📖 **Documentation**: Check this README and code comments
- 🐛 **Issues**: [GitHub Issues](https://github.com/yourusername/nvidia-docs-rag/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/yourusername/nvidia-docs-rag/discussions)
- 📧 **Contact**: your.email@example.com



<div align="center">



</div>
