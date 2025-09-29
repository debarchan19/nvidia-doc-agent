#!/usr/bin/env python3
"""
Command-line interface for the NVIDIA Documentation RAG System.
"""
import argparse
import sys
from pathlib import Path
from typing import Optional

from .main import RagApp, RagAgent
from .agent.tools import get_retrieval_stats


def cmd_chat(args):
    """Run the interactive chat interface."""
    app = RagApp(model_name=args.model)
    app.run()


def cmd_query(args):
    """Process a single query and return the result."""
    agent = RagAgent(model_name=args.model)
    
    if args.query:
        response = agent.chat(args.query)
        print(response)
    else:
        print("Error: --query is required for single query mode")
        sys.exit(1)


def cmd_status(args):
    """Show system status and statistics."""
    try:
        stats = get_retrieval_stats.invoke({})
        print(f"Status: {stats.get('status', 'unknown')}")
        print(f"Documents: {stats.get('document_count', 0)}")
        print(f"Health: {'OK' if stats.get('health_check') else 'Issues'}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def cmd_test(args):
    """Run a quick system test."""
    try:
        from .agent.tools import search_nvidia_docs
        
        # Test search
        results = search_nvidia_docs.invoke({"query": "NVIDIA GPU", "max_results": 2})
        search_ok = results and results[0].get('source') != 'no_results'
        
        # Test agent
        agent = RagAgent(model_name=args.model)
        response = agent.chat("What is NVIDIA?")
        agent_ok = response and len(response) > 50
        
        print(f"Search: {'OK' if search_ok else 'FAIL'}")
        print(f"Agent: {'OK' if agent_ok else 'FAIL'}")
        print("Test completed!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="nvidia-rag",
        description="NVIDIA Documentation RAG System"
    )
    
    parser.add_argument(
        "--model", 
        default="llama3.2:3b",
        help="Ollama model to use (default: llama3.2:3b)"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Start interactive chat interface")
    chat_parser.set_defaults(func=cmd_chat)
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Process a single query")
    query_parser.add_argument("query", help="Query to process")
    query_parser.set_defaults(func=cmd_query)
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Show system status")
    status_parser.set_defaults(func=cmd_status)
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Run system test")
    test_parser.set_defaults(func=cmd_test)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()