#!/usr/bin/env python3
"""
Main RAG application with LangGraph agent for NVIDIA documentation search.
"""
from __future__ import annotations

import logging
from typing import TypedDict, Annotated, Sequence

from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from operator import add
from langchain_ollama import ChatOllama

from .agent.tools import AGENT_TOOLS

logger = logging.getLogger(__name__)


def add_messages(left: Sequence[BaseMessage], right: Sequence[BaseMessage]) -> Sequence[BaseMessage]:
    """Add messages together for state updates."""
    return list(left) + list(right)


class AgentState(TypedDict):
    """State for the LangGraph RAG agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]


class RagAgent:
    """LangGraph-based RAG agent for NVIDIA documentation search."""
    
    def __init__(self, model_name: str = "llama3.2:3b"):
        """
        Initialize the RAG agent.
        
        Args:
            model_name: Ollama model to use for chat
        """
        self.model_name = model_name
        self.llm = ChatOllama(model=model_name)
        
        # Create tools dictionary for easy lookup
        self.tools_dict = {tool.name: tool for tool in AGENT_TOOLS}
        
        # Build the LangGraph
        self.graph = self._build_graph()
        self.rag_agent = self.graph.compile()
        
        logger.info(f"Initialized RAG agent with model: {model_name}")
        logger.info(f"Available tools: {list(self.tools_dict.keys())}")
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        graph = StateGraph(AgentState)
        
        # Add nodes
        graph.add_node("llm", self.call_llm)
        graph.add_node("retriever_agent", self.take_action)
        
        # Add conditional edges
        graph.add_conditional_edges(
            "llm",
            self.should_continue,
            {True: "retriever_agent", False: END}
        )
        
        # Add edge from retriever back to LLM
        graph.add_edge("retriever_agent", "llm")
        
        # Set entry point
        graph.set_entry_point("llm")
        
        return graph
    
    def should_continue(self, state: AgentState) -> bool:
        """Check if the last message contains tool calls."""
        last_message = state['messages'][-1]
        return hasattr(last_message, 'tool_calls') and len(last_message.tool_calls) > 0
    
    def call_llm(self, state: AgentState) -> AgentState:
        """Function to call the LLM with the current state."""
        messages = list(state['messages'])
        
        # Add system prompt if not already present
        if not messages or not isinstance(messages[0], SystemMessage):
            system_prompt = """
You are an intelligent AI assistant specialized in answering questions about NVIDIA documentation and technology.

You have access to a comprehensive knowledge base of NVIDIA technical documentation including:
- GPU architecture and specifications
- CUDA programming guides
- DGX system administration
- Networking hardware documentation
- AI acceleration technologies
- Driver and firmware information

Use the available retriever tools to search for relevant information when answering questions. 
You can make multiple tool calls if needed to gather comprehensive information.

Guidelines:
1. Always use the search tools to find relevant documentation before answering
2. Cite specific document sources in your answers
3. If you need clarification or additional context, use the tools to search for more information
4. Provide accurate, technical information based on the retrieved documents
5. If no relevant information is found, clearly state that and suggest alternative search terms

Available tools:
- search_nvidia_docs: Search the NVIDIA documentation database
- get_retrieval_stats: Get information about the knowledge base

Please provide helpful, accurate answers based on the NVIDIA documentation.
"""
            messages = [SystemMessage(content=system_prompt)] + messages
        
        # Bind tools to the LLM for function calling
        llm_with_tools = self.llm.bind_tools(AGENT_TOOLS)
        message = llm_with_tools.invoke(messages)
        
        return {'messages': [message]}
    
    def take_action(self, state: AgentState) -> AgentState:
        """Execute tool calls from the LLM's response."""
        last_message = state['messages'][-1]
        tool_calls = last_message.tool_calls
        results = []
        
        for tool_call in tool_calls:
            tool_name = tool_call['name']
            tool_args = tool_call.get('args', {})
            
            logger.debug(f"Calling tool: {tool_name} with args: {tool_args}")
            
            if tool_name not in self.tools_dict:
                logger.error(f"Tool '{tool_name}' not found")
                result = f"Error: Tool '{tool_name}' is not available. Available tools: {list(self.tools_dict.keys())}"
            else:
                try:
                    result = self.tools_dict[tool_name].invoke(tool_args)
                    logger.debug(f"Tool executed successfully: {len(str(result)) if result else 0} chars")
                except Exception as e:
                    logger.error(f"Tool execution failed: {e}")
                    result = f"Error executing tool '{tool_name}': {str(e)}"
            
            # Create tool message
            results.append(
                ToolMessage(
                    tool_call_id=tool_call['id'],
                    name=tool_name,
                    content=str(result)
                )
            )
        
        logger.debug("Tools execution complete")
        return {'messages': results}
    
    def chat(self, message: str) -> str:
        """
        Send a message to the RAG agent and get a response.
        
        Args:
            message: User's question or message
            
        Returns:
            Agent's response
        """
        messages = [HumanMessage(content=message)]
        result = self.rag_agent.invoke({"messages": messages})
        return result['messages'][-1].content
    
    def run_interactive(self):
        """Run the agent in interactive mode."""
        print("\n🤖 NVIDIA Documentation RAG Assistant")
        print("Ask questions about NVIDIA technology, GPUs, CUDA, DGX systems, and more!")
        print("Type 'exit', 'quit', or 'q' to stop.\n")
        
        while True:
            try:
                user_input = input("🧑 Your question: ").strip()
                
                if user_input.lower() in ['exit', 'quit', 'q', '']:
                    print("👋 Goodbye!")
                    break
                
                print("\n🔍 Searching NVIDIA docs...")
                response = self.chat(user_input)
                
                print(f"\n🤖 Answer:")
                print(response)
                print()
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                logger.error(f"Error in interactive mode: {e}")


class RagApp:
    """Main application class for the RAG system."""
    
    def __init__(self, model_name: str = "llama3.2:3b"):
        """
        Initialize the RAG application.
        
        Args:
            model_name: Ollama model to use
        """
        self.model_name = model_name
        self.agent = None
    
    def initialize_agent(self) -> RagAgent:
        """Initialize and return the RAG agent."""
        if not self.agent:
            self.agent = RagAgent(self.model_name)
        return self.agent
    
    def run(self):
        """Run the main application."""
        try:
            print("🚀 Initializing NVIDIA Documentation RAG System...")
            agent = self.initialize_agent()
            
            # Check system status
            from .agent.tools import get_retrieval_stats
            stats = get_retrieval_stats.invoke({})
            
            if stats.get('status') == 'ready':
                print(f"✅ System ready! {stats.get('document_count', 0)} documents available")
            else:
                print("⚠️  Database not fully ready, but system is functional")
            
            agent.run_interactive()
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
        except Exception as e:
            print(f"❌ Failed to initialize RAG system: {e}")
            logger.error(f"Application startup failed: {e}")


def main():
    """Main entry point for the application."""
    import sys
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Parse command line arguments
    model_name = "llama3.2:3b"
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    
    # Run the application
    app = RagApp(model_name)
    app.run()


if __name__ == "__main__":
    main()
