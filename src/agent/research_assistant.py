"""Research Assistant LangGraph implementation.

A multi-node workflow that processes research questions, searches for information,
analyzes findings, and generates structured reports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, TypedDict
import asyncio
import requests
import os
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from tavily import TavilyClient

os.environ["LANGSMITH_PROJECT"] = "agentForge"  
os.environ["LANGSMITH_TRACING"] = "true"

class ResearchStatus(str, Enum):
    """Status of the research process."""

    AWAITING_QUESTION = "awaiting_question"
    SEARCHING = "searching"
    ANALYZING = "analyzing"
    REPORTING = "reporting"
    COMPLETE = "complete"
    ERROR = "error"


class Configuration(TypedDict):
    """Configurable parameters for the research assistant.

    Set these when creating assistants OR when invoking the graph.
    """

    model: str
    temperature: float
    tavily_api_key: str
    max_search_results: int
    search_depth: str  # 'basic' or 'advanced'


@dataclass
class ResearchState:
    """State for the research assistant workflow."""

    # Input
    question: Optional[str] = None
    
    # Processing state
    status: ResearchStatus = ResearchStatus.AWAITING_QUESTION
    error: Optional[str] = None
    
    # Intermediate data
    search_results: List[Dict[str, Any]] = field(default_factory=list)
    analysis: Optional[str] = None
    
    # Output
    report: Optional[str] = None
    
    # Messages for conversation history
    messages: List[Dict[str, Any]] = field(default_factory=list)


def get_llm(config: RunnableConfig):
    """Get the LLM based on configuration."""

    configuration = config["configurable"]
    API_KEY = os.environ.get('NEBIUS_API_KEY')
    BASE_URL = "https://api.studio.nebius.ai/v1/"

    # USE WHICHEVER LLM SUITS YOUR NEEDS
    
    openai = ChatOpenAI(
        model="gpt-4o-mini",  # $0.15/$0.60 per M tokens
        api_key=os.environ.get("OPENAI_API_KEY"),
        temperature=0.7
    )

    # DeepSeek R1 - Best reasoning model (OpenAI o1 competitor)
    deepseek_reasoning = ChatOpenAI(
        base_url=BASE_URL,
        api_key=API_KEY,
        model="deepseek-ai/DeepSeek-R1",  # $0.80/$2.40 per M tokens
        temperature=0.1  # Lower for more consistent reasoning
    )

    # Different models, same base URL
    llama_70b = ChatOpenAI(
        base_url=BASE_URL,
        api_key=API_KEY,
        model="meta-llama/Meta-Llama-3.1-70B-Instruct"  # 70B model
    )

    llama_8b_fast = ChatOpenAI(
        base_url=BASE_URL,
        api_key=API_KEY,
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-fast"  # 8B fast variant
    )

    qwen_model = ChatOpenAI(
        base_url=BASE_URL,
        api_key=API_KEY,
        model="Qwen/Qwen2.5-Coder-32B-Instruct"  # Qwen coding model
    )

    return llama_70b


async def process_question(state: ResearchState, config: RunnableConfig) -> Dict[str, Any]:
    print(state.messages)
    state.question = state.messages['content']['text']
    """Process the research question and prepare for search."""
    if not state.question:
        state.error = "No research question provided."
        state.status = ResearchStatus.ERROR
        state.messages.append({"role": "assistant", "content": f"Error: {state.error}"})
        return state
    
    llm = get_llm(config)
    
    # Create a prompt to refine the search query
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a research assistant. Your task is to convert the user's research question into effective search queries."),
        ("user", "Create 3 effective search queries for the following research question. Return only the queries separated by newlines, no explanations:\n\n{question}")
    ])
    
    # Generate search queries
    chain = prompt | llm | StrOutputParser()
    search_queries = await chain.ainvoke({"question": state.question})
    
    # Add to messages
    state.messages.append({"role": "user", "content": state.question})
    state.messages.append({"role": "assistant", "content": f"I'll research: {state.question}"})
    
    # Update state
    state.status = ResearchStatus.SEARCHING
    state.search_queries = search_queries.strip().split("\n")
    
    return state


async def search_web(state: ResearchState, config: RunnableConfig) -> ResearchState:
    """Search the web for information related to the research question using Tavily API."""
    # Skip if we're in error state
    if state.status == ResearchStatus.ERROR:
        return state
        
    configuration = config["configurable"]
    tavily_api_key = configuration.get("tavily_api_key", os.getenv("TAVILY_API_KEY"))
    max_results = configuration.get("max_search_results", 5)
    search_depth = configuration.get("search_depth", "basic")  # 'basic' or 'advanced'
    
    if not tavily_api_key:
        state.error = "No Tavily API key provided. Please provide a Tavily API key in the configuration or set the TAVILY_API_KEY environment variable."
        state.status = ResearchStatus.ERROR
        state.messages.append({"role": "assistant", "content": f"Error: {state.error}"})
        return state
    
    # Initialize Tavily client
    try:
        client = TavilyClient(api_key=tavily_api_key)
    except Exception as e:
        state.error = f"Failed to initialize Tavily client: {str(e)}"
        state.status = ResearchStatus.ERROR
        state.messages.append({"role": "assistant", "content": f"Error: {state.error}"})
        return state
    
    search_queries = state.search_queries if hasattr(state, "search_queries") else [state.question]
    all_results = []
    
    # Define a function to run the blocking Tavily search in a separate thread
    def run_tavily_search(query: str) -> Dict[str, Any]:
        return client.search(
            query=query,
            search_depth=search_depth,
            max_results=max_results,
            include_domains=[],  # Optional: specific domains to include
            exclude_domains=[],  # Optional: specific domains to exclude
            include_answer=True,  # Get an AI-generated answer along with results
            include_raw_content=False,  # Don't include full page content to save tokens
        )
    
    try:
        # Process each query and collect results
        for query in search_queries:
            # Run the blocking Tavily search in a separate thread
            response = await asyncio.to_thread(run_tavily_search, query)
            
            # Extract results
            if "results" in response:
                for result in response["results"]:
                    all_results.append({
                        "title": result.get("title", "No title"),
                        "snippet": result.get("content", "No content available"),
                        "url": result.get("url", "#"),
                        "score": result.get("score", 0)
                    })
            
            # If there's an answer, add it as a special result
            if "answer" in response and response["answer"]:
                all_results.append({
                    "title": "Tavily AI Answer",
                    "snippet": response["answer"],
                    "url": "#tavily_answer",
                    "score": 1.0  # Give it highest relevance
                })
    except Exception as e:
        state.error = f"Error during Tavily search: {str(e)}"
        state.status = ResearchStatus.ERROR
        state.messages.append({"role": "assistant", "content": f"Error: {state.error}"})
        return state
    
    # Sort results by relevance score (if available)
    all_results = sorted(all_results, key=lambda x: x.get("score", 0), reverse=True)
    
    # Add to messages
    state.messages.append({"role": "assistant", "content": f"I found {len(all_results)} relevant results using Tavily Search API."})
    
    # Update state
    state.status = ResearchStatus.ANALYZING
    state.search_results = all_results
    
    return state


async def analyze_results(state: ResearchState, config: RunnableConfig) -> ResearchState:
    """Analyze the search results and extract relevant information."""
    # Skip if we're in error state
    if state.status == ResearchStatus.ERROR:
        return state
        
    if not state.search_results:
        state.error = "No search results to analyze."
        state.status = ResearchStatus.ERROR
        state.messages.append({"role": "assistant", "content": f"Error: {state.error}"})
        return state
    
    llm = get_llm(config)
    
    # Format search results for the prompt
    formatted_results = "\n\n".join([
        f"Title: {result['title']}\nSnippet: {result['snippet']}\nURL: {result['url']}"
        for result in state.search_results
    ])
    
    # Create a prompt to analyze the results
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a research analyst. Analyze the search results and extract key information relevant to the research question."),
        ("user", "Research Question: {question}\n\nSearch Results:\n{results}\n\nProvide a comprehensive analysis of these results, highlighting key findings, patterns, and insights.")
    ])
    
    # Generate analysis
    chain = prompt | llm | StrOutputParser()
    analysis = await chain.ainvoke({
        "question": state.question,
        "results": formatted_results
    })
    
    # Add to messages
    state.messages.append({"role": "assistant", "content": "I've analyzed the search results and extracted key information."})
    
    # Update state
    state.status = ResearchStatus.REPORTING
    state.analysis = analysis
    
    return state


async def generate_report(state: ResearchState, config: RunnableConfig) -> ResearchState:
    """Generate a structured report based on the analysis."""
    # Skip if we're in error state
    if state.status == ResearchStatus.ERROR:
        return state
        
    if not state.analysis:
        state.error = "No analysis to generate report from."
        state.status = ResearchStatus.ERROR
        state.messages.append({"role": "assistant", "content": f"Error: {state.error}"})
        return state
    
    llm = get_llm(config)
    
    # Create a prompt to generate the report
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a research report writer. Create a well-structured, comprehensive report based on the analysis provided."),
        ("user", "Research Question: {question}\n\nAnalysis: {analysis}\n\nGenerate a structured research report with the following sections: Executive Summary, Key Findings, Detailed Analysis, Conclusions, and Recommendations.")
    ])
    
    # Generate report
    chain = prompt | llm | StrOutputParser()
    report = await chain.ainvoke({
        "question": state.question,
        "analysis": state.analysis
    })
    
    # Add to messages
    state.messages.append({"role": "assistant", "content": "I've prepared a comprehensive research report for you."})
    
    # Update state
    state.status = ResearchStatus.COMPLETE
    state.report = report

    print(state)
    print(type(state))
    print(state.search_results)
    
    return state


# Define the research assistant graph with a linear workflow
research_graph = (
    StateGraph(ResearchState, config_schema=Configuration)
    .add_node("process_question", process_question)
    .add_node("search_web", search_web)
    .add_node("analyze_results", analyze_results)
    .add_node("generate_report", generate_report)
    .add_edge("__start__", "process_question")
    .add_edge("process_question", "search_web")
    .add_edge("search_web", "analyze_results")
    .add_edge("analyze_results", "generate_report")
    .add_edge("generate_report", "__end__")
    .compile(name="Research Assistant")
)
