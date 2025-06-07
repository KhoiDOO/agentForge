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
from typing import Literal
from langgraph.types import interrupt, Command

os.environ["LANGSMITH_PROJECT"] = "bio"  
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

async def alert(state: ResearchState, config: RunnableConfig):
    print("ALERT: Human did not approve. Taking appropriate action.")
    # You can add more alert logic here
    return state

async def another_node(state: ResearchState, config: RunnableConfig):
    print("Another node was chosen. Continuing the workflow.")
    # You can add more logic for this node here
    return state

async def init_question(state: ResearchState) -> Command[Literal["alert", "another_node"]]:
    is_approved = interrupt(
        {
            "question": "Can you interact with me?",
            # Surface the output that should be
            # reviewed and approved by the human.
            "llm_output": state["llm_output"]
        }
    )

    if is_approved:
        return Command(goto="alert")
    else:
        return Command(goto="another_node")

bio_graph = (
    StateGraph(ResearchState, config_schema=Configuration)
    .add_node("init_question", init_question)
    .add_node("alert", alert)
    .add_node("another_node", another_node)
    .add_edge("__start__", "init_question")
    .compile(name="Research Assistant")
)