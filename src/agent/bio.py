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

from .research_assistant import get_llm

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


status = {
    'user' : "",
    'state' : ""
}


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

@dataclass
class State:
    llm_question: str = "Are you concious?"
    human_answer: str = ""
    action: str = ""

def who(state: State) -> Command[Literal["research"]]:
    
    is_approved = interrupt(state.llm_question)

    if is_approved:
        state.human_answer = "the user is patient"
        status['user'] = "patient"
        return Command(goto="research")
    else:
        state.human_answer = "the user is supporter"
        status['user'] = "supporter"
        return Command(goto="research")

def init_question(state: State) -> Command[Literal["who", "alert"]]:

    is_approved = interrupt(state.llm_question)

    if is_approved:
        state.human_answer = "Approved"
        return Command(goto="who", update={"llm_question": "Are you patient or supporter?"})
    else:
        return Command(goto="alert")
    
def research(state: State, config: RunnableConfig) -> ResearchState:
    
    llm = get_llm(config)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system", 
            "You are a professional healthcare doctor. \
                Your task is to ask the user some questions.\
                    The user that you will ask questions is a {user}. \
                        Return only the questions, no explanations."
        ),
        (
            "user", 
            "Create 3 effective search queries for the following research question. \
            Return only the queries separated by newlines, no explanations:\n\n{question}"
        )
    ])

    chain = prompt | llm | StrOutputParser()
    search_queries = await chain.ainvoke({"question": state.question})

    state.search_queries = search_queries.strip().split("\n")

    return state

def alert(state: State) -> ResearchState:
    """Process the state in another way."""
    # Example processing logic
    state.action = "Call for ambulance."
    return state

# Example of how to add nodes and edges to the graph
bio_graph = (
    StateGraph(ResearchState, config_schema=Configuration)
    .add_node("who", who)
    .add_node("init_question", init_question)
    .add_node("research", research)
    .add_node("alert", alert)
    .add_edge("__start__", "init_question")
)