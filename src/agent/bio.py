from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, TypedDict
import asyncio
import requests
import os
from datetime import datetime
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from tavily import TavilyClient
from typing import Literal
from langgraph.types import interrupt, Command
from src.agent import redis_interaction

from google.genai import types
from google import genai
import httpx

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
    questions: str = ""

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

    if is_approved == 'yes':
        state.human_answer = "Approved"
        return Command(goto="who", update={"llm_question": "Are you patient or supporter?"})
    else:
        return Command(goto="alert", update={"action": "Call for ambulance."})
    
def research(state: State, config: RunnableConfig) -> Command[Literal["ask_questions"]]:

    configuration = config["configurable"]
    google_api = configuration.get("google-api", os.getenv("GOOGLE_API_KEY"))

    user_prompt = f"You are a professional healthcare doctor. \
        Your task is to ask the user some (3) critical questions.\
        The user that you will ask questions is a {status['user']}. \
        Only yes or no questions. \
        Return only the questions, separated by a slash, no explanations."
    
    print(f"User prompt: {user_prompt}")

    doc_url = "https://www.b2btrainingnetwork.ie/wp-content/uploads/2024/08/phecc-cfr-cpgs.pdf"
    
    client = genai.Client(api_key=google_api)

    doc_data = httpx.get(doc_url).content

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[
            types.Part.from_bytes(
                data=doc_data,
                mime_type='application/pdf',
            ),
            user_prompt
            ],
        config=types.GenerateContentConfig(
            max_output_tokens=512,
            temperature=0.1
        )
    )
    response = {'choices': [{'message' : {'content' : response.text}}]}

    questions = response['choices'][0]['message']['content'].strip().split('/')

    print(f"Generated questions: {questions}")

    questions = ['Where do you feel the pain?'] + questions

    return Command(goto="ask_questions", update={"questions": questions})

def ask_questions(state: State, config: RunnableConfig) -> Command[Literal["make_decision"]]:
    """Ask the generated questions to the user."""
    
    questions = state.questions

    save_dct = {}

    for question in questions:
        is_approved = interrupt(question)
        save_dct[question] = is_approved
    
    redis_interaction.input_record(str(save_dct))

    return Command(goto="make_decision", update={"human_answer": str(save_dct)})

def make_decision(state: State, config: RunnableConfig) -> Command[Literal["alert", "minor_alert"]]:

    configuration = config["configurable"]
    google_api = configuration.get("google-api", os.getenv("GOOGLE_API_KEY"))

    context = redis_interaction.export_records(datetime.now())

    user_prompt = f"You are a professional healthcare doctor. You will now receive the answers \
        from the user to the questions you asked. \
        Your task is to analyze the answers and make a decision whether to call for an ambulance or not. \
        The user that you will analyze is a {status['user']}. \
        Return only the decision, either 'call ambulance' or 'do not call ambulance'. \
        Do not return any explanations, just the decision. \
            The answers are: {state.human_answer}. Previous records are {context}."
    
    client = genai.Client(api_key=google_api)

    my_file = client.files.upload(file="./assets/images/images.jpeg")
    
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[
            my_file,
            user_prompt
            ],
        config=types.GenerateContentConfig(
            max_output_tokens=512,
            temperature=0.1
        )
    )

    decision = response.text

    if "call ambulance" in decision.lower():
        return Command(goto="alert", update={"action": "Call for ambulance."})
    else:
        return Command(goto="minor_alert", update={"action": 'Suggest calling the ambulance.'})

import tkinter as tk
from tkinter import messagebox

def alert(state: State) -> ResearchState:
    """Process the state and show a big red cross alert in the UI."""
    if state.is_critical:  # Giả sử có thuộc tính is_critical
        show_alert()

    return state

def show_alert():
    """Hiển thị cảnh báo với hình cross đỏ."""
    # Tạo cửa sổ chính
    root = tk.Tk()
    root.title("Alert")

    # Tạo canvas để vẽ hình
    canvas = tk.Canvas(root, width=200, height=200, bg='white')
    canvas.pack()

    # Vẽ hình cross đỏ
    canvas.create_line(50, 50, 150, 150, fill='red', width=10)
    canvas.create_line(150, 50, 50, 150, fill='red', width=10)

    # Hiển thị cửa sổ
    root.mainloop()

def minor_alert(state: State) -> ResearchState:
    """Process the state in another way."""
    # Example processing logic
    return state

# Example of how to add nodes and edges to the graph
bio_graph = (
    StateGraph(ResearchState, config_schema=Configuration)
    .add_node("who", who)
    .add_node("init_question", init_question)
    .add_node("research", research)
    .add_node("make_decision", make_decision)
    .add_node("alert", alert)
    .add_node("minor_alert", minor_alert)
    .add_node("ask_questions", ask_questions)
    .add_edge("__start__", "init_question")
)