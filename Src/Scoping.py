import os
from typing import Annotated
from typing_extensions import TypedDict

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

load_dotenv()

class State(TypedDict):
    messages: Annotated[list, add_messages]

llm = ChatGoogleGenerativeAI(
    model="gemini-flash-latest",
    temperature=0.7
)

SYSTEM_PROMPT = """You are a helpful learning guide.
RULE: If a user asks a broad question, DO NOT answer immediately.
INSTEAD, ask a clarifying question to understand their specific goal.
you can ask n number of questions to the user until you clarified about the user query.

Examples:
User: "I want to learn Python." -> You: "For Data Science or Web Dev?"
User: "Explain History." -> You: "Which era?"
User: "What is a variable?" -> You: (Answer directly)

As like the above example, whatever the user asks query don't give the answer blindly.
Only when the user clarifies, provide the full answer.
Note: if the user asks anything like what is what? then you don't need to clarrifiesanything you just gave the results to the user.(e.g. User: "What is Python?" -> You: answer directly)
"""

def chatbot_node(state: State):
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    
    response = llm.invoke(messages)
    return {"messages": [response]}

workflow = StateGraph(State)

workflow.add_node("chatbot", chatbot_node)

workflow.add_edge(START, "chatbot")
workflow.add_edge("chatbot", END)

app = workflow.compile()
