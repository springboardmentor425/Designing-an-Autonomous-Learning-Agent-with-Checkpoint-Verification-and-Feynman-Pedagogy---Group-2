import os
from typing import Annotated
from typing_extensions import TypedDict

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

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
"""

def chatbot_node(state: State):
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    
    response = llm.invoke(messages)
    return {"messages": [response]}

workflow = StateGraph(State)

workflow.add_node("chatbot", chatbot_node)

workflow.add_edge(START, "chatbot")
workflow.add_edge("chatbot", END)

memory = MemorySaver()

app = workflow.compile(checkpointer=memory)

def run_chat():
    config = {"configurable": {"thread_id": "session_graph_1"}}
    
    print("--- LangGraph Clarification Agent (Explicit) ---")
    print("Type 'quit' to exit.\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit"]:
            break
            
        events = app.stream(
            {"messages": [HumanMessage(content=user_input)]}, 
            config, 
            stream_mode="values"
        )
        
        for event in events:
            if "messages" in event:
                last_msg = event["messages"][-1]
                if last_msg.type == "ai":
                    print(f"Agent: {last_msg.content}")

if __name__ == "__main__":
    run_chat()