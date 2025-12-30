import os
from typing import Literal
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from datetime import datetime
from tavily import TavilyClient
from langgraph.checkpoint.memory import InMemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

# Tavily client
tavily_client = TavilyClient()

# Gemini model initialization   
gemini_model = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash", 
    temperature=0,
    max_tokens=2000,
)

@tool
def get_datetime() -> str:
    """Get the current date and time in a simple format."""
    now = datetime.now()
    return f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')} ({now.strftime('%A')})"

@tool
def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Run a web search using Tavily."""
    return tavily_client.search(
        query=query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )

# Create the agent
agent = create_agent(
    model=gemini_model,
    tools = [internet_search, get_datetime],
    checkpointer=InMemorySaver() #Short Term Memory
)

# Invoke the agent
config = {"configurable": {"thread_id": "1"}}
chat_history = []

while True:
    user_input = input("\n💭 You: ")
    
    if user_input.lower() == 'history':
        print("\n📜 Chat History:")
        for i, entry in enumerate(chat_history, 1):
            print(f"\n{i}. Q: {entry['question']}")
            print(f"   A: {entry['answer'][:100]}...")
        continue
    
    if user_input.lower() in ['quit', 'exit']:
        print(f"\n👋 Goodbye! You asked {len(chat_history)} questions.")
        break
    
    if not user_input.strip():
        continue
    
    result = agent.invoke({
        "messages": [{"role": "user", "content": user_input}]
    }, config)
    
    answer = result["messages"][-1].content
    chat_history.append({"question": user_input, "answer": answer})
    
    print(f"\n🤖 Agent: {answer}")
    print("-"*80)