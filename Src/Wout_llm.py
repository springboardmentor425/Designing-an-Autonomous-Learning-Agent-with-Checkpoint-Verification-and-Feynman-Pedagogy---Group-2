import os
from typing import TypedDict, List
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from tavily import TavilyClient

load_dotenv()

class AgentState(TypedDict):
    query: str         
    results: List[str] 

def web_search_node(state: AgentState):
    query = state['query']
    
    tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    
    response = tavily.search(query=query, max_results=2)
    
    extracted_text = [result['content'] for result in response['results']]
    
    return {"results": extracted_text}

def decide_next_step(state: AgentState):

    if not state.get('results'):
        return "search"
    return "done"

workflow = StateGraph(AgentState)

workflow.add_node("web_search", web_search_node)

workflow.set_entry_point("web_search")

workflow.add_edge("web_search", END)

app = workflow.compile()

if __name__ == "__main__":

    user_query = "What is Python"

    final_state = app.invoke({"query": user_query, "results": []})

    for i, res in enumerate(final_state['results']):
        print(f"Result {i+1}: {res[:]}")