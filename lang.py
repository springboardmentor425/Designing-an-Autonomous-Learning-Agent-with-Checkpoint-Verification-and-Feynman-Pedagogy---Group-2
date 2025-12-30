import os
import dotenv
from tavily import TavilyClient
from langgraph.graph import StateGraph
from langgraph.graph import START, END
from typing_extensions import TypedDict, List, Optional, Dict, Any




dotenv.load_dotenv()
tavily_client = TavilyClient(os.getenv("TAVILY_API_KEY"))


class InputState(TypedDict):
    checkpoint_topic: str


class AgentState(TypedDict):
    checkpoint_topic: str
    gathered_context: str
    context_source: str
    context_metadata: Dict[str, Any]
    search_attempts: int
    max_search_attempts: int
    error_message: Optional[str]


def search_web(query: str,max_retries: int = 3) -> List[Dict[str, Any]]:
    try:
        response=tavily_client.search(
            query=query,
            auto_parameters=True,
            max_results=10,
            search_depth="advanced",
            max_retries=max_retries,
            include_raw_content=True,
            time_range="month"
        )
        return response
    
    except Exception as e:
        print(f"Error during web search: {e}")




def gather_context_node(state: InputState) -> AgentState:
    
    topic = state["checkpoint_topic"]
    search_results = search_web(topic)

    return {
        "gathered_context":search_results,
        "context_source":search_web
            
            
            }


builder=StateGraph(AgentState,input_schema=InputState)

builder.add_node("gather_context",gather_context_node)
builder.add_edge(START,"gather_context")
builder.add_edge("gather_context",END)
graph=builder.compile()