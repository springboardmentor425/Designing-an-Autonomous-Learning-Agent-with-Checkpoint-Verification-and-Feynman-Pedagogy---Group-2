import os
from typing_extensions import TypedDict, List, Optional, Dict, Any
from typing import Literal
from my_agent.utils.state import AgentState, InputState, LearningPlan, AmbiguityCheck, RefinedTopic
from tavily import TavilyClient
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.types import interrupt, Command
from dotenv import load_dotenv

load_dotenv() 

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0,
)

# NODE 1 (get user input)
def get_input_node(state:InputState)->AgentState:
    topic = state['topic']
    return{
        "topic":topic
    }



# NODE 2 (check if user clarificatioin is needed)
def intent_clarification(topic: str, clarification: str | None) -> AmbiguityCheck:
    try:
        prompt = f"""
        You are an intent analysis agent.
        Analyze the user's topic and decide if clarification is required
        before performing a web search.

        Rules:
        - Clarification is required if the topic is ambiguous, too broad,
        or missing critical parameters such as level, goal, scope, or timeframe.
        - If clarification is required, ask ONLY ONE clear question.

        User topic:
        "{topic}"

        Additional user clarification (if any):
        "{clarification}"

        """
        return llm.with_structured_output(AmbiguityCheck).invoke(prompt)
    
    except Exception as e:
        print(f"Error during intent clarification: {e}")
        
def intent_clarification_node(state: AgentState) -> AgentState:
    context = (
    "\n".join(state.get("user_clarification", [])) or "No clarification provided yet.")
    result = intent_clarification(
        topic = state["topic"],
        clarification = context
    )

    return {
        "needs_clarification": result["needs_clarification"],
        "clarification_question": result["clarification_question"]
    }



# NODE 3 (get the answer for the clarification question)
def get_user_clarification_node(state: AgentState) -> AgentState:
    user_response = interrupt(state["clarification_question"])
    return {
        "user_clarification": [user_response],
        "clarification_question": None 
    }



# NODE 4 (after clarification get the final refined topic)
def get_refine_topic_node(state: AgentState) -> AgentState:
    prompt = f"""
    Refine the following topic into a clear, search-ready query:
    Topic: "{state['topic']}"
    User clarification: "{state['user_clarification']}"
    """
    refined = llm.with_structured_output(RefinedTopic).invoke(prompt)
    return {"refined_topic": refined['refined_topic']}



# NODE 5 (gather context using web search)
tavily_client = TavilyClient(os.getenv("TAVILY_API_KEY"))

def search_web(query: str,max_retries: int = 3) -> List[Dict[str, Any]]:
    try:
        response = tavily_client.search(
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

def gather_context_node(state: AgentState) -> AgentState:
    
    topic = state["refined_topic"]
    search_results = search_web(topic)

    return {
        "gathered_context":search_results,
        "context_source":search_web
        }



# NODE 6 (gather checkpoints based on context & topic)

def generate_checkpoints(context:str, topic:str)-> List[Dict[str, Any]]:
    try:
        prompt = f"""
        You are an expert curriculum designer.
        Topic:
        {topic}

        Context:
        {context}

        Create a structured learning plan with progressive checkpoints.
        """
        result = llm.with_structured_output(LearningPlan).invoke(prompt)
        return result
    
    except Exception as e:
        print(f"Error during checkpoint generation: {e}")

def generate_checkpoints_node(state: AgentState) -> AgentState:
    topic = state["topic"]
    context = state["gathered_context"]
    checkpoints = generate_checkpoints(context, topic)
    return {"checkpoints":checkpoints}


