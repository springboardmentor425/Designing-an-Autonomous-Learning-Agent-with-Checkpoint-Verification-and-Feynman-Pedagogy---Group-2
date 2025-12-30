import dotenv
dotenv.load_dotenv()

from langgraph.graph import StateGraph, START, END

from scoping_agent.llm_tavily import GeminiWrapper
from scoping_agent.memory import ScopingMemory
from scoping_agent.scoper import Scoper
from scoping_agent.node import InputState, AgentState

# -----------------------------
# Initialize core components
# -----------------------------
llm = GeminiWrapper()
memory = ScopingMemory()
scoper = Scoper(llm=llm, memory=memory)

# -----------------------------
# LangGraph node
# -----------------------------
def scoping_node(state: InputState) -> AgentState:
    if state.get("user_answers"):
        refined = scoper.apply_user_answers(
            user_query=state["user_query"],
            task_id=state.get("task_id", "task_1"),
            answers=state["user_answers"]
        )
        return {
            "need_clarification": False,
            "clarifying_questions": [],
            "refined_query": refined
        }

    result = scoper.scope(state["user_query"])
    return {
        "need_clarification": result["need_clarify"],
        "clarifying_questions": result["questions"],
        "refined_query": result.get("refined_query")
    }

# -----------------------------
# Build graph
# -----------------------------
builder = StateGraph(AgentState, input_schema=InputState)

builder.add_node("scoping", scoping_node)
builder.add_edge(START, "scoping")
builder.add_edge("scoping", END)
graph = builder.compile()
