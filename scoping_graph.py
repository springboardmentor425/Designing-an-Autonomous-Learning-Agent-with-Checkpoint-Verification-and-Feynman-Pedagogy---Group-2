from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

# ---------- STATE ----------
class AgentState(TypedDict):
    user_input: str
    clarified_scope: str


# ---------- NODE ----------
def scope_clarifier(state: AgentState) -> AgentState:
    question = state["user_input"]

    clarified = f"""
Clarified scope:
- User intent: {question}
- Assumptions removed
- Ready for next agent
"""

    return {
        "user_input": question,
        "clarified_scope": clarified
    }


# ---------- GRAPH ----------
graph = StateGraph(AgentState)

graph.add_node("scope", scope_clarifier)

graph.add_edge(START, "scope")
graph.add_edge("scope", END)

app = graph.compile()


# ---------- RUN ----------
if __name__ == "__main__":
    result = app.invoke(
        {"user_input": "Build an autonomous learning agent"}
    )

    print("\n=== OUTPUT ===")
    print(result["clarified_scope"])
