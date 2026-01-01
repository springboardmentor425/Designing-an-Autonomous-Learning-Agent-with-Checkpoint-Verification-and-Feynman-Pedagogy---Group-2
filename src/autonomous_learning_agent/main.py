from langgraph.graph import StateGraph, START,END
from autonomous_learning_agent.scoping.scope_state import AgentInputState, AgentState, ClarifyWithUser, ResearchQuestion
from autonomous_learning_agent.scoping.scope_agent import clarify_with_user, write_research_brief
from langgraph.checkpoint.memory import InMemorySaver


def build_graph():
    builder = StateGraph(AgentState,input_schema=AgentInputState)
    builder.add_node("clarify_with_user",clarify_with_user)
    builder.add_node("write_research_brief",write_research_brief)

    builder.add_edge(START,"clarify_with_user")
    builder.add_edge("write_research_brief",END)
    checkpointer = InMemorySaver()
    deep_research_agent = builder.compile(checkpointer=checkpointer)
    return deep_research_agent