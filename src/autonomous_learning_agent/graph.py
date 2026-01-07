from langgraph.graph import StateGraph, START,END
from autonomous_learning_agent.state import AgentInputState, AgentState
from autonomous_learning_agent.scope_agent import clarify_with_user, write_research_brief
from autonomous_learning_agent.supervisor_agent import get_structured_research_brief

from langgraph.checkpoint.memory import InMemorySaver


def build_graph():
    builder = StateGraph(AgentState,input_schema=AgentInputState)
    builder.add_node("clarify_with_user",clarify_with_user)
    builder.add_node("write_research_brief",write_research_brief)
    builder.add_node("get_structured_research_brief",get_structured_research_brief)


    builder.add_edge(START,"clarify_with_user")
    builder.add_edge("write_research_brief","get_structured_research_brief")
    builder.add_edge("get_structured_research_brief",END)
    checkpointer = InMemorySaver()
    deep_research_agent = builder.compile(checkpointer=checkpointer)
    return deep_research_agent