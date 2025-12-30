from langgraph.graph import StateGraph, START, END
from my_agent.utils.state import AgentState, InputState
from my_agent.utils.nodes import get_input_node,intent_clarification_node,get_user_clarification_node,get_refine_topic_node,gather_context_node, generate_checkpoints_node


def build_graph():
    builder = StateGraph(AgentState, input_schema=InputState)
    builder.add_node("get_input", get_input_node)
    builder.add_node("intent_clarification", intent_clarification_node)
    
    builder.add_node("get_user_clarification", get_user_clarification_node)
    builder.add_node("get_refine_topic", get_refine_topic_node)
    
    builder.add_node("gather_context", gather_context_node)
    builder.add_node("create_checkpoints", generate_checkpoints_node)


    builder.add_edge(START, "get_input")
    builder.add_edge("get_input", "intent_clarification")
    
    builder.add_conditional_edges(
    "intent_clarification",
    lambda state: "clarify" if state["needs_clarification"] else "refine",
    {
        "clarify": "get_user_clarification",
        "refine": "get_refine_topic",
    })

    builder.add_edge("get_user_clarification", "intent_clarification")
    
    builder.add_edge("get_refine_topic", "gather_context")
    builder.add_edge("gather_context", "create_checkpoints")
    builder.add_edge("create_checkpoints", END)

    return builder.compile()
