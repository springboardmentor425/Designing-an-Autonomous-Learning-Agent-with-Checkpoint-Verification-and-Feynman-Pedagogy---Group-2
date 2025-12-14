from langgraph.graph import StateGraph
from core.state import TutorState
from core.context_manager import gather_from_notes, gather_from_web
from core.context_validator import validate_context
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(temperature=0)

def gather_context_node(state: TutorState):
    checkpoint = state["checkpoint"]
    notes = state.get("user_notes")

    if notes:
        context = gather_from_notes(checkpoint, notes)
        source = "notes"
    else:
        context = gather_from_web(checkpoint)
        source = "web"

    score = validate_context(llm, checkpoint, context)

    if score < 4:
        context = gather_from_web(checkpoint)
        source = "web"
        score = validate_context(llm, checkpoint, context)

    return {
        "gathered_context": context,
        "context_source": source,
        "context_score": score
    }
