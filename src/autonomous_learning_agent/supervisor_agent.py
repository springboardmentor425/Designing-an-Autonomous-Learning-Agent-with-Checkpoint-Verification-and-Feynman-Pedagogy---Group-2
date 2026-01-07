from typing_extensions import Literal
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from autonomous_learning_agent.prompts import create_structured_research_brief
from autonomous_learning_agent.state import SupervisorState
from autonomous_learning_agent.state import StructuredResearchBrief
from utils import get_today_str
from langgraph.types import Command



# LLM model
model=init_chat_model("google_genai:models/gemini-flash-lite-latest")


def supervisor(state: SupervisorState) -> Command[Literal["supervisor_tools"]]:
    """Lead research supervisor that plans research strategy and delegates to researchers.
    
    The supervisor analyzes the research brief and decides how to break down the research
    into manageable tasks. It can use think_tool for strategic planning, ConductResearch
    to delegate tasks to sub-researchers, or ResearchComplete when satisfied with findings.
    
    Args:
        state: Current supervisor state with messages and research context
        
    Returns:
        Command to proceed to supervisor_tools for tool execution
    """
    # Set up structured output model
    structured_output_model = model.with_structured_output(StructuredResearchBrief, method="json_mode")
    
    # Available tools: research delegation, completion signaling, and strategic thinking
    # lead_researcher_tools = [ConductResearch, ResearchComplete, think_tool]
    
    # Invoke the model with clarification instructions
    response = structured_output_model.invoke([
        HumanMessage(content=create_structured_research_brief.format(
            brief=state["research_brief"], 
            date=get_today_str()
        ))
    ])
    return {
        "structured_research_brief": response
    }


