import operator
from typing_extensions import Optional,Annotated,List,Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages
from pydantic import BaseModel,Field

class AgentInputState(MessagesState):
    """Input state for the full agent - only contains messages from user input."""
    pass

class AgentState(MessagesState):
    """
    Main state for the full multi-agent research system.

    Extends MessagesState with additional fields for research coordination.
    Note: Some fields are duplicated across different state classes for proper
    state management between subgraphs and the main workflow.
    """
    research_brief:Optional[str]
    supervisor_messages:Annotated[Sequence[BaseMessage],add_messages]
    raw_notes:Annotated[list[str],operator.add]=[]
    notes:Annotated[list[str],operator.add]=[]
    final_report:str

# structured output schema
class ClarifyWithUser(BaseModel):
    """Schema for user clarification decision and questions."""
    need_clarification:bool=Field(
        description="Whether the user needs to be asked a clarification qustion"
    )
    question:str=Field(
        description=" A question to ad the user to clarify the report scope"
    )
    verification:str=Field(
        description="verify message that will start research after user provided the necessary information"
    )
class ResearchQuestion(BaseModel):
    """Schema for structured research brief generation."""
    research_brief:str=Field(
        description="A research qustion that will be used to guide the research"
    )
