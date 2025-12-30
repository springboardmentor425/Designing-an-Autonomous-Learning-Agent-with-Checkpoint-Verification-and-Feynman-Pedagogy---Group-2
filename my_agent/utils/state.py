from typing import Annotated
from typing_extensions import TypedDict, List, Optional, Dict, Any
from operator import add


class InputState(TypedDict):
    topic: str


class AgentState(TypedDict):
    topic: str
    
    needs_clarification: bool
    clarification_question: Optional[str]
    user_clarification: Annotated[list[str], add]
    refined_topic: Optional[str]
    
    gathered_context: str
    context_source: str
    
    checkpoints: Dict[str, Any]


class Checkpoint(TypedDict):
    checkpoint_title: str
    checkpoint_description: str


class LearningPlan(TypedDict):
    topic: str
    checkpoints: List[Checkpoint]
    
    
class AmbiguityCheck(TypedDict):
    needs_clarification: bool
    clarification_question: Optional[str]


class RefinedTopic(TypedDict):
    refined_topic: str