# scoping_agent/node.py
from typing_extensions import TypedDict
from typing import List, Dict
from .scoper import Scoper

class InputState(TypedDict, total=False):
    user_query: str
    task_id: str
    user_answers: Dict[str,str]

class AgentState(TypedDict, total=False):
    need_clarification: bool
    clarifying_questions: List[str]
    refined_query: str
