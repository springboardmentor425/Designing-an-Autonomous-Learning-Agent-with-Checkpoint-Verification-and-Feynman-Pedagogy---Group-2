from pydantic import BaseModel
from typing import List

class Checkpoint(BaseModel):
    checkpoint_id: str
    topic: str
    objectives: List[str]
    success_threshold: float = 0.7
