from typing import List, Dict, Optional
from checkpoints.checkpoint_schema import Checkpoint

class TutorState(dict):
    checkpoint: Optional[Checkpoint]
    user_notes: Optional[str]
    gathered_context: Optional[str]
    context_source: Optional[str]   # "notes" or "web"
    context_score: Optional[float]
