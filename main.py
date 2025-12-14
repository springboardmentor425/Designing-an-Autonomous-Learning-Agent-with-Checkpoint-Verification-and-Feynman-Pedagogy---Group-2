import os
import sys

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from checkpoints.sample_checkpoint import checkpoint_1
from core.state_graph import app

initial_state = {
    "checkpoint": checkpoint_1,
    "user_notes": "Machine learning allows computers to learn from data..."
}

result = app.invoke(initial_state)

print("Context Source:", result["context_source"])
print("Context Score:", result["context_score"])
print("Context:", result["gathered_context"][:500])
