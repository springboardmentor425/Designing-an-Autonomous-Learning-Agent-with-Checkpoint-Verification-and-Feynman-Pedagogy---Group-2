# scoping_agent/evaluator.py

def detect_goal_conflict(previous: str, current: str) -> bool:
    """
    Detects intent drift or contradiction across turns.
    Example:
      - "I want to learn Python"
      - "I want to learn Data Science"
    """
    prev = previous.lower()
    curr = current.lower()

    if "learn python" in prev and "learn data science" in curr:
        return True

    return False
