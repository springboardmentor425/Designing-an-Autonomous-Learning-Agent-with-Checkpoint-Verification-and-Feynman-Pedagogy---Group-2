def validate_context(llm, checkpoint, context):
    prompt = f"""
    Topic: {checkpoint.topic}
    Objectives: {checkpoint.objectives}

    Context:
    {context}

    Rate the relevance of the context to the objectives from 1 to 5.
    Return ONLY the number.
    """
    score = float(llm.predict(prompt))
    return score

# Rule

# Score ≥ 4 → accept context
# Score < 4 → reject and re-gather