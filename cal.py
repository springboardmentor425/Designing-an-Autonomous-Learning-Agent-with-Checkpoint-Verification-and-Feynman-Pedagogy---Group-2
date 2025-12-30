from langchain_google_genai import ChatGoogleGenerativeAI
import math


# Create model
# Gemini model initialization
gemini_model = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash", 
    temperature=0,
    max_tokens=2000,
)

# Define calculator tool
def calculator(expression):
    """Evaluates mathematical expressions"""
    try:
        # Safe evaluation using Python's eval with math functions
        result = eval(expression, {"__builtins__": {}}, {
            "sqrt": math.sqrt,
            "pow": math.pow,
            "abs": abs,
            "round": round
        })
        return result
    except Exception as e:
        return f"Error: {e}"

# Agent function
def agent_calculator(question):
    # Ask model to extract the math expression
    prompt = f"""Extract only the mathematical expression from this question.
Return ONLY the expression in Python format (use ** for power, sqrt() for square root).

Question: {question}

Expression:"""
    
    response = model.generate_content(prompt)
    expression = response.text.strip()
    
    print(f"Question: {question}")
    print(f"Expression extracted: {expression}")
    
    # Calculate using the tool
    result = calculator(expression)
    print(f"Result: {result}\n")
    
    return result

# Use the agent
agent_calculator("What is 25 * 4 + 10?")
agent_calculator("What is the square root of 144 multiplied by 5?")