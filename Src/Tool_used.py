import os
from typing import Literal
from dotenv import load_dotenv
from tavily import TavilyClient
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import ast
import operator as op
from typing import Union

load_dotenv()
tavily_client = TavilyClient()

gemini_flash_latest = ChatGoogleGenerativeAI(
   model="models/gemini-flash-latest",
   temperature=0,
   max_tokens=None,
   timeout=None,
   max_retries=2,)

@tool
def internet_search(
       query: str,
       max_results: int = 5,
       topic: Literal["general", "news", "finance"] = "general",
       include_raw_content: bool = False,
):
   """Run a web search using Tavily"""
   return tavily_client.search(
       query,
       max_results=max_results,
       include_raw_content=include_raw_content,
       topic=topic,
   )

#@tool
#def weather_api():
#   """Get weather information (not implemented)"""
#   pass
#@tool
#def db_query():
#   """Query database """
#   pass

_OPS = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
    ast.Div: op.truediv, ast.FloorDiv: op.floordiv, ast.Mod: op.mod,
    ast.Pow: op.pow, ast.UAdd: op.pos, ast.USub: op.neg,
}

def _eval_node(node) -> Union[int, float]:
    # Python 3.8+ uses Constant for numbers
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.Expr):
        return _eval_node(node.value)
    raise ValueError("Unsupported expression. Use numbers, + - * / // % ** and parentheses only.")

def safe_calc(expr: str) -> float:
    expr = expr.strip()
    if len(expr) > 200:
        raise ValueError("Expression too long.")
    tree = ast.parse(expr, mode="eval")
    return float(_eval_node(tree.body))

@tool
def calculate(expression: str) -> str:
    """Safely evaluate arithmetic like '(12345*6789)+98765'. Returns only the number."""
    try:
        val = safe_calc(expression)
        if abs(val) > 1e18:
            return "Error: result too large."
        # Pretty output: drop trailing .0 if it's an integer
        return str(int(val)) if val.is_integer() else str(val)
    except Exception as e:
        return f"Error: {e}"


agent=create_agent(
   gemini_flash_latest,
   [internet_search,calculate]
)

result=agent.invoke({
   "messages":[("user", "2**10 + 1000 * 3 - 50 / 2")],
})
print(result["messages"][-1].content)
#print(result)