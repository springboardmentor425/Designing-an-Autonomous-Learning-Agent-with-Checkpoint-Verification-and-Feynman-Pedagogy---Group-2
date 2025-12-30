from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Define proper state schema
class CalcState(dict):
    expression: str
    result: str

llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",
    temperature=0
)

prompt = PromptTemplate.from_template(
    "Calculate the following expression and return only the final answer: {expression}"
)

def calculate(state: CalcState):
    chain = prompt | llm
    res = chain.invoke({"expression": state["expression"]})
    return {"result": res.content}

graph = StateGraph(CalcState)
graph.add_node("calculator", calculate)
graph.set_entry_point("calculator")
graph.add_edge("calculator", END)

app = graph.compile()

while True:
    exp = input("Enter a math expression (or type exit): ")
    if exp.lower() == "exit":
        break

    output = app.invoke({"expression": exp})
    print("Answer:", output["result"])
