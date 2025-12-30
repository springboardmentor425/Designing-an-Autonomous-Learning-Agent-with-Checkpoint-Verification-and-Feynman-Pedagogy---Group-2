from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",
    temperature=0
)

# Prompt
prompt = PromptTemplate.from_template(
    "Calculate the following expression and return only the final answer: {expression}"
)

# Runnable chain
calculator = prompt | llm

# Run
result = calculator.invoke({"expression": "45 * 85 + 5"})
print(result.content)
