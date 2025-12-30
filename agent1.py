import os
from typing import Literal
from dotenv import load_dotenv
from datetime import datetime
import PyPDF2
from tavily import TavilyClient
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

# Tavily client
tavily_client = TavilyClient()

# Gemini model initialization
# ❗ THIS IS THE CORRECT, WORKING MODEL NAME
gemini_model = ChatGoogleGenerativeAI(
    model="models/gemini-1.0-pro",   # <-- VALID MODEL
    temperature=0,
    max_tokens=1000,
    timeout=None,
    max_retries=2
)

# ------------------------------
#   TOOL: CURRENT TIME FETCHER
# ------------------------------
@tool
def get_current_time(format_type: str = "full") -> str:
    """Return the current date/time in various formats."""
    now = datetime.now()

    formats = {
        "full": now.strftime("%Y-%m-%d %H:%M:%S"),
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "day": now.strftime("%A"),
        "month": now.strftime("%B %Y"),
        "year": str(now.year),
        "12hour": now.strftime("%I:%M:%S %p"),
        "24hour": now.strftime("%H:%M:%S"),
        "timestamp": str(int(now.timestamp())),
        "iso": now.isoformat(),
    }

    return formats.get(format_type.lower(), formats["full"])


# ------------------------------
#   TOOL: PDF SUMMARIZER
# ------------------------------
@tool
def summarize_pdf(file_path: str, max_words: int = 200) -> str:
    """Extract & summarize a PDF using Gemini."""
    try:
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text = ""

            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"

        if not text.strip():
            return "Could not extract text from this PDF."

        prompt = (
            f"Summarize the following PDF content in about {max_words} words.\n"
            f"Keep the summary clear and concise.\n\n"
            f"PDF TEXT:\n{text}"
        )

        response = gemini_model.invoke(prompt)
        return response.content

    except Exception as e:
        return f"Error processing PDF: {str(e)}"


# ------------------------------
#   TOOL: INTERNET SEARCH
# ------------------------------
@tool
def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Run a web search using Tavily."""
    return tavily_client.search(
        query=query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )


# ------------------------------
#   CREATE AGENT
# ------------------------------
agent = create_agent(
    model=gemini_model,
    tools=[internet_search, get_current_time, summarize_pdf]
)

# ------------------------------
#   RUN AGENT
# ------------------------------
result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": "Summarize this PDF: student_scores.pdf"
    }]
})

print(result["messages"][-1].content)
