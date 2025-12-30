import os
from typing import Literal
from dotenv import load_dotenv
from tavily import TavilyClient
from datetime import datetime
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# Initialize Tavily
tavily_client = TavilyClient()

# Gemini model
gemini_flash_latest = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    max_tokens=1000,
    timeout=None,
    max_retries=2,
)

# ============================================================
# 1. TIME TOOL
# ============================================================
@tool
def get_current_time(format_type: str = "full") -> str:
    """Return the current date and time."""
    
    now = datetime.now()
    formats = {
        "full": now.strftime("%Y-%m-%d %H:%M:%S"),
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "day": now.strftime("%A"),
        "month": now.strftime("%B"),
        "year": str(now.year),
        "12hour": now.strftime("%I:%M:%S %p"),
        "24hour": now.strftime("%H:%M:%S"),
        "timestamp": str(int(now.timestamp())),
        "iso": now.isoformat()
    }

    return formats.get(format_type.lower(), formats["full"])


# ============================================================
# 3. TAVILY GENERAL SEARCH TOOL
# ============================================================
@tool
def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False
):
    """Run a web search using Tavily."""
    return tavily_client.search(
        query=query,
        max_results=max_results,
        topic=topic,
        include_raw_content=include_raw_content
    )

# ============================================================
# 4. NEWS API TOOL (NEW TOOL)
# ============================================================
@tool
def news_api_search(query: str, max_results: int = 5):
    """
    Search news using NewsAPI and return real URLs.
    Requires NEWS_API_KEY in .env
    """
    from newsapi import NewsApiClient

    newsapi = NewsApiClient(api_key=os.getenv("NEWS_API_KEY"))

    response = newsapi.get_everything(q=query, language="en", sort_by="relevancy")

    articles = response.get("articles", [])[:max_results]

    if not articles:
        return "No news found."

    result = "Top News Articles:\n"

    for a in articles:
        title = a.get("title", "No title")
        url = a.get("url", "No URL")
        source = a.get("source", {}).get("name", "Unknown Source")

        result += f"\n• {title}\n  Source: {source}\n  URL: {url}\n"

    return result

# ============================================================
# CREATE AGENT
# ============================================================
agent = create_agent(
    model=gemini_flash_latest,
    tools=[internet_search, get_current_time, get_pdf_info, news_api_search]
)

# ============================================================
# EXAMPLE USER REQUEST
# ============================================================
result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "Summarize student_scores.pdf"
        }
    ]
})

print(result["messages"][-1].content)
