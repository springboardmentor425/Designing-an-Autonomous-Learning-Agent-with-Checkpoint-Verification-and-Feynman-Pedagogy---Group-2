from datetime import datetime, timedelta
import os
import requests

from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent  

# -----------------------
# TOOL 1: Current Time
# -----------------------
@tool
def get_current_time(format_type: str = "full") -> str:
    """
    Returns the current date and time in various formats.

    Args:
        format_type: Type of time format (full, date, time, iso)
    """
    now = datetime.now()
    formats = {
        "full": now.strftime("%Y-%m-%d %H:%M:%S"),
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "iso": now.isoformat(),
    }
    return formats.get(format_type.lower(), formats["full"])

# -----------------------
# TOOL 2: News API
# -----------------------
@tool
def news_api_search(query: str, max_results: int = 5) -> str:
    """
    Fetches news articles from NewsAPI.org and returns formatted results.

    Args:
        query: The search query.
        max_results: Maximum number of articles to fetch.
    """
    api_key = os.getenv("NEWS_API_KEY")
    if not api_key:
        return "ERROR: NEWS_API_KEY missing."

    url = "https://newsapi.org/v2/everything"
    from_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    params = {
        "q": query,
        "apiKey": api_key,
        "pageSize": max_results,
        "sortBy": "publishedAt",
        "from": from_date,
        "language": "en",
    }

    try:
        r = requests.get(url, params=params)
        data = r.json()
        if data.get("status") != "ok":
            return f"NewsAPI error: {data.get('message')}"

        text = f"Latest news for '{query}':\n\n"
        for i, art in enumerate(data["articles"], 1):
            text += f"{i}. {art['title']}\nSource: {art['source']['name']}\nURL: {art['url']}\n\n"
        return text
    except Exception as e:
        return f"Error fetching news: {str(e)}"


# -----------------------
# LLM & Agent Setup
# -----------------------
llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash")
tools = [get_current_time, news_api_search]

agent = create_agent(model=llm, tools=tools)

# -----------------------
# TEST AGENT
# -----------------------
if __name__ == "__main__":
    print("--- Current Time ---")
    result = agent.invoke({"messages": [{"role": "user", "content": "What is the current time?"}]})
    print(result["output"])

    print("\n--- Latest AI News ---")
    result2 = agent.invoke({"messages": [{"role": "user", "content": "Give me latest AI news"}]})
    print(result2["output"])
