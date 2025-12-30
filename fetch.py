import os
from typing import Literal, Optional
from dotenv import load_dotenv
from tavily import TavilyClient
from langchain_core.tools import tool
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
tavily_client = TavilyClient() 

gemini_flash_latest = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    max_tokens=1000,
    timeout=None,
    max_retries=2,
)

@tool
def get_current_time(format_type: str = "full") -> str:
    """Gets the current date and time in various formats.
    
    Args:
        format_type: Type of time/date format to return. Options:
            - "full": Complete date and time (default)
            - "date": Only the date
            - "time": Only the time
            - "day": Day of the week
            - "month": Current month
            - "year": Current year
            - "12hour": 12-hour time format
            - "24hour": 24-hour time format
            - "timestamp": Unix timestamp
            - "iso": ISO 8601 format
    
    Use this when user asks about current time, date, day, month, year, or timestamp.
    """
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

@tool
def fetch_real_time_news(
    query: Optional[str] = None,
    max_results: int = 5,
    days: int = 3
) -> str:
    """Fetches real-time news articles using Tavily API.
    
    Args:
        query: Search query for news. If None, fetches general latest news.
        max_results: Number of news articles to return (default: 5)
        days: Number of days to look back for news (default: 3)
    
    Returns:
        Formatted string with news articles including title, source, and summary.
    
    Use this when user asks about:
        - Latest news, current events, breaking news
        - News about specific topics (technology, sports, politics, etc.)
        - Recent developments or updates on any subject
    """
    try:
        # If no query provided, fetch general news
        search_query = query if query else "latest news today"
        
        # Use Tavily with news topic
        results = tavily_client.search(
            query=search_query,
            max_results=max_results,
            topic="news",
            days=days,
            include_raw_content=False
        )
        
        if not results or 'results' not in results:
            return "No news articles found."
        
        # Format the news results
        news_output = f"📰 Real-time News Results for: '{search_query}'\n"
        news_output += f"Found {len(results['results'])} articles:\n\n"
        
        for idx, article in enumerate(results['results'], 1):
            title = article.get('title', 'No title')
            url = article.get('url', 'No URL')
            content = article.get('content', 'No content available')
            score = article.get('score', 0)
            
            news_output += f"{idx}. {title}\n"
            news_output += f"   Source: {url}\n"
            news_output += f"   Relevance: {score:.2f}\n"
            news_output += f"   Summary: {content[:300]}...\n\n"
        
        return news_output
        
    except Exception as e:
        return f"Error fetching news: {str(e)}"

@tool
def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Run a general web search using Tavily.
    
    Args:
        query: Search query string
        max_results: Number of results to return (default: 5)
        topic: Type of search - "general", "news", or "finance"
        include_raw_content: Whether to include full page content
    
    Use this for general web searches, research, and finding information online.
    """
    return tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )

# Collect all tools
tools = [internet_search, get_current_time, fetch_real_time_news]

# Create agent using initialize_agent
agent = initialize_agent(
    tools=tools,
    llm=gemini_flash_latest,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

# Example usage
if __name__ == "__main__":
    # Test the news fetching
    result = agent.invoke({
        "input": "What are the latest technology news today?"
    })
    print("\n" + "="*50)
    print("FINAL ANSWER:")
    print("="*50)
    print(result["output"])