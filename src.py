from tavily import TavilyClient
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
load_dotenv()

# Initialize Tavily client
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

gemini_flash_latest = ChatGoogleGenerativeAI(
    model="models/gemini-flash-latest",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,)

# User query
query = "Today what"

# Search the web using Tavily
search_results = tavily.search(query=query, max_results=5)

# Format search results for context
context = "Here are the latest search results:\n\n"
for result in search_results['results']:
    context += f"Title: {result['title']}\n"
    context += f"Content: {result['content']}\n"
    context += f"URL: {result['url']}\n\n"

# Create prompt with search context
prompt = f"{context}\nBased on the above information, answer: {query}"

response=gemini_flash_latest.invoke(prompt)

if hasattr(response, 'content'):
    response_text = response.content
else:
    response_text = str(response)

# Clean and format output
print("=" * 70)
print("📰 BASED ON SEARCH RESULTS")
print("=" * 70)
print()
print(response_text)
print()
print("=" * 70)
print("📌 SOURCE REFERENCES")
print("=" * 70)
for idx, result in enumerate(search_results['results'], 1):
    print(f"\n[Source {idx}]")
    print(f"Title: {result['title']}")
    print(f"URL: {result['url']}")
print()
print("=" * 70)