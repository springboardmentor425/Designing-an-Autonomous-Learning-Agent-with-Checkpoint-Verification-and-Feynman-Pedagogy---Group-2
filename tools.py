import os
from typing import Literal
from dotenv import load_dotenv
from tavily import TavilyClient
from langchain_core.tools import tool
from langchain.agents import create_agent
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
tavily_client = TavilyClient() 

# FIXED: Correct model name for Google Gemini
gemini_flash_latest = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest",  # Changed from "models/gemini-flash-latest"
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
def get_pdf_info(file_path: str, summarize: bool = True) -> str:
    """Gets metadata and optionally a summary of a PDF file.
    
    Args:
        file_path: Path to the PDF file
        summarize: If True, extracts text and generates a summary. Default is True.
    """
    try:
        import PyPDF2
        from collections import Counter
        
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Basic metadata
            metadata = pdf_reader.metadata
            num_pages = len(pdf_reader.pages)
            
            info = f"PDF Information:\nFile: {file_path}\nTotal Pages: {num_pages}\n"
            
            if metadata:
                info += "\nMetadata:\n"
                if metadata.title:
                    info += f"  Title: {metadata.title}\n"
                if metadata.author:
                    info += f"  Author: {metadata.author}\n"
                if metadata.subject:
                    info += f"  Subject: {metadata.subject}\n"
            
            # Optional summarization
            if summarize:
                text_content = ""
                # Extract all pages for better summary
                for i, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    if text:
                        text_content += text + "\n"
                
                if text_content:
                    # Extract common keywords
                    words = [w.lower() for w in text_content.split() if len(w) > 4]
                    common = Counter(words).most_common(10)
                    keywords = ", ".join([w for w, _ in common])
                    
                    info += f"\nKeywords: {keywords}\n"
                    info += f"\nFull Content:\n{text_content}\n"
                else:
                    info += "\nNote: Could not extract text from PDF\n"
            
            return info
            
    except FileNotFoundError:
        return f"Error: File '{file_path}' not found"
    except ImportError:
        return "Error: PyPDF2 not installed. Run: pip install PyPDF2"
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

    
@tool
def internet_search(
        query: str,
        max_results: int = 5,
        topic: Literal["general", "news", "finance"] = "general",
        include_raw_content: bool = False,
) -> str:
    """Run a web search using Tavily"""
    return tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )

# Create the agent
agent = create_agent(
    model=gemini_flash_latest,
    tools=[internet_search, get_current_time, get_pdf_info]
)

# ============================================================
# SPECIFY PDF PATH HERE
# ============================================================
# Option 1: PDF in the same directory as your script
pdf_path = "student_scores.pdf"

# Option 2: PDF in a specific folder
# pdf_path = "data/student_scores.pdf"

# Option 3: Absolute path (Windows)
# pdf_path = r"C:\Users\YourUsername\Documents\student_scores.pdf"

# Option 4: Absolute path (Mac/Linux)
# pdf_path = "/home/username/documents/student_scores.pdf"

# ============================================================

# FIXED: Typo "Summaize" -> "Summarize"
result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": f"Summarize the {pdf_path}"  # Using f-string to insert path
    }]
})

# Print the final response
print("\n" + "="*60)
print("AGENT RESPONSE:")
print("="*60)
print(result["messages"][-1].content)

# Optional: Print all messages to see agent's thinking process
# print("\n" + "="*60)
# print("ALL MESSAGES (including agent reasoning):")
# print("="*60)
# for msg in result["messages"]:
#     print(f"\n{msg.type}: {msg.content}")

# ============================================================
# DIRECT TOOL TESTING (uncomment to test tools directly)
# ============================================================
# Test 1: Metadata only
# print(get_pdf_info.invoke({"file_path": "student_scores.pdf", "summarize": False}))

# Test 2: Full summary
# print(get_pdf_info.invoke({"file_path": "student_scores.pdf", "summarize": True}))