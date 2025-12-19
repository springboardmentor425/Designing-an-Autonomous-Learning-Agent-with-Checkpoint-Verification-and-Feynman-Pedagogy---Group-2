import pprint
from langchain_core.messages import AIMessage
import time

import os
os.environ["USER_AGENT"] = "my-langchain-agent/1.0"
from dotenv import load_dotenv

from langchain_community.document_loaders import (
    UnstructuredPowerPointLoader,
    UnstructuredPDFLoader,
    WebBaseLoader,
    YoutubeLoader,
    Docx2txtLoader,
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.utils import filter_complex_metadata


from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from uuid import uuid4

from langchain_core.tools import tool
from tavily import TavilyClient

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent

from langgraph.checkpoint.memory import InMemorySaver  

from langchain.agents.middleware import ModelCallLimitMiddleware

load_dotenv()

# Data ingestion 
def get_loader(source: str):
    if source.startswith("http"):
        if "youtube.com" in source or "youtu.be" in source:
            return YoutubeLoader.from_youtube_url(
                source,
                add_video_info=True
            )
        else:
            return WebBaseLoader(source)
        
    if source.endswith(".pdf"):
        return UnstructuredPDFLoader(
            source,
            mode="elements",
        )

    elif source.endswith(".pptx"):
        return UnstructuredPowerPointLoader(source)

    elif source.endswith(".docx"):
        return Docx2txtLoader(source)

    else:
        raise ValueError("Unsupported source type")

user_input = "./example_data/Selections.pdf"
loader = get_loader(user_input)
docs = loader.load()

# Splitting docs
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(docs)
chunks = filter_complex_metadata(chunks)

print(f"Splited the document into {len(chunks)} sub-documents.\n")
for chunk in chunks:
    chunk.metadata["source"] = os.path.basename(user_input)

# Embedding model
embeddings = GoogleGenerativeAIEmbeddings(
    model="text-embedding-004",
    google_api_key = os.getenv("GEMINI_API_KEY")
)

# Chorma vector store
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)

uuids = [str(uuid4()) for _ in range(len(chunks))]
vector_store.add_documents(documents=chunks, ids=uuids)


@tool
def search_knowledge_base(query: str) -> str:
    """Search Chroma vector store."""
    results = vector_store.similarity_search(
        query,
        k=4,
    )
    if not results:
        return "No relevant documents found."
    
    return "\n\n".join([f"[{doc.metadata.get('source', 'unknown')}] {doc.page_content}" for doc in results])

tavily_client = TavilyClient() 
@tool
def tavily_search(
        query: str,
        max_results: int = 5,
        include_raw_content: bool = False,
):
    """Run a web search using Tavily"""
    return tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
)
    
    
# Google Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0,
)

tools = [search_knowledge_base, tavily_search]

system_prompt = """
You are an intelligent assistant that answers user questions using external tools when required.

Rules you MUST follow strictly:

1. For every user question, FIRST determine whether the answer can be obtained from:
   - the conversation context (memory), or
   - the knowledge base, or
   - requires recent / up-to-date information.

2. If the question requires factual information that is not already present in the conversation context:
   - Call the `search_knowledge_base` tool first.
   - Do NOT answer the question before calling this tool.

3. After receiving the knowledge base results:
   - If the information is sufficient and relevant, answer ONLY using the knowledge base content.
   - If the information is missing, outdated, incomplete, or insufficient,
     THEN call the `tavily_search` tool to retrieve recent or current information.

4. Use the `tavily_search` tool ONLY when:
   - The question explicitly requires recent, latest, or current information, OR
   - The knowledge base does not contain enough information to answer accurately.

5. DO NOT re-run tools for questions that have already been answered in the current conversation,
   unless the user explicitly asks again or requests updated information.

6. Do NOT use any tools when the question can be answered directly from:
   - prior user messages (e.g., the user's name), or
   - the assistant's previous responses.

7. In your final answer:
   - Base your response strictly on the tool outputs or conversation context.
   - Do NOT add information that was not returned by the tools or already present in the conversation.

8. If neither the knowledge base nor the Tavily search returns relevant information:
   - Explicitly state: "The answer is not available in the knowledge base or current web search results."

"""


# Initialize agent
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=system_prompt,
    checkpointer=InMemorySaver(),
    middleware=[
        ModelCallLimitMiddleware(
            thread_limit=10,
            run_limit=3,
            exit_behavior="end",
        ),
    ],
)

# Helper function
def extract_text(msg):
    content = msg.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part["text"] for part in content if part["type"] == "text")
    return str(content)

def print_tool_calls(agent_response):
    for msg in agent_response.get("messages", []):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for call in msg.tool_calls:
                print("-" * 40)
                print("\nTool name:", call.get("name"))
                print("Arguments:", call.get("args"))
                print("Tool call id:", call.get("id"))
                print("-" * 40)
                print("\n")





response_1 = agent.invoke(
    {"messages": [{"role": "user", "content": "Hi my name is Chaitanya!"}]},
    {"configurable": {"thread_id": "1"}}
)
print(extract_text(response_1["messages"][-1]))
print_tool_calls(response_1)
time.sleep(15)

response_2 = agent.invoke(
    {"messages": [{"role": "user", "content": "I have a question about the poem 'Auguries of Innocence', What central ideas about morality and human responsibility does William Blake express in the poem, and how are they conveyed through the poem's imagery?"}]},
    {"configurable": {"thread_id": "2"}}
)

print(extract_text(response_2["messages"][-1]))
print_tool_calls(response_2)
time.sleep(15)

response_3 = agent.invoke(
    {"messages": [{"role": "user", "content": "What is my name?"}]},
    {"configurable": {"thread_id": "1"}}
)
print(extract_text(response_3["messages"][-1]))
print_tool_calls(response_3)
time.sleep(15)

response_4 = agent.invoke(
    {"messages": [{"role": "user", "content": "What happened in the recent match between India and South africa?"}]},
    {"configurable": {"thread_id": "3"}}
)
print(extract_text(response_4["messages"][-1]))
print_tool_calls(response_4)
