import os
from typing import Literal

from dotenv import load_dotenv
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from tavily import TavilyClient

from langgraph.checkpoint.memory import InMemorySaver  #Memory element

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_groq import ChatGroq
load_dotenv()
#LLM model


model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,)

# Embedding model
embeddings = GoogleGenerativeAIEmbeddings(
    model="text-embedding-004",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

#Vector store
vector_store = InMemoryVectorStore(embeddings)

#Steps:1
#Indexing
file_path = "deeplearningbook-rnn.pdf"
loader = PyPDFLoader(file_path)

docs = loader.load()

#Splitting
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # chunk size (characters)
    chunk_overlap=200,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
)
all_splits = text_splitter.split_documents(docs)

#Storing in vector store
document_ids = vector_store.add_documents(documents=all_splits)

#Steps:2
#Retrieval and Generation
#RAG agent tool
@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs
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

prompt = (
    "You have access to a tool that retrieves context from a blog post and a tool to search the web for information. "
    "Use the tools to help answer user queries."
)

agent=create_agent(
    model,
    tools=[retrieve_context, internet_search],
    system_prompt=prompt,
    checkpointer=InMemorySaver() #Short term memory
)
while True:
    user_input = input("Enter your query (or 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    result=agent.invoke({
        "messages":[{
            "role":"user",
            "content":user_input
        }]},
        {"configurable": {"thread_id": "1"}}
    )
    print(result["messages"][-1].content)
