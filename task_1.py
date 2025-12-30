import os
from typing import Literal
from dotenv import load_dotenv
from langchain_core.tools import tool
from datetime import datetime
from tavily import TavilyClient
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()

# Tavily client
tavily_client = TavilyClient()

# Gemini model initialization   
gemini_model = ChatGoogleGenerativeAI(
    model="models/gemini-2.0-flash-exp", 
    temperature=0,
    max_tokens=2000,
)
# ============================================================================
# RAG SETUP
# ============================================================================

# Global variable to store vector store
vector_store = None

def setup_rag(documents_path):
    """Setup RAG system by loading documents and creating vector store"""
    global vector_store
    
    print("\n🔧 Setting up RAG system...")
    
    try:
        # Load documents
        loader = DirectoryLoader(
            documents_path,
            glob="**/*.txt",
            loader_cls=TextLoader
        )
        documents = loader.load()
        print(f"✓ Loaded {len(documents)} documents")
        
        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        print(f"✓ Split into {len(chunks)} chunks")
        
        # Create embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Create vector store
        vector_store = FAISS.from_documents(chunks, embeddings)
        vector_store.save_local("faiss_index")
        print("✓ Vector store created successfully!\n")
        
        return True
    except Exception as e:
        print(f"✗ Error setting up RAG: {e}\n")
        return False


def load_existing_rag(index_path="faiss_index"):
    """Load existing RAG vector store"""
    global vector_store
    
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vector_store = FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
        print("✓ Loaded existing RAG system\n")
        return True
    except Exception as e:
        print(f"✗ Could not load existing RAG: {e}\n")
        return False

# ============================================================================
# TOOLS
# ============================================================================

@tool
def get_datetime() -> str:
    """Get the current date and time in a simple format."""
    now = datetime.now()
    return f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')} ({now.strftime('%A')})"


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

@tool
def search_documents(query: str, k: int = 3) -> str:
    """
    Search through local documents using RAG.
    Use this tool to find information from your knowledge base.
    
    Args:
        query: The search query
        k: Number of relevant documents to retrieve (default: 3)
    
    Returns:
        Relevant information from the documents
    """
    global vector_store
    
    if vector_store is None:
        return "RAG system not initialized. Please set up documents first."
    
    try:
        # Perform similarity search
        results = vector_store.similarity_search(query, k=k)
        
        if not results:
            return "No relevant documents found."
        
        # Format results
        response = "Found relevant information from documents:\n\n"
        for i, doc in enumerate(results, 1):
            response += f"Document {i}:\n{doc.page_content}\n\n"
        
        return response
    except Exception as e:
        return f"Error searching documents: {str(e)}"


# ============================================================================
# AGENT SETUP
# ============================================================================

# Create the agent with all tools
agent = create_agent(
    gemini_model,
    tools=[internet_search, get_datetime, search_documents],
    checkpointer=MemorySaver()
)

# ============================================================================
# MAIN CHAT LOOP
# ============================================================================

if __name__ == "__main__":
    
    # Setup RAG (choose one option)
    print("=" * 80)
    print("RAG SETUP OPTIONS:")
    print("=" * 80)
    print("1. Load documents from a directory")
    print("2. Load existing vector store")
    print("3. Skip RAG setup (use only web search and datetime)")
    
    choice = input("\nSelect option (1-3): ")
    
    if choice == "1":
        docs_path = input("Enter documents directory path: ")
        setup_rag(docs_path)
    elif choice == "2":
        index_path = input("Enter vector store path (default: faiss_index): ") or "faiss_index"
        load_existing_rag(index_path)
    else:
        print("Skipping RAG setup\n")
    
    # Start chat
    config = {"configurable": {"thread_id": "1"}}
    chat_history = []
    
    print("=" * 80)
    print("🤖 AI AGENT WITH RAG")
    print("=" * 80)
    print("Commands:")
    print("  - 'history' to view chat history")
    print("  - 'quit' or 'exit' to end session")
    print("=" * 80 + "\n")
    
    while True:
        user_input = input("\n💭 You: ")
        
        if user_input.lower() == 'history':
            print("\n📜 Chat History:")
            for i, entry in enumerate(chat_history, 1):
                print(f"\n{i}. Q: {entry['question']}")
                print(f"   A: {entry['answer'][:100]}...")
            continue
        
        if user_input.lower() in ['quit', 'exit']:
            print(f"\n👋 Goodbye! You asked {len(chat_history)} questions.")
            break
        
        if not user_input.strip():
            continue
        
        try:
            result = agent.invoke({
                "messages": [{"role": "user", "content": user_input}]
            }, config)
            
            answer = result["messages"][-1].content
            chat_history.append({"question": user_input, "answer": answer})
            
            print(f"\n🤖 Agent: {answer}")
            print("-" * 80)
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("-" * 80)
