from tavily import TavilyClient
import datetime
from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools import TavilySearchResults
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver 
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import Tool

load_dotenv()

tavily_client = TavilyClient()

date = datetime.datetime.now().strftime("%Y-%m-%d")

def load_pdf(file_path: str) -> List[Document]:
    """Load a PDF file and return its documents."""
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        return documents
    except FileNotFoundError:
        print(f" Error: File {file_path} not found.")
        return []
    except Exception as e:
        print(f" Error loading PDF: {e}")
        return []

def split_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """Split documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    splits = text_splitter.split_documents(documents)
    print(f" Split into {len(splits)} chunks")
    return splits

def create_vector_store(documents: List[Document]):
    """Create a vector store from documents."""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        vectorstore = Chroma.from_documents(documents, embeddings)
        print("Vector store created successfully")
        return vectorstore
    except Exception as e:
        print(f"Error creating vector store: {e}")
        return None

def create_rag_chain(vector_store):
    """Create a RAG question-answering chain."""
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-flash-latest",
            temperature=0.1, 
            convert_system_message_to_human=True,
        )
        
        retriever = vector_store.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": 5}
        )
        
        prompt = ChatPromptTemplate.from_template(
            """Answer the question based only on the following context from the document:

{context}

Question: {question}

Answer:"""
        )
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        return chain
    except Exception as e:
        print(f" Error creating RAG chain: {e}")
        return None

pdf_path = "Report.pdf"
rag_chain = None

documents = load_pdf(pdf_path)
if documents:
    split_docs = split_documents(documents)
    vector_store = create_vector_store(split_docs)
    if vector_store:
        rag_chain = create_rag_chain(vector_store)

llm = ChatGoogleGenerativeAI(
   model="models/gemini-flash-latest",
   temperature=0,
   max_tokens=None,
   timeout=None,
   max_retries=2,
)


search_tool = TavilySearchResults(
  max_results=5,
  include_raw_content=True,
)

tools = [search_tool]

if rag_chain:
    def rag_query(question: str) -> str:
        """Query the PDF document using RAG."""
        try:
            response = rag_chain.invoke(question)
            return response
        except Exception as e:
            return f"Error querying document: {e}"
    
    rag_tool = Tool(
        name="document_search",
        description="Search and query information from the Report.pdf document. Use this when the user asks about information that might be in the uploaded PDF document.",
        func=rag_query
    )
    tools.append(rag_tool)
    print(" RAG tool added to agent")

agent_executor = create_agent(llm, tools, checkpointer=InMemorySaver())

def chat_with_memory():

    config = {"configurable": {"thread_id": "session_1"}}
    print("\nType 'bye' to stop")   
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["bye", "exit"]:
            break
            
        print("Agent is thinking...")
        
        response = agent_executor.invoke(
            {"messages": [("user", user_input)]},
            config=config
        )
        
        print(f"Agent: {response['messages'][-1].content}")


prompt = ChatPromptTemplate.from_messages([
     ("system", "You are a helpful assistant. The current date is {date}. Always use this date for time-sensitive answers."),
     ("human", "{question}"),
 ])

chat_with_memory()