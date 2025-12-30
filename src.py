import os
import warnings
import logging
from dotenv import load_dotenv
from langchain_core._api.deprecation import LangChainDeprecationWarning

warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

load_dotenv()

from tavily import TavilyClient
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FakeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter


llm = ChatGoogleGenerativeAI(
    model="models/gemini-flash-latest",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    max_retries=0
)


memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)


tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

def web_search_tool(query: str) -> str:
    try:
        res = tavily_client.search(query)
        return "\n".join(
            r["content"] for r in res.get("results", []) if r.get("content")
        )[:1500]
    except:
        return ""

def build_pdf_kb(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(docs)

    embeddings = FakeEmbeddings(size=384)
    return FAISS.from_documents(chunks, embeddings)

kb_vectorstore = build_pdf_kb("data/knowledge.pdf")


def kb_search_tool(query: str) -> str:
    docs = kb_vectorstore.similarity_search(query, k=3)
    return "\n".join(d.page_content for d in docs)[:1500]


def final_answer_tool(query: str) -> str:
    kb_context = kb_search_tool(query)
    web_context = web_search_tool(query)

    prompt = f"""
Question: {query}

PDF Knowledge Base:
{kb_context}

Web Search:
{web_context}

Give a short and clear final answer.
"""
    return llm.invoke(prompt).content


tools = [
    Tool("PDFKnowledgeBase", kb_search_tool, "Search PDF knowledge base"),
    Tool("WebSearch", web_search_tool, "Search live web"),
    Tool("FinalAnswer", final_answer_tool, "Generate final answer")
]


REACT_PROMPT = PromptTemplate.from_template(
"""
You are an intelligent agent.

You can use the following tools:
{tools}

Use this format:

Question: the input question
Thought: think what to do
Action: one of [{tool_names}]
Action Input: input to the action
Observation: result
Thought: I now know the final answer
Final Answer: the final answer to the user

Question: {input}
{agent_scratchpad}
"""
)


react_agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=REACT_PROMPT
)

agent_executor = AgentExecutor(
    agent=react_agent,
    tools=tools,
    memory=memory,
    verbose=False,
    handle_parsing_errors=True,
    max_iterations=6
)


question = "Explain Feynman Pedagogy from the PDF"

try:
    result = agent_executor.invoke({"input": question})
    print(result["output"])
except Exception as e:
    print("Error:", e)
