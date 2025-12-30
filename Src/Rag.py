import os
import warnings
from typing import List
from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

def load_pdf(file_path: str) -> List[Document]:

    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        return documents
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return []
    except Exception as e:
        print(f"Error loading PDF: {e}")
        return []

def split_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_documents(documents)

def create_vector_store(documents: List[Document]):

    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        return Chroma.from_documents(documents, embeddings)
    except Exception as e:
        print(f"Error creating vector store: {e}")
        return None

def create_qa_chain(vector_store):

    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-flash-latest",
            temperature=0.1, 
            convert_system_message_to_human=True,
            model_kwargs={
                "max_output_tokens": 8192,  
                "top_k": 10,
                "top_p": 0.95
            }
        )
        
        retriever = vector_store.as_retriever(
            search_type="similarity", 
            search_kwargs={
                "k": 5
            }
        )
        
        prompt = ChatPromptTemplate.from_template(
            """Answer the question based only on the following context:

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
        print(f"Error creating QA chain: {e}")
        return None

def main():

    pdf_path = "Report.pdf"

    documents = load_pdf(pdf_path)
    if not documents:
        return

    split_docs = split_documents(documents)

    vector_store = create_vector_store(split_docs)
    if not vector_store:
        return

    qa_chain = create_qa_chain(vector_store)
    if not qa_chain:
        return

    query = "What is the project title?"
    
    try:
        print(f"Question: {query}\n")
        response = qa_chain.invoke(query)

        print("Answer:", response)
    
    except Exception as e:
        print(f"Error processing query: {e}")

if __name__ == "__main__":
    main()