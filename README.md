# Designing an Autonomous Learning Agent with Checkpoint Verification and Feynman Pedagogy

## 📌 Project Overview
This project focuses on building an **Autonomous AI Learning Agent** using the **LangGraph** framework. Unlike standard chatbots, this agent provides a personalized, structured tutoring experience. It guides learners through sequential goals, strictly enforcing mastery before allowing progression.

The core innovation lies in the **Feynman Pedagogy Module**: when a learner fails an assessment, the agent adapts by switching to a "Feynman Mode," re-explaining complex concepts using simpler analogies and fundamental language.

## 🚀 Key Features
* **Structured Learning Pathways:** Guides users through specific "Checkpoints" rather than open-ended conversation.
* **Checkpoint Verification:** Quantitative assessment at each stage (e.g., 70% pass threshold) to ensure true understanding.
* **Adaptive Simplification:** Automatically triggers the **Feynman Technique** to simplify explanations if the user struggles.
* **Hybrid Knowledge Engine:** Combines user-uploaded notes (RAG) with real-time **Web Search** to generate dynamic teaching content.

## 🛠️ Tech Stack
* **Frameworks:** LangGraph, LangChain
* **LLM:** Google Gemini 1.5 Flash
* **Vector Database:** ChromaDB (for RAG)
* **Search Tool:** Tavily API
* **Language:** Python 3.11+

## ✅ Completed Tasks & Milestones

### 1. Tool Integration & Environment Setup
* Configured **Google Gemini API** for the intelligence core.
* Integrated **Tavily API** for autonomous web searching capabilities.
* Implemented **LangGraph** state management to handle conversation history and context.

### 2. Retrieval-Augmented Generation (RAG) System
* Built a context-aware pipeline using **ChromaDB** and **GoogleGenerativeAIEmbeddings**.
* Implemented `PyPDFLoader` to ingest academic PDFs and project reports.
* Successfully connected the LLM to local documents, allowing the agent to answer questions based *strictly* on uploaded files (reducing hallucinations).

### 3. Deterministic "No-LLM" Web Search Agent
* Designed a rule-based architecture to demonstrate LangGraph's control flow without an LLM.
* **Workflow:**
    1.  **Gather Content Node:** Checks local state for existing information.
    2.  **Logic Gate:** Automatically routes to search if information is missing.
    3.  **Web Search Node:** Fetches real-time data using Tavily and loops back to the start.
* *Achievement:* Proved the ability to build deterministic, loop-based agents alongside AI-driven ones.

### 4. Active Listening & Clarification Agent
* Engineered a **System Prompt** mechanism that forces the agent to ask clarifying questions (e.g., "What is your specific goal?") before answering broad queries.
* Enabled **MemorySaver** (Checkpointer) to retain context across the clarification loop.

## ⚙️ Installation & Setup

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/yourusername/autonomous-learning-agent.git](https://github.com/yourusername/autonomous-learning-agent.git)
    cd autonomous-learning-agent
    ```

2.  **Create Virtual Environment**
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure API Keys**
    Create a `.env` file in the root directory:
    ```env
    GOOGLE_API_KEY=your_google_key
    TAVILY_API_KEY=your_tavily_key
    ```

5.  **Run the Agent**
    ```bash
    python LLM_Agent_Final.py
    ```

## 📄 License
This project is developed for educational and research purposes.