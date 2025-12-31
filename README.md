# Autonomous Learning Agent with LangGraph

Python/LangGraph implementation of an autonomous tutor that guides learners through structured checkpoints, blends user notes with live web search, and uses mastery checks plus Feynman-style simplifications to reinforce understanding before advancing.

## Highlights
- Structured checkpoint graph with explicit objectives and success criteria.
- Dual-source context: prioritize learner-provided notes, fall back to web search when gaps remain.
- Targeted question generation and quantitative scoring against each checkpoint.
- Feynman-based remediation: simplify weak concepts with analogies, then reassess.
- Mastery progression: learners must meet the threshold before moving to the next checkpoint.
- Interactive surface for delivering questions, collecting answers, and presenting explanations.

## Architecture
- **LangGraph state graph:** orchestrates nodes and transitions across the learning loop.
- **Checkpoint definition module:** holds topics, objectives, and pass criteria.
- **Context management module:** ingests user notes, performs web search when needed, validates relevance, and prepares text (chunking/embedding).
- **Question generation module:** creates checkpoint-specific questions from processed context.
- **Understanding verification module:** scores learner responses and enforces the mastery threshold.
- **Feynman teaching module:** produces simplified explanations for weak areas and routes back to reassessment.
- **LLM integration:** powers reasoning, question generation, scoring, and simplification.
- **Learner interface:** front-end or I/O layer enabling direct interaction.

## Learning Workflow
1. Define the active checkpoint and its objectives.
2. Gather context from user notes; use web search if coverage is insufficient.
3. Validate that context meets the checkpoint goals; re-gather if it does not.
4. Process context (chunking/embedding) for retrieval.
5. Generate targeted questions tied to the checkpoint.
6. Assess learner answers and compute an understanding score.
7. If below threshold, generate Feynman-style explanations for weak spots and loop back to reassessment.
8. When the threshold is met, mark the checkpoint complete and advance; finish when the path ends.

## Tech Stack
- Python 3.11+
- LangGraph for stateful orchestration
- LangChain for LLM tooling, prompt management, text splitting, and embeddings
- LLM provider SDKs (e.g., langchain-openai, langchain-google-genai, langchain-anthropic)
- Web search tools (e.g., Tavily, SerpApi, DuckDuckGo via langchain-community)
- Embedding backends and optional vector stores (faiss, chromadb, pinecone-client)
- python-dotenv for environment management
- Jupyter Notebooks for iterative development

## Getting Started
1. Create and activate a virtual environment:
   - `python -m venv venv`
   - On PowerShell: `.\venv\Scripts\Activate.ps1`
2. Install dependencies (adjust providers as needed):
   - `pip install -U pip`
   - `pip install langgraph langchain langchain-community python-dotenv faiss-cpu`  
     Add your chosen LLM and search provider packages, e.g.:
     - `pip install langchain-openai` or `pip install langchain-google-genai`
     - `pip install tavily-python` or another search client
3. Create a `.env` file with your credentials:
   - `OPENAI_API_KEY=...` or `GEMINI_API_KEY=...` or `ANTHROPIC_API_KEY=...`
   - `TAVILY_API_KEY=...` (or other search API key)

## Developing the Agent
- Define your checkpoint syllabus (topics, objectives, pass thresholds).
- Implement LangGraph nodes for context gathering, validation, processing, question generation, scoring, and Feynman remediation.
- Wire the loop so sub-threshold scores route through the Feynman module and then back to reassessment.
- Add a simple interface (CLI, notebook, or web front end) to deliver questions, collect answers, and present explanations.

## Repository Layout
- `notebooks/` — workspace for experimentation and scoping.
- `venv/` — local virtual environment (optional; recreate as needed).
- `LICENSE` — licensing terms.
- `.gitignore` — excludes the virtual environment.

## Testing and Quality
- Add unit tests for scoring logic, question relevance, and graph transitions.
- Use tracing/observability (e.g., LangSmith) to inspect state transitions, search queries, and LLM outputs during development.
