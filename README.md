# Scoping Agent – Clarification-First AI

##  Overview

This project implements a **Scoping Agent** whose only responsibility is to  
**clarify the user’s intent before answering**.

Instead of directly responding to vague or broad questions, the agent:
- Analyzes the user input
- Detects ambiguity
- Asks targeted clarification questions
- Confirms scope before proceeding

This follows a **clarification-first design pattern**, inspired by  
**Feynman Pedagogy** and modern AI safety principles.

---

##  What This Scoping Agent Does

 Accepts raw user input  
 Detects unclear or broad questions  
 Asks clarification questions  
 Maintains conversation state  
 Stops once the scope is clear  

 Does **NOT**:
- Generate final answers
- Perform web search
- Do research or RAG
- Teach content directly

---


---

##  Tech Stack

- **Python 3.11**
- **LangChain**
- **LangGraph**
- **Google Gemini (LLM)**
- **Jupyter Notebook**
- **Rich** (formatted prompt visualization)

---



)

