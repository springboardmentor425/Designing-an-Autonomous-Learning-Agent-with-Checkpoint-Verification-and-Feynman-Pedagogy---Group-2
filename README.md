## LangGraph Intent Clarification Agent

This code implements a stateful agent using LangGraph, organized as a directed graph of functional nodes. The agent processes a user-provided topic, determines whether clarification is required, refines the topic, gathers external context, and generates learning checkpoints.

### Graph Structure and Nodes

The agent workflow is defined as a LangGraph `StateGraph` with the following nodes:

* **get_input**
  Initializes the agent state with the user-provided topic.

* **intent_clarification**
  Analyzes the topic to determine whether clarification is required. If the topic is ambiguous or underspecified, it produces a single clarification question.

* **get_user_clarification**
  Pauses execution and collects user input in response to the clarification question, then feeds the response back into the clarification step.

* **get_refine_topic**
  Refines the clarified topic into a precise, search-ready query suitable for information retrieval.

* **gather_context**
  Retrieves recent and relevant web context for the refined topic using the Tavily search API.

* **create_checkpoints**
  Generates a structured learning plan with progressive checkpoints based on the topic and gathered context.

### Project Structure
- my_agent/
  - utils/
    - state.py
    - nodes.py
  - agent.py


- state.py defines the input and agent state schemas used across the graph.
- nodes.py contains the individual node functions responsible for each step in the workflow.
- agent.py defines the LangGraph StateGraph, adds nodes, and specifies the control flow and conditional transitions.

### Control Flow
<p align="center">
  <img src="https://github.com/user-attachments/assets/8cbb9830-230a-4603-84b8-d0e91d9a0cb9" width="638" />
</p>


Execution starts with user input ingestion, followed by intent clarification. If clarification is required, the agent loops until sufficient detail is obtained. Once clarified, the agent proceeds to topic refinement, context gathering, and checkpoint generation before terminating.

