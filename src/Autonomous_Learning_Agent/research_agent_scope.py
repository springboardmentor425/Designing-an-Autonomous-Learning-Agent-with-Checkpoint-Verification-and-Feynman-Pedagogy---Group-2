from datetime import datetime
from email import message
from urllib import response
from langchain_core import messages
from typing_extensions import Literal
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage,AIMessage,get_buffer_string
from langgraph.graph import StateGraph,START,END
from langgraph.types import Command
from Autonomous_Learning_Agent.prompts import clarify_with_user_instructions,transform_messages_into_research_topic_prompt

def get_today_str() -> str:
    """Get current date in human readable format"""
    return datetime.now().strftime("%a %b %#d,%Y")

model=init_chat_model("google_genai:models/gemini-flash-lite-latest")

def clarify_with_user(state: AgentState) -> Command[Literal["write_research_brief", "__end__"]]:
    """
    Determine if the user's request contains sufficient information to proceed with research.

    Uses structured output to make deterministic decisions and avoid hallucination.
    Routes to either research brief generation or ends with a clarification question.
    """
    # Set up structured output model
    structured_output_model = model.with_structured_output(ClarifyWithUser)

    # Invoke the model with clarification instructions
    response = structured_output_model.invoke([
        HumanMessage(content=clarify_with_user_instructions.format(
            messages=get_buffer_string(messages=state["messages"]), 
            date=get_today_str()
        ))
    ])

    # Route based on clarification need
    if response.need_clarification:
        return Command(
            goto=END, 
            update={"messages": [AIMessage(content=response.question)]}
        )
    else:
        return Command(
            goto="write_research_brief", 
            update={"messages": [AIMessage(content=response.verification)]}
        )

def write_research_brief(state: AgentState):
    """
    Transform the conversation history into a comprehensive research brief.

    Uses structured output to ensure the brief follows the required format
    and contains all necessary details for effective research.
    """
    # Set up structured output model
    structured_output_model = model.with_structured_output(ResearchQuestion)

    # Generate research brief from conversation history
    response =structured_output_model.invoke([
        HumanMessage(content=transform_messages_into_research_topic_prompt.format(
            messages=get_buffer_string(state.get("messages", [])),
            date=get_today_str()
        ))
    ])

    # Update state with generated research brief and pass it to the supervisor
    return {
        "research_brief": response.research_brief,
        "supervisor_messages": [HumanMessage(content=f"{response.research_brief}.")]
    }
deep_research_builder=StateGraph(AgentState,input_schema=AgentInputState)
deep_research_builder.add_node("clarify_with_user",clarify_with_user)
deep_research_builder.add_node("write_research_brief",write_research_brief)

deep_research_builder.add_edge(START,"clarify_with_user")
deep_research_builder.add_edge("write_research_brief",END)
# scope_research=deep_research_builder.compile()
