from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def gather_from_notes(checkpoint, notes):
    prompt = PromptTemplate(
        input_variables=["topic", "objectives", "notes"],
        template="""
        Topic: {topic}
        Learning Objectives: {objectives}

        User Notes:
        {notes}

        Extract only the information relevant to the objectives.
        """
    )
    return llm.predict(
        prompt.format(
            topic=checkpoint.topic,
            objectives=", ".join(checkpoint.objectives),
            notes=notes
        )
    )
