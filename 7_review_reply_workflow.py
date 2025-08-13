import operator
from typing import TypedDict, Annotated, List, Literal, Dict
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langchain_openai.chat_models import ChatOpenAI
from pydantic import BaseModel, Field


load_dotenv()
model = ChatOpenAI(model="gpt-4o-mini")


class SentimentSchema(BaseModel):
    """
    Pydantic model defining the expected structured output for sentiment analysis.

    Attributes:
        sentiment (str): The sentiment label of the text.
                         Must be either 'Positive' or 'Negative'.
    """

    sentiment: Annotated[str, Field(Literal["Positive", "Negative"])]


class DiagnosisSchema(BaseModel):
    issue_type: Literal["UX", "Performance", "Bug", "Support", "Other"] = Field(
        description="The category of issue mentioned in the review"
    )
    tone: Literal["angry", "frustrated", "disappointed", "calm"] = Field(
        description="The emotional tone expressed by the user"
    )
    urgency: Literal["low", "medium", "high"] = Field(
        description="How urgent or critical the issue appears to be"
    )


model_with_structured_output = model.with_structured_output(SentimentSchema)
diagnosis_model_with_structured_output = model.with_structured_output(DiagnosisSchema)


class ReviewState(TypedDict):
    """
    Defines the state dictionary structure passed between workflow nodes.

    Attributes:
        review (str): The product review text to analyze.
        sentiment (str): The sentiment of the review ('Positive' or 'Negative').
        diagnosis (dict): Any additional diagnostic info for analysis.
        response (str): A response message or processed output.
    """

    review: Annotated[str, Field(description="Customer review text")]
    sentiment: Annotated[str, Field(Literal["Positive", "Negative"])]
    diagnosis: dict
    response: str


def find_sentiment(state: ReviewState):
    """
    Node function to determine the sentiment of a given review.

    Args:
        state (ReviewState): The current state of the workflow containing the 'review' text.

    Returns:
        dict: A dictionary with the detected sentiment as {'sentiment': value}.
    """
    prompt = f'For the following review find out the sentiment \n {state["review"]}'
    sentiment = model_with_structured_output.invoke(prompt).sentiment
    return {"sentiment": sentiment}


def positive_response(state: ReviewState):
    prompt = f"""Write a warm thank-you message in response to this review:
    \n\n\"{state['review']}\"\n
    Also, kindly ask the user to leave feedback on our website."""

    response = model.invoke(prompt).content

    return {"response": response}


def run_diagnosis(state: ReviewState):
    prompt = f"""Diagnose this negative review:\n\n{state['review']}\n"
    "Return issue_type, tone, and urgency.
"""
    response = diagnosis_model_with_structured_output.invoke(prompt)
    return {"diagnosis": response.model_dump()}


def negative_response(state: ReviewState):
    diagnosis = state["diagnosis"]

    prompt = f"""You are a support assistant.
    The user had a '{diagnosis['issue_type']}' issue, sounded '{diagnosis['tone']}', and marked urgency as '{diagnosis['urgency']}'.
    Write an empathetic, helpful resolution message.
    """
    response = model.invoke(prompt).content

    return {"response": response}


def check_sentiment(
    state: ReviewState,
) -> Literal["positive_response", "run_diagnosis"]:
    if str(state["sentiment"]).lower() == "positive":
        return "positive_response"
    else:
        return "run_diagnosis"


# Build a StateGraph workflow
graph = StateGraph(ReviewState)

# Add the sentiment analysis node to the graph
graph.add_node("find_sentiment", find_sentiment)
graph.add_node("positive_response", positive_response)
graph.add_node("run_diagnosis", run_diagnosis)
graph.add_node("negative_response", negative_response)

# Define the graph edges (execution order)
graph.add_edge(START, "find_sentiment")
graph.add_conditional_edges("find_sentiment", check_sentiment)

graph.add_edge("positive_response", END)

graph.add_edge("run_diagnosis", "negative_response")
graph.add_edge("negative_response", END)


# Compile the workflow
workflow = graph.compile()

# Initial input to the workflow
initial_state = {
    "review": "I’ve been trying to log in for over an hour now, and the app keeps freezing on the authentication screen. I even tried reinstalling it, but no luck. This kind of bug is unacceptable, especially when it affects basic functionality."
}

# Run the workflow
final_state = workflow.invoke(initial_state)

# Output the result
print(final_state)
