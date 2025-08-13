from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from langchain_openai.chat_models import ChatOpenAI
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver


load_dotenv()

model = ChatOpenAI()


class JokeState(TypedDict):
    topic: str
    joke: str
    explanation: str


def generate_joke(state: JokeState):
    prompt = f"generate a joke on the topic {state['topic']}"
    response = model.invoke(prompt).content
    return {"joke": response}


def generate_explanation(state: JokeState):
    prompt = f"write an explanation for the joke - {state['joke']}"
    response = model.invoke(prompt).content
    return {"explanation": response}


graph = StateGraph(JokeState)

graph.add_node("generate_joke", generate_joke)
graph.add_node("generate_explanation", generate_explanation)

graph.add_edge(START, "generate_joke")
graph.add_edge("generate_joke", "generate_explanation")
graph.add_edge("generate_explanation", END)

checkpointer = InMemorySaver()

workflow = graph.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "1"}}
final_state = workflow.invoke(input={"topic": "pizza"}, config=config)

print(final_state)
print()

print(workflow.get_state(config=config))
print()

print(list(workflow.get_state_history(config=config)))
