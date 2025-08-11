"""Prompt chaining: START -> generate_outline -> generate_blog -> END"""

from typing import TypedDict
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langchain_openai.chat_models import ChatOpenAI


load_dotenv()
model = ChatOpenAI()


class BlogState(TypedDict):
    title: str
    outline: str
    content: str


def create_outline(state: BlogState) -> BlogState:
    title = state["title"]

    prompt = f"generate a detailed outline for the blog on the topic: {title}"
    state["outline"] = model.invoke(prompt).content
    return state


def create_blog(state: BlogState) -> BlogState:
    title = state["title"]
    outline = state["outline"]
    prompt = f"write a detailed blog on the title: {title} using the following outline: {outline}"
    state["content"] = model.invoke(prompt).content
    return state


graph = StateGraph(BlogState)

graph.add_node("create_outline", create_outline)
graph.add_node("create_blog", create_blog)

graph.add_edge(START, "create_outline")
graph.add_edge("create_outline", "create_blog")
graph.add_edge("create_blog", END)

workflow = graph.compile()

initial_state = {"title": "rise of ai in india"}

final_state = workflow.invoke(initial_state)
print(final_state["title"])
print(final_state["outline"])
print(final_state["content"])
