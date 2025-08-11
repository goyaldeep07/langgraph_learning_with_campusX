"""LLM workflow"""

from typing import TypedDict
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langchain_openai.chat_models import ChatOpenAI


load_dotenv()
model = ChatOpenAI()


class LLMState(TypedDict):
    question: str
    answer: str


def llm_qa(state: LLMState) -> LLMState:
    # extract question from the state
    question = state["question"]

    # form a prompt
    prompt = f"answer the following question: {question}"

    # ask that question to the LLM
    answer = model.invoke(prompt).content

    # update the answer in the state
    state["answer"] = answer

    return state


# create graph
graph = StateGraph(LLMState)

# add nodes
graph.add_node("llm_qa", llm_qa)

# add edges
graph.add_edge(START, "llm_qa")
graph.add_edge("llm_qa", END)

# compile graph
workflow = graph.compile()

# invoke graph
initial_state = {
    "question": "yesterday i ate maida naan and a 100gm piece of cheese cake please provide me my today's schedule to neutralize the effect of these on my body i live in delhi, india and this august month and i am 30 years old boy please suggest me according to my details"
}
final_state = workflow.invoke(initial_state)
print(final_state)
