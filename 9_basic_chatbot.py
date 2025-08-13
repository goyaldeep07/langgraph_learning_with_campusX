from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai.chat_models import ChatOpenAI
from dotenv import load_dotenv
from pydantic import Field
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver


load_dotenv()
model = ChatOpenAI()


class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


def chat_node(state: ChatState):
    messages = state["messages"]
    response = model.invoke(messages)

    return {"messages": [response]}


checkpointer = MemorySaver()
graph = StateGraph(ChatState)


graph.add_node("chat_node", chat_node)

graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

thread_id = 1
while 1:
    user_msg = input("Type here: ")
    print(f"User: {user_msg}")

    if user_msg.strip().lower() in ["exit", "quit", "bye"]:
        break
    config = {"configurable": {"thread_id": thread_id}}
    response = chatbot.invoke({"messages": [HumanMessage(user_msg)]}, config=config)
    print(f"AI: {response["messages"][-1].content}")
