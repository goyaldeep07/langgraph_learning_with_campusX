from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai.chat_models import ChatOpenAI

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages  # reducer
from dotenv import load_dotenv
import sqlite3


load_dotenv()
model = ChatOpenAI()


class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


def chat_node(state: ChatState):
    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}


conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)

graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

# *************************** Testing code ****************************************
# thread_id = 1
# CONFIG = {"configurable": {"thread_id": f"thread-{thread_id}"}}


# response = chatbot.invoke({
# "messages": [HumanMessage(content="what is my name")]}, config=CONFIG)
# print(response)'
def retrieve_all_threads():
    all_threads = dict()
    for checkpoint in checkpointer.list(None):
        thread_id = checkpoint.config["configurable"]["thread_id"]
        if thread_id not in all_threads:
            all_threads[thread_id] = {
                "id": checkpoint.config["configurable"]["thread_id"],
                "title": checkpoint.checkpoint["channel_values"]["messages"][0].content[
                    :30
                ],
            }
    # return as list of dicts with default titles
    return [all_threads[thread] for thread in all_threads]
