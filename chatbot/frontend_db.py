"""streamlit run chatbot\frontend.py"""

import streamlit as st
from backend_db import chatbot, retrieve_all_threads
from langchain_core.messages import HumanMessage
import uuid


# **************************************** utility functions *************************
def generate_thread_id():
    thread_id = uuid.uuid4()
    return thread_id


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(st.session_state["thread_id"], title="New Chat")
    st.session_state["message_history"] = []


def add_thread(thread_id, title="New Chat"):
    if thread_id not in [t["id"] for t in st.session_state["chat_threads"]]:
        st.session_state["chat_threads"].append({"id": thread_id, "title": title})


def load_conversation(thread_id):
    return chatbot.get_state(
        config={"configurable": {"thread_id": thread_id}}
    ).values.get("messages")


# **************************************** Session Setup ******************************

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    all_threads = retrieve_all_threads()
    st.session_state["chat_threads"] = all_threads

add_thread(st.session_state["thread_id"])

# **************************************** Sidebar UI *********************************
st.sidebar.title("Chatbot")
if st.sidebar.button("New Chat"):
    reset_chat()
st.sidebar.header("My Conversations")

for idx, thread in enumerate(reversed(st.session_state["chat_threads"])):
    if st.sidebar.button(thread["title"], key=f"btn_{idx}"):
        st.session_state["thread_id"] = thread["id"]
        messages = load_conversation(thread_id=thread["id"])

        temp_msgs = []
        if messages:
            for msg in messages:
                role = "user" if isinstance(msg, HumanMessage) else "assistant"
                temp_msgs.append({"role": role, "content": msg.content})
        st.session_state["message_history"] = temp_msgs

# **************************************** Main UI ************************************
# loading the conversation history
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])

user_input = st.chat_input("Type here")

if user_input:
    for thread in st.session_state["chat_threads"]:
        if (
            thread["id"] == st.session_state["thread_id"]
            and thread["title"] == "New Chat"
        ):
            thread["title"] = user_input[:30] + ("..." if len(user_input) > 30 else "")
    # add the message to message_history
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar=None):
        st.text(user_input)

    CONFIG = {"configurable": {"thread_id": st.session_state["thread_id"]}}
    # add the message to message_history
    with st.chat_message("assistant"):
        ai_message = st.write_stream(
            message_chunk.content
            for message_chunk, metadata in chatbot.stream(
                input={"messages": HumanMessage(user_input)},
                config=CONFIG,
                stream_mode="messages",
            )
        )
    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )
