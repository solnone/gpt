# https://python.langchain.com/en/latest/modules/memory/types/buffer_window.html

from langchain.prompts import (
    ChatPromptTemplate, 
    MessagesPlaceholder, 
    SystemMessagePromptTemplate, 
    HumanMessagePromptTemplate
)
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv
import os

system_message = "You are a helpful assistant."
k=2
conversation_key = "conversation"
human_message_key = "human"

def get_api_key():
    return st.text_input(label="OpenAI API Key ", type="password", placeholder="Ex: sk-2twm...", key="openai_api_key_input")

def getConversation():
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_message),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ])
    
    llm = ChatOpenAI(temperature=0)
    memory = ConversationBufferWindowMemory(k=k, return_messages=True)
    conversation = ConversationChain(
      memory=memory, prompt=prompt, llm=llm, verbose=True
    )
    return conversation

def submit():
    user_input = st.session_state.user_input
    st.session_state.user_input = ''
    if (len(user_input) > 1):
        conversation = st.session_state[conversation_key]
        conversation.predict(input=user_input)
    
def main():
    st.set_page_config(page_title="Conversation Buffer Window Memory", page_icon=":robot:")
    st.title("Conversation")
    st.markdown(f"System Message: {system_message}")
    st.header(f"Buffer Window Memory k={k}")
    
    load_dotenv()
    
    # Load the OpenAI API key from the environment variable
    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        print("OPENAI_API_KEY is not set")
        os.environ["OPENAI_API_KEY"] = get_api_key()

    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        return
    
    placeholder = st.empty()
    
    if conversation_key not in st.session_state:
        st.session_state[conversation_key] = getConversation()

    conversation = st.session_state[conversation_key]
    with placeholder.container():
        for msg in conversation.memory.chat_memory.messages:
            if msg.type == human_message_key:
                message(msg.content, is_user=True)    
            else:
                message(msg.content)                          

    st.text_input(label="Enter your message", placeholder="Send a message", key="user_input", on_change=submit)

if __name__ == '__main__':
    main()
