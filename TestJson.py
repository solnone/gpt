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
from dotenv import load_dotenv
import json

llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
k = 0 # keep the last {k} interactions in memory
memory = ConversationBufferWindowMemory(k = k, return_messages = True)    
        
def create_chat(state):
    system_message = f"""
`no explanations`
`no prompt`
You are a wise steward, The current environment is represented in JSON: 
{{{json.dumps(state)}}}
Please check the environment to update JSON.
Format your response as a JSON object with "msg", "light", "door" keys.
"""        
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_message),
        MessagesPlaceholder(variable_name = "history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ])
    conversation = ConversationChain(
        memory = memory, prompt = prompt, llm = llm, verbose = True
    )
    return conversation

def get_state(response):
    state = json.loads(response[response.find("{"):response.find("}") + 1])
    # state = json.loads(response)
    return state

def main():
    load_dotenv()

    state = {
        "light": 0,
        "door": 0, 
        "msg": "The light is off, the door is closed"
    }
    
    while True:
        print(json.dumps(state))
        user_input = input("> ")
        if len(user_input) == 0:
            continue
        
        conversation = create_chat(state)  
        response = conversation.predict(input=user_input)
        print(f"Assistant: {response}\n")
        try:
            state = get_state(response)
        except Exception as e:
            print(e)

if __name__ == '__main__':
    main()