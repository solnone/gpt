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
import os

def main():
    load_dotenv()

    # Load the OpenAI API key from the environment variable
    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        print("OPENAI_API_KEY is not set")
        exit(1)
    else:
        print("OPENAI_API_KEY is set")

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template("You are a helpful assistant."),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ])
    llm = ChatOpenAI(temperature=0)
    # memory = ConversationBufferMemory(return_messages=True)
    memory = ConversationBufferWindowMemory( k=3, return_messages=True)
    conversation = ConversationChain(
      memory=memory, prompt=prompt, llm=llm, verbose=True
    )

    print("Hello, I am ChatGPT CLI!")

    while True:
        user_input = input("> ")

        response = conversation.predict(input=user_input)
        # messages.append(HumanMessage(content=user_input))
        # assistant_response = chat(messages)
        # messages.append(AIMessage(content=assistant_response.content))

        print("\nAssistant:\n", response, "\n")


if __name__ == '__main__':
    main()
