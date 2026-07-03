import os
import httpx
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()
client=httpx.Client(verify=False)
llm = ChatOpenAI(
    base_url="https://genailab.tcs.in/",
    model="genailab-maas-gpt-35-turbo",
    api_key="sk-ZalSmsmKK5U9yBHG08zVHQ",
    http_client=client
)
print(llm.invoke("Hello, how are you?"))

