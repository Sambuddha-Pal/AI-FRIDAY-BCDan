import os
import httpx
import tiktoken

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

load_dotenv()
tiktoken_cache_dir = "tiktoken_cache"

os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir
assert os.path.exists(os.path.join(tiktoken_cache_dir, "9b5ad71b2ce5302211f9c61530b329a4922fc6a4"))
http_client = httpx.Client(
    verify=False
)

llm = ChatOpenAI(

    base_url="https://genailab.tcs.in/",

   model="azure/genailab-maas-gpt-4o",

    api_key=os.getenv("OPEN_AI_API_KEY"),

    http_client=http_client,

    temperature=0

)

embeddings = OpenAIEmbeddings(

    base_url="https://genailab.tcs.in/",

    model="azure/genailab-maas-text-embedding-3-large",

    api_key=os.getenv("OPEN_AI_API_KEY"),

    http_client=http_client,

    

)