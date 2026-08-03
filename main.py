from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek
import os

load_dotenv()
print(os.getenv("DEEPSEEK_API_KEY"))

# llm = ChatOpenAI(temperature=0.4)
# reponsens = llm.invoke("What is the capital city of Nepal?")
# print(reponsens.content)

# llm = HuggingFaceEndpoint(
#     repo_id="meta-llama/Llama-3.1-8B-Instruct",
#     task="text-generation"
# )

# model = ChatHuggingFace(llm=llm)

# result = model.invoke("What is the capital city of Nepal")
# print(result.content)


# llm = ChatDeepSeek(
#     model="deepseek-chat",
#     temperature=0.3
# )
# response = llm.invoke("What is the capital city of nepal")
# print(response)