from langchain_deepseek import ChatDeepSeek
from dotenv import load_dotenv


load_dotenv()

llm = ChatDeepSeek(
    model="deepseek-chat"
)


response = llm.invoke("Hello")
print(response)