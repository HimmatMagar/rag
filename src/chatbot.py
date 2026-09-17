import os
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage


load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id='deepseek-ai/DeepSeek-R1',
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_ACCESS_TOKEN"),
    task='text-generation',
    max_new_tokens=50
)

message = [
    SystemMessage(content="You are an AI teacher.")
]

model = ChatHuggingFace(llm=llm)

while True:
    prompt = input("User: ")

    if prompt == "0":
        break

    message.append(HumanMessage(content=prompt))

    response = model.invoke(message)

    message.append(AIMessage(content=response.content))

    print("Assistant:", response.content)