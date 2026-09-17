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

print('1: for angry mode')
print('2: for a very helpful ai teacher')

choice = int(input("Enter choice: "))
if choice == 1:
    mode = "You are the angry ai"
elif choice == 2:
    mode = "You are the very helpful ai teacher to students"

message = [
    SystemMessage(content=mode)
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

