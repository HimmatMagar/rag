import os
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint


load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id='deepseek-ai/DeepSeek-R1',
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_ACCESS_TOKEN"),
    task='text-generation',
    temperature=0.5,
    max_new_tokens=50
)


model = ChatHuggingFace(llm=llm)

response = model.invoke("what is machine learning")
print(response.content)