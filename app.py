import os
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings


load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id='deepseek-ai/DeepSeek-R1',
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_ACCESS_TOKEN"),
    task='text-generation',
    temperature=0.5,
    max_new_tokens=50
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector = embeddings.embed_query("What is artificial intelligence?")
print(vector)
print(len(vector))

model = ChatHuggingFace(llm=llm)

response = model.invoke("what is machine learning")
print(response.content)