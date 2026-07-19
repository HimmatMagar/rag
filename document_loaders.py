from langchain_community.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()


'''
Text Loader
'''
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template="Write a short summary about {topic}",
    input_variables=['topic']
)

parser = StrOutputParser()

loader = TextLoader("text.txt")

docs = loader.load()

chain = prompt | model | parser
print(chain.invoke({'topic': docs[0].page_content}))


'''
PDF Loader
'''
pdf_loader = PyPDFLoader(file_path="docs.pdf")

docs = pdf_loader.load()
print(docs[0].metadata)
print(docs[0].page_content)


'''
Directory Loader
'''

loader = DirectoryLoader(
    path='folder_name',
    glob="*.pdf", # all the pdf
    loader_cls=PyPDFLoader
)
loader.lazy_load()