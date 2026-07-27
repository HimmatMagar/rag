from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


loader = PyPDFLoader("docs.pdf")

docs = loader.load()
"""
Lenght Based Text Splitting

Advantages: Fast, Simple
Disadvantages: Semantic meaning, grammer, structure
"""


split = CharacterTextSplitter(
    chunk_size=40,
    chunk_overlap=0,
    separator="\n"
)

char = split.split_documents(docs)
print(char[0])