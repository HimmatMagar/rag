from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


loader = PyPDFLoader("docs.pdf")

docs = loader.load()
"""
Lenght Based Text Splitting

Advantages: Fast, Simple
Disadvantages: Semantic meaning, grammer, structure
"""


# split = CharacterTextSplitter(
#     chunk_size=40,
#     chunk_overlap=0,
#     separator="\n"
# )

# char = split.split_documents(docs)
# print(char[0])



"""
Recursive Character Text Splitter

It follows a structure: \n\n -> Paragraph
\n -> line
-  -> words
"" -> character
"""

text = """""
Machine learning (ML) is a field of study in artificial intelligence concerned
with the development and study of statistical algorithms that can learn from data
and generalize to unseen data, and thus perform tasks without being explicitly programmed.
Advances in the field of deep learning have allowed neural networks, a class of statistical algorithms,
to surpass many previous machine learning approaches in performance.

Statistics and mathematical optimisation methods compose the foundations
of machine learning. Data mining is a related field of study, focusing on exploratory
data analysis (EDA) through unsupervised learning.[3][4]
"""""

split = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=0
)

chunk = split.split_text(text)
print(chunk)