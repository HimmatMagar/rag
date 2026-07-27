from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter, Language
from langchain_community.document_loaders import PyPDFLoader


loader = PyPDFLoader("docs.pdf")

# docs = loader.load()
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

# text = """""
# Machine learning (ML) is a field of study in artificial intelligence concerned
# with the development and study of statistical algorithms that can learn from data
# and generalize to unseen data, and thus perform tasks without being explicitly programmed.
# Advances in the field of deep learning have allowed neural networks, a class of statistical algorithms,
# to surpass many previous machine learning approaches in performance.

# Statistics and mathematical optimisation methods compose the foundations
# of machine learning. Data mining is a related field of study, focusing on exploratory
# data analysis (EDA) through unsupervised learning.[3][4]
# """""

# split = RecursiveCharacterTextSplitter(
#     chunk_size=150,
#     chunk_overlap=0
# )

# chunk = split.split_text(text)
# print(chunk)


"""
For Documents and Piece of code
"""

docs = """
class Pen:
    
    def __init__(self, name, brand, price):
        self.name = name
        self.brand = brand
        self.price = price
    

    def get_detail(self):
        print(f"Pen name is {self.name} and it's a {self.brand} pen and price is {self.price}")
    
ball_pen = Pen("Gel pen", "Gel", 50)
"""
splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=175,
    chunk_overlap=0
)
chunk = splitter.split_text(docs)
print(chunk[0])