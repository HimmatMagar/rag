from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

# Prompt
prompt = PromptTemplate(
    template="Generate 5 interesting fact about {topic}",
    input_variables=['topic']
)

# ## LLM
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

# ## Plain output
parser = StrOutputParser()


# ### Simple Chain
chain = prompt | model | parser

response = chain.invoke({'topic': "nepal"})
print(response)

chain.get_graph().print_ascii()


### Sequential chain
prompt1 = PromptTemplate(
    template="Generate a detail report about {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Generate 5 pointer summary from the following {text}",
    input_variables=['text']
)

chain = prompt1 | model | parser | prompt2 | model | parser

response = chain.invoke({'topic': "education in nepal"})
print(response)