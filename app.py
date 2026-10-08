import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableParallel, RunnableSequence

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", max_output_tokens=50)

short_prompt = PromptTemplate.from_template("Explain the {topic} in 1 short sentence")
long_prompt = PromptTemplate.from_template("Explain the {topic} in 2 short sentence")

output = StrOutputParser()


chain = RunnableParallel({
    "short": RunnableSequence(short_prompt, model, output),
    "long": long_prompt | model | output
})
result = chain.invoke("Machine learning")

print(result)