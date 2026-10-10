from rich import print
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

load_dotenv()

@tool
def get_length(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {len(query)}"

tools = {
    "get_length": get_length
}

model = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", max_output_tokens=60)

# ### Binding the tool
tool_model = model.bind_tools([get_length])

message = []

prompt = input("user: ")
query = HumanMessage(prompt)
message.append(query)

result = tool_model.invoke(message)
message.append(result)

if result.tool_calls:
    tool_name = result.tool_calls[0]['name']
    tool_result = tools[tool_name].invoke(result.tool_calls[0])
    message.append(tool_result)

response = model.invoke(message)
print(response)