import os
import requests
from pydantic import BaseModel
from dotenv import load_dotenv
from tavily import TavilyClient
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, ToolMessage
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call


load_dotenv()


## Creating tools
@tool
def get_weather(city: str) -> str:
    """Get the weather of city include: temprature, weather situation and humidity"""
    API_KEY = os.getenv("OPENWEATHER_API_KEY")
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city},NP&appid={API_KEY}&units=metric"

    response = requests.get(url=url)
    data = response.json()

    if response.status_code != 200:
        return f"Unable to get the weather of {city}"
    
    temp = data['main']['temp']
    info = data['weather'][0]['description']
    humidity = data['main']['humidity']

    return f"Current weather in {city} are temp: {temp}°C, weather info: {info}, humidity: {humidity}"


@tool
def get_news(city: str) -> str:
    """Get the weather of any city """
    client = TavilyClient(
        api_key=os.getenv("TAVILY_API_KEY")
    )

    response = client.search(
        query=city,
        search_depth='basic',
        max_results=3
    )

    news_result = f"Latest News for {city}\n"
    news_result += "=" * 40 + "\n\n"

    for index, result in enumerate(response["results"], start=1):
        news_result += f"{index}. {result['title']}\n"
        news_result += f"   URL: {result['url']}\n\n"

    return news_result


## Creating agents
model = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")


@wrap_tool_call
def human_approval(request, handler):
    """Ask human before calling any tools"""
    tool_name = request.tool_call['name']
    confirm = input(
        f"Agent wants to use '{tool_name}'. Approve? (yes/no): "
    ).strip().lower()

    if confirm == 'yes':
        return handler(request)
    
    return ToolMessage(
        content=f"User rejected the '{tool_name}' tool call.",
        tool_call_id = request.tool_call['id']
    )

agent = create_agent(
    model=model,
    tools=[get_weather, get_news],
    system_prompt="You are a helpful assistant. Be concise and accurate.",
    middleware=[human_approval]
)

print("AI Agent: Type 'quit' to end conversations")

while True:
    query = input("user: ")
    if query.lower() == "quit":
        print("Thank you for using me...")
        break

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": query
                }
            ]
        } 
    )
    print(result['messages'][-1].text)

