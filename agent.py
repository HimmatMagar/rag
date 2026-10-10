import os
import requests
from dotenv import load_dotenv
from tavily import TavilyClient
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

load_dotenv()


## Creating tools
@tool("get weather")
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


@tool("get news")
def get_news(city: str) -> str:
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
model = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", max_output_tokens=60)

tools = {
    "get_weather": get_weather,
    "get_news": get_news
}

llm_with_tools = model.bind_tools([get_weather, get_news])


message = []

print("City Expert chat bot")
print("type quit to end conversations")

while True:
    user_input = input("User: ")
    if user_input.lower() == 'quit':
        break
    message.append(HumanMessage(user_input))
