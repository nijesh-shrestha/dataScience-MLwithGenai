import os
from dotenv import load_dotenv
import requests
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

load_dotenv()

groq_key = os.getenv("GROQ_API_KEY")

def get_weather(lat:float, lng:float) -> dict:
    """Get current weather and next-5-hour forecast for a location.

    Args:
        lat (float): Latitude of the city.
        lng (float): Longitude of the city.

    Returns:
        dict: A dictionary containing the current weather information and hourly projection for next 5 hours.
    """

    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lng}"
            f"&current_weather=true"
            f"&hourly=temperature_2m,apparent_temperature,relative_humidity_2m,windspeed_10m,rain"
        )
        response = requests.get(url)
        data = response.json()
        current_weather = data.get("current_weather", {})
        hourly_data = data.get("hourly", {})

        if not current_weather:
            return {"error": "Weather data not available for the provided coordinates/city"}

        result = {
            "current_weather": current_weather,
            "next_5_hours": [
                {
                    "time": hourly_data["time"][i],
                    "temperature_2m": hourly_data["temperature_2m"][i],
                    "apparent_temperature": hourly_data["apparent_temperature"][i],
                    "relative_humidity_2m": hourly_data["relative_humidity_2m"][i],
                    "windspeed_10m": hourly_data["windspeed_10m"][i],
                    "rain": hourly_data["rain"][i]
                }
                for i in range(5)
            ]
        }

        return result
    except Exception as e:
        return {"message": "An error occurred while fetching weather data", "error": str(e)}
    

SYSTEM_PROMPT = """You are a helpful assistant that answers questions about weather.
When calling a tool, always provide valid, complete JSON arguments with matching braces and quotes.
If the user asks about weather in a specific city, use the get_weather tool with its latitude and longitude.
Anything not related to weather should be answered from your own knowledge."""


root_agent = Agent (
    name = "weather_agent",
    model = LiteLlm(model="groq/llama-3.1-8b-instant"),
    description="Answers the user query related to weather",
    instruction=SYSTEM_PROMPT,
    tools=[get_weather]
)