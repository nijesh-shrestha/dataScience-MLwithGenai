"""
We're going to make an agent that can tell us about weather in different cities

for example: user: "What is the temperature in Sarlahi today?"

AI: cannot answer this because it does not know today's information of city Sarlahi.

so we wireup a tool that llm can decide to call with required args
and we as a client of the llm, call that function with the args provided by the llm
and attach the response to the llm

First we need open api to get the weather information of different cities. 
We will use open-meteo.com for this purpose.

from this api we make a function tool named get_weather that takes lat and lng of a city and
returns the weather information of that city.
"""

import requests
from dotenv import load_dotenv
import os
from groq import Groq
import streamlit as st
import json

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def get_weather(lat, lng):
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
            return json.dumps({"error": "Weather data not available for the provided coordinates/city"})

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

        return json.dumps(result, indent=4)
    except Exception as e:
        return json.dumps({"message": "An error occurred while fetching weather data", "error": str(e)})
    
def get_weather_tool_properties():
    # This function returns the properties of the get_weather tool in OpenAPI format.
    # This is useful for the LLM to understand how to call the tool, what is the purpose 
    # and what are the required parameters.
    return {
        "type": "function",
        "function":{
            "name": "get_weather",
            "description": """Get the weather information of a city using the city's latitude and longitude. 
            it provides current weather and next 5 hours weather information.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "lat": {
                        "type": "number",
                        "description": "Latitude of the city"
                    },
                    "lng": {
                        "type": "number",
                        "description": "Longitude of the city"
                    }
                },
                "required": ["lat", "lng"]
            }
        }
    }


def run_agent_with_tool(user_query):
    system_prompt = {
        "role": "system",
        "content": """You are a helpful assistant that can answer questions about the weather in different cities.
        If the user asks about the weather in a specific city, you should use relevant tools to get the information. 
        Anything not related to weather information should be answered based on your own knowledge."""
}
    user_prompt = {
        "role": "user",
        "content": user_query
    }

    tools = [get_weather_tool_properties()]

    available_functions = {
        "get_weather": get_weather
    }

    messages = [system_prompt, user_prompt]

    llm_response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        tools=tools,
        max_tokens=1500,
        temperature=0.7,
        tool_choice='auto'
    )
    
    llm_answer = llm_response.choices[0].message
    tool_call_decisions = llm_answer.tool_calls
    if tool_call_decisions:
        for tool_call in tool_call_decisions:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            print(f"Calling function: {function_name} with arguments: {function_args}")
            
            tool_response = available_functions[function_name](**function_args)
            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": tool_response
            })
            llm_response = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=messages,
                max_tokens=1500,
                temperature=0.7
            )
            result = llm_response.choices[0].message.content
    else:
        result = llm_answer.content
    
    return result

st.title("Weather Information Agent")
st.subheader("Ask me about the weather in different cities")
user_query = st.text_input("Ask a question about the weather:")
send_btn = st.button("Send")

if send_btn and user_query:
    result = run_agent_with_tool(user_query)
    st.write(result)