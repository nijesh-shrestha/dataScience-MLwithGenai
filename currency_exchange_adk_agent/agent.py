import os
from dotenv import load_dotenv
import requests
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

load_dotenv()

groq_key = os.getenv("GROQ_API_KEY")

def get_currency_exchange_rate(base:str, target:str) -> dict:
    """Get current exchange rate of currencies.

    Args:
        base (str): Base currency code (e.g. USD, EUR, NPR).
        target (str): Target currency code (e.g. NPR, INR, GBP).

    Returns:
        dict: Current exchange rate information.
    """

    try:
        url = (
            f"https://api.frankfurter.dev/v2/rates?base={base}&quotes={target}"
        )
        response = requests.get(url)
        data = response.json()

        if not data:
            return {"error": "Exchange rate data not available for the provided countries"}

        exchange = data[0]

        return {
            "base": exchange["base"],
            "target": exchange["quote"],
            "date": exchange["date"],
            "rate": exchange["rate"]
        }
    except Exception as e:
        return {"message": "An error occurred while fetching exchange rate data", "error": str(e)}
    

SYSTEM_PROMPT = """You are a helpful assistant that answers questions about currency exchange rate.
When calling a tool, always provide valid, complete JSON arguments with matching braces and quotes.
If the user asks about exchange rate for specific countries, use the get_currency_exchange_rate tool with ISO currency codes such as USD, EUR, NPR, INR.
Anything not related to currency exchange rate should be answered from your own knowledge."""


root_agent = Agent (
    name = "exchange_rate_agent",
    model = LiteLlm(model="groq/llama-3.1-8b-instant"),
    description="Answers the user query related to currency exchange rate",
    instruction=SYSTEM_PROMPT,
    tools=[get_currency_exchange_rate]
)