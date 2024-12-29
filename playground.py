from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import openai
import os
from dotenv import load_dotenv
import phi
from phi.playground import Playground
from fastapi import FastAPI
from uvicorn import run



# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
phi.api=os.getenv("PHI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables.")
openai.api_key = api_key

# Debugging: Verify API key
print(f"API Key Loaded: {openai.api_key[:5]}...")

# Web Search Agent
try:
    agent_search = Agent(
        name="Web Search Agent",
        role="Search the web for the information",
        model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
        tools=[DuckDuckGo()],
        instructions=["Always include sources"],
        show_tool_calls=True,
        markdown=True,
    )
except Exception as e:
    print(f"Error initializing Web Search Agent: {e}")

# Financial Tools Agent
try:
    agent_finance = Agent(
        name="Finance AI Agent",
        model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
        tools=[
            YFinanceTools(
                stock_price=True, 
                analyst_recommendations=True, 
                stock_fundamentals=True, 
                company_news=True,
                historical_prices=True,
                company_info=True
            )
        ],
        show_tool_calls=True,
        description="You are an investment analyst that researches stock prices, analyst recommendations, and stock fundamentals.",
        instructions=["Format your response using markdown and use tables to display data where possible."],
        markdown=True,
    )
except Exception as e:
    print(f"Error initializing Finance Agent: {e}")

# Multi-Agent
try:
    multi_ai_agent = Agent(
        team=[agent_search, agent_finance],
        instructions=["Always include sources", "Use tables to display the data"],
        show_tool_calls=True,
        markdown=True,
    )
except Exception as e:
    print(f"Error initializing Multi-Agent: {e}")

# Playground
playground = Playground(agents=[agent_search, agent_finance])
app = playground.get_app()

if __name__=="__main__":
    run("playground:app", host="127.0.0.1", port=8000, reload=True)
