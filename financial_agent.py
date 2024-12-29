from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import openai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
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
        model=Groq(id="llama-2-70b"),
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
        model=Groq(id="llama-2-70b"),
        tools=[
            YFinanceTools(
                stock_price=True, 
                analyst_recommendations=True, 
                stock_fundamentals=True, 
                company_news=True
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

# Agent Task
try:
    multi_ai_agent.print_response(
        "Summarize analyst recommendation and share the latest news for NVDA", 
        stream=True,
        show_message=True
    )
except Exception as e:
    print(f"Error in agent response: {e}")
