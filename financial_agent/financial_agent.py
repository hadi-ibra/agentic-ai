from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import os
import openai
from dotenv import load_dotenv

import phi
from phi.playground import Playground, serve_playground_app
from phi.model.groq import Groq

## Load OpenAI API Key
load_dotenv()
phi.api = os.getenv("PHI_API_KEY")


## Web Search Agent
web_search_agent = Agent(
    name="Web Search Agent",
    role="Search the web for information",
    model=Groq(id="llama3-70b-8192"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tool_calls=True,
    markdown=True,
)


## Financial Agent
finance_agent = Agent(
    name="Finance AI Agent",
    model=Groq(id="llama3-70b-8192"),
    tools=[
        YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            stock_fundamentals=True,
            company_news=True,
        )
    ],
    instructions=["Use tables to display the data"],
    show_tool_calls=True,
    markdown=True,
)

multi_ai_agent = Agent(
    team=[web_search_agent, finance_agent],
    model=Groq(id="llama3-70b-8192"),
    instructions=["Always include sources", "Use tables to display the data"],
    show_tool_calls=True,
    markdown=True,
)

multi_ai_agent.print_response(
    "Summarize analyst recommendation and share the latest news for NVDA", stream=True
)
