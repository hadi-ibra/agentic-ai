from phi.agent import Agent
from phi.model.groq import Groq
from phi.model.openai import OpenAIChat
import os
import openai
from dotenv import load_dotenv

import phi
from phi.model.groq import Groq
import logging

from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools
from phi.playground import Playground, serve_playground_app


# logging.basicConfig(level=logging.DEBUG)
load_dotenv()
# Log API keys
# logging.debug(f"PHI_API_KEY: {os.getenv('PHI_API_KEY')}")
# logging.debug(f"OPENAI_API_KEY: {os.getenv('OPENAI_API_KEY')}")


## Load OpenAI API Key

phi.api = os.getenv("PHI_API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")

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
    # role="Search the web for financial information",
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


app = Playground(agents=[finance_agent, web_search_agent]).get_app()


if __name__ == "__main__":
    serve_playground_app("playground:app", reload=True)
