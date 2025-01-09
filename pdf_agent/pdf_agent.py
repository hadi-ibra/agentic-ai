import typer
from rich.prompt import Prompt
from typing import Optional
from phi.model.groq import Groq
from phi.agent import Agent
from phi.knowledge.pdf import PDFUrlKnowledgeBase
from phi.vectordb.chroma import ChromaDb
import phi
import os
from dotenv import load_dotenv

load_dotenv()

phi.api = os.getenv("PHI_API_KEY")

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

knowledge_base = PDFUrlKnowledgeBase(
    urls=["https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    vector_db=ChromaDb(collection="recipes"),
)

# Comment out after first run
knowledge_base.load(recreate=False)


def pdf_agent(user: str = "user"):
    run_id: Optional[str] = None

    agent = Agent(
        model=Groq(id="llama3-70b-8192"),
        run_id=run_id,
        user_id=user,
        knowledge_base=knowledge_base,
        use_tools=True,
        search_knowledge=True,
        show_tool_calls=True,
        debug_mode=True,
    )
    if run_id is None:
        run_id = agent.run_id
        print(f"Started Run: {run_id}\n")
    else:
        print(f"Continuing Run: {run_id}\n")

    agent.cli_app(markdown=True)


if __name__ == "__main__":
    typer.run(pdf_agent)
