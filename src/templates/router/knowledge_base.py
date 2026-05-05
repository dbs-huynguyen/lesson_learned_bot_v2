import operator
from typing import Annotated, Literal, TypedDict
from pydantic import BaseModel, Field

from langchain.tools import tool
from langchain.agents import create_agent
from langchain.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langgraph.checkpoint.memory import InMemorySaver


class AgentInput(TypedDict):
    """Simple input state for each sub-agent."""

    query: str


class AgentOutput(TypedDict):
    """Output from each sub-agent."""

    source: str
    result: str


class Classification(TypedDict):
    """A single routing decision: which agent to call with what query."""

    source: str
    query: str


class RouterState(TypedDict):
    query: str
    classifications: list[Classification]
    results: Annotated[list[AgentOutput], operator.add]
    final_answer: str


@tool(parse_docstring=True)
def search_code(query: str, repo: str = "main") -> str:
    """Search code in GitHub repositories.

    Args:
        query: The search query, e.g. "API authentication code examples".
        repo: The repository to search in, default is "main".

    Returns:
        str: A summary of the search results.
    """
    return f"Found code matching '{query}' in {repo}: authentication middileware in src/auth.py"


@tool(parse_docstring=True)
def search_issues(query: str) -> str:
    """Search GitHub issues and pull requests.

    Args:
        query: The search query, e.g. "API authentication issues".

    Returns:
        str: A summary of the search results.
    """
    return f"Found 3 issues matching '{query}': #142 (API auth docs), #89 (OAuth flow), #203 (token refresh)"


@tool(parse_docstring=True)
def search_prs(query: str) -> str:
    """Search pull requests for implementation details.

    Args:
        query: The search query, e.g. "API authentication implementation".

    Returns:
        str: A summary of the search results.
    """
    return f"PR #156 added JWT authentication, PR #178 updated OAuth scopes"


@tool(parse_docstring=True)
def search_notion(query: str) -> str:
    """Search Notion workspace for documentation.

    Args:
        query: The search query, e.g. "API authentication design decisions".

    Returns:
        str: A summary of the search results.
    """
    return f"Found documentation: 'API Authentication Guide' - covers OAuth2 flow, API keys, and JWT tokens"


@tool(parse_docstring=True)
def get_page(page_id: str) -> str:
    """Get a specific Notion page by ID.

    Args:
        page_id: The ID of the Notion page to retrieve.

    Returns:
        str: The content of the Notion page.
    """
    return f"Page content: Step-by-step authentication setup instructions"


@tool(parse_docstring=True)
def search_slack(query: str) -> str:
    """Search Slack messages and threads.

    Args:
        query: The search query, e.g. "API authentication discussions".

    Returns:
        str: A summary of the search results.
    """
    return f"Found discussion in #engineering: 'Use Bearer tokens for API auth, see docs for refresh flow'"


@tool(parse_docstring=True)
def get_thread(thread_id: str) -> str:
    """Get a specific Slack thread.

    Args:
        thread_id: The ID of the Slack thread to retrieve.

    Returns:
        str: The content of the Slack thread.
    """
    return f"Thread discusses best practices for API key rotation"


OLLAMA_BASE_URL = "http://192.168.88.179:11434"
model = ChatOllama(
    model="qwen3.5:9b",
    validate_model_on_init=True,
    base_url=OLLAMA_BASE_URL,
    reasoning=False,
    client_kwargs={"timeout": 120},
)


github_agent = create_agent(
    model,
    tools=[search_code, search_issues, search_prs],
    system_prompt=(
        "You are a GitHub expert. Answer questions about code, "
        "API references, and implementation details by searching "
        "repositories, issues, and pull requests."
    ),
)


notion_agent = create_agent(
    model,
    tools=[search_notion, get_page],
    system_prompt=(
        "You are a Notion expert. Answer questions about internal "
        "processes, policies, and team documentation by searching "
        "the organization's Notion workspace."
    ),
)


slack_agent = create_agent(
    model,
    tools=[search_slack, get_thread],
    system_prompt=(
        "You are a Slack expert. Answer questions by searching "
        "relevant threads and discussions where team members have "
        "shared knowledge and solutions."
    ),
)


# Define structured output schema for the classifier
class ClassificationResult(BaseModel):
    """Result of classifying a user query into agent-specific sub-questions."""

    classifications: list[Classification] = Field(
        description="List of agents to invoke with their targeted sub-questions"
    )


def classify_query(state: RouterState) -> dict:
    """Classify query and determine which agents to invoke.

    Args:
        state: The current state of the router, including the user query and any previous classifications.

    Returns:
        dict: A dictionary containing the classification results, specifying which agents to call and with what queries.
    """

    structured_llm = model.with_structured_output(ClassificationResult).bind(
        reasoning=True
    )

    result = structured_llm.invoke(
        [
            SystemMessage(
                "Analyze this query and determine which knowledge bases to consult. "
                "For each relevant source, generate a targeted sub-question optimized for that source.\n\n"
                "Available sources:\n"
                "- github: Code, API references, implementation details, issues, pull requests\n"
                "- notion: Internal documentation, processes, policies, team wikis\n"
                "- slack: Team discussions, informal knowledge sharing, recent conversations\n\n"
                "Return ONLY the sources that are relevant to the query. Each source should be have "
                "a targeted sub-question optimized for that specific knowledge domain.\n\n"
                'Example for "How do I authenticate API requests?":\n'
                '- github: "What authentication code exists? Search for auth middleware, JWT handling"\n'
                '- notion: "What authentication documentation exists? Look for API auth guides "'
                "(slack omitted because it's not relevant for this technical question)"
            ),
            HumanMessage(state["query"]),
        ],
    )

    return {"classifications": result.classifications}


def route_to_agents(state: RouterState) -> list[Send]:
    """Fan out to agents based on classifications.

    Args:
        state: The current state of the router, including the user query and classifications.

    Returns:
        list[Send]: A list of Send objects specifying which agents to call and with what queries.
    """
    return [Send(c["source"], {"query": c["query"]}) for c in state["classifications"]]


def query_github(state: AgentInput) -> dict:
    """Query the GitHub agent.

    Args:
        state: The input state containing the query for the GitHub agent.

    Returns:
        dict: A dictionary containing the response from the GitHub agent.
    """
    result = github_agent.invoke(HumanMessage(state["query"]))
    return {"results": [{"source": "github", "result": result["messages"][-1].content}]}


def query_notion(state: AgentInput) -> dict:
    """Query the Notion agent.

    Args:
        state: The input state containing the query for the Notion agent.

    Returns:
        dict: A dictionary containing the response from the Notion agent.
    """
    result = notion_agent.invoke(HumanMessage(state["query"]))
    return {"results": [{"source": "notion", "result": result["messages"][-1].content}]}


def query_slack(state: AgentInput) -> dict:
    """Query the Slack agent.

    Args:
        state: The input state containing the query for the Slack agent.

    Returns:
        dict: A dictionary containing the response from the Slack agent.
    """
    result = slack_agent.invoke(HumanMessage(state["query"]))
    return {"results": [{"source": "slack", "result": result["messages"][-1].content}]}


def synthesize_results(state: RouterState) -> dict:
    """Combine results from all agents into a coherent answer.

    Args:
        state: The current state of the router, including the original query and results from all agents.

    Returns:
        dict: A dictionary containing the final synthesized answer that combines the information retrieved from all sources.
    """
    if not state["results"]:
        return {"final_answer": "No results found from any knowledge source."}

    # Format results for synthesis
    formatted = [
        f"**From {r['source'].title()}:**\n{r['result']}" for r in state["results"]
    ]

    synthesis_response = model.invoke(
        [
            SystemMessage(
                f"Synthesize these search results to answer the original question: \"{state['query']}\"\n\n"
                "- Combine information from multiple sources without redundancy\n"
                "- Highlight the most relevant and actionable information\n"
                "- Note any discrepancies between sources\n"
                "- Keep the response concise and well-organized"
            ),
            HumanMessage("\n\n".join(formatted)),
        ]
    )
    return {"final_answer": synthesis_response.content}


workflow = (
    StateGraph(RouterState)
    .add_node("classify", classify_query)
    .add_node("github", query_github)
    .add_node("notion", query_notion)
    .add_node("slack", query_slack)
    .add_node("synthesize", synthesize_results)
    .add_edge(START, "classify")
    .add_conditional_edges("classify", route_to_agents, ["github", "notion", "slack"])
    .add_edge("github", "synthesize")
    .add_edge("notion", "synthesize")
    .add_edge("slack", "synthesize")
    .add_edge("synthesize", END)
    .compile()
)


@tool(parse_docstring=True)
def search_knowledge_base(query: str) -> str:
    """Search across multiple knowledge sources (GitHub, Notion, Slack).

    Use this to find information about code, documentation, or team discussions.

    Args:
        query: The user's question or information need.

    Returns:
        str: A synthesized answer combining information from all relevant sources.
    """
    result = workflow.invoke({"query": query})
    return result["final_answer"]


conversation_agent = create_agent(
    model,
    tools=[search_knowledge_base],
    system_prompt=(
        "You are helpful assistant that answers questions about our organization. "
        "Use the search_knowledge_base tool to find information across our code, "
        "documentation, and team discussions."
    ),
    checkpointer=InMemorySaver(),
)


config = {"configurable": {"thread_id": "user-123"}}
result = conversation_agent.invoke(
    {"messages": [HumanMessage("How do I authenticate API requests?")]},
    config=config,
)
print(result["messages"][-1].content)
result = conversation_agent.invoke(
    {"messages": [HumanMessage("What about rate limiting for those endpoints?")]},
    config=config,
)
print(result["messages"][-1].content)
