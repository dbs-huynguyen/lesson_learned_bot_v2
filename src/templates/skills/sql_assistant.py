from typing import TypedDict, Callable, NotRequired

from langchain.tools import tool, ToolRuntime
from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelRequest,
    ModelResponse,
    AgentMiddleware,
    AgentState,
)
from langchain_core.utils.uuid import uuid7
from langchain.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_ollama.chat_models import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command


class Skill(TypedDict):
    """A skill that can be progressively disclosed to the agent."""

    name: str  # Unique identifier for the skill
    description: str  # 1-2 sentence description to show in system prompt
    content: str  # Full skill content with detailed instructions


SKILLS: list[Skill] = [
    {
        "name": "sales_analytics",
        "description": "Database schema and business logic for sales data analysis including customers, orders, and revenue.",
        "content": (
            "# Sales Analytics Schema\n\n"
            "## Tables\n\n"
            "### customers\n"
            "- customer_id (PRIMARY KEY)\n"
            "- name\n"
            "- email\n"
            "- signup_date\n"
            "- status (active/inactive)\n"
            "- customer_tier (bronze/silver/gold/platinum)\n\n"
            "### orders\n"
            "- order_id (PRIMARY KEY)\n"
            "- customer_id (FOREIGN KEY -> customers)\n"
            " - order_date\n"
            "- status (pending/completed/cancelled/refunded)\n"
            "- total_amount\n"
            "- sales_region (north/south/east/west)\n\n"
            "### order_items\n"
            "- item_id (PRIMARY KEY)\n"
            "- order_id (FOREIGN KEY -> orders)\n"
            "- product_id\n"
            "- quantity\n"
            "- unit_price\n"
            "- discount_percent\n\n"
            "## Business Logic\n\n"
            "**Active customers**: status = 'active' AND signup_date <= CURRENT_DATE - INTERVAL '90 days'\n\n"
            "**Revenue calculation**: Only count orders with status = 'completed'. Use total_amount from orders table, which already accounts for discounts.\n\n"
            "**Customer lifetime value (CLV)**: Sum of all completed order amounts for a customer.\n\n"
            "**High-value orders**: Orders with total_amount > 1000\n\n"
            "## Example Query\n\n"
            "-- Get top 10 customers by revenue in the last quarter\n"
            "SELECT \n"
            "c.customer_id, \n"
            "c.name, \n"
            "c.customer_tier, \n"
            "SUM(o.total_amount) AS total_revenue \n"
            "FROM customers c \n"
            "JOIN orders o ON c.customer_id = o.customer_id \n"
            "WHERE o.status = 'completed' \n"
            "AND o.order_date >= CURRENT_DATE - INTERVAL '3 months' \n"
            "GROUP BY c.customer_id, c.name, c.customer_tier \n"
            "ORDER BY total_revenue DESC \n"
            "LIMIT 10;"
        ),
    },
    {
        "name": "inventory_management",
        "description": "Database schema and business logic for inventory tracking including products, warehouses, and stock levels.",
        "content": (
            "# Inventory Management Schema\n\n"
            "## Tables\n\n"
            "### products\n"
            "- product_id (PRIMARY KEY)\n"
            "- product_name\n"
            "- sku\n"
            "- category\n"
            "- unit_cost\n"
            "- unit_cost\n"
            "- reorder_point (minimum stock level before reordering)\n"
            "- discontinued (boolean)\n\n"
            "### warehouses\n"
            "- warehouse_id (PRIMARY KEY)\n"
            "- warehouse_name\n"
            "- location\n"
            "- capacity\n\n"
            "### inventory\n"
            "- inventory_id (PRIMARY KEY)\n"
            "- product_id (FOREIGN KEY -> products)\n"
            "- warehouse_id (FOREIGN KEY -> warehouses)\n"
            "- quantity_on_hand\n"
            "- last_updated\n\n"
            "### stock_movements\n"
            "- movement_id (PRIMARY KEY)\n"
            "- product_id (FOREIGN KEY -> products)\n"
            "- warehouse_id (FOREIGN KEY -> warehouses)\n"
            "- movement_type (inbound/outbound/transfer/adjustment)\n"
            "- quantity (positive for inbound, negative for outbound)\n"
            "- movement_date\n"
            "- reference_number\n\n"
            "## Business Logic\n\n"
            "**Available stock**: quantity_on_hand from inventory table where quantity_on_hand > 0\n\n"
            "**Products needing reorder**: Products where total quantity_on_hand across all warehouses is less than or equal to the product's reorder_point\n\n"
            "**Active products only**: Exclude products where discontinued = true unless specifically analyzing discontinued items\n\n"
            "**Stock valuation**: quantity_on_hand * unit_cost for each product\n\n"
            "## Example Query\n\n"
            "-- Find products below reorder point across all warehouses\n"
            "SELECT \n"
            "p.product_id, \n"
            "p.product_name, \n"
            "p.reorder_point, \n"
            "SUM(i.quantity_on_hand) AS total_stock, \n"
            "p.unit_cost, \n"
            "(p.reorder_point - SUM(i.quantity_on_hand)) AS units_to_reorder \n"
            "FROM products p \n"
            "JOIN inventory i ON p.product_id = i.product_id \n"
            "WHERE p.discontinued = false \n"
            "GROUP BY p.product_id, p.product_name, p.reorder_point, p.unit_cost \n"
            "HAVING SUM(i.quantity_on_hand) <= p.reorder_point \n"
            "ORDER BY units_to_reorder DESC;"
        ),
    },
]


class CustomState(AgentState):
    skills_loaded: NotRequired[list[str]]


@tool(parse_docstring=True)
def load_skill(skill_name: str, runtime: ToolRuntime) -> Command:
    """Load the full content of a skill into the agent's context.

    Use this when you need detailed information about how to handle a specific
    type of request. This will provide you with comprehensive instructions,
    policies, and guidelines for the skill area.

    Args:
        skill_name (str): The unique name of the skill to load. (e.g., "expense_reporting", "travel_booking")
        runtime (ToolRuntime): The tool runtime context, which can be used to access agent state or other tools if needed.

    Returns:
        Command: A command that updates the agent's messages with the skill content and tracks which skills have been loaded.
    """
    # Find and return the requested skill
    for skill in SKILLS:
        if skill["name"] == skill_name:
            skill_content = f"Loaded skill: {skill_name}\n\n{skill['content']}"

            # Update state to track loaded skill
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content=skill_content,
                            tool_call_id=runtime.tool_call_id,
                        )
                    ],
                    "skills_loaded": [skill_name],
                }
            )

    # Skill not found
    available = ", ".join(s["name"] for s in SKILLS)
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=f"Skill '{skill_name}' not found. Available skills: {available}",
                    tool_call_id=runtime.tool_call_id,
                )
            ]
        }
    )


@tool(parse_docstring=True)
def write_sql_query(query: str, vertical: str, runtime: ToolRuntime) -> str:
    """Write and validate a SQL query for a specific business vertical.

    This tool helps format and validate SQL queries. You must load the
    appropriate skill first to understand the database schema.

    Args:
        query (str): The SQL query to write and validate.
        vertical (str): The business vertical (e.g., "sales_analytics", "inventory_management") that the query is targeting.
        runtime (ToolRuntime): The tool runtime context, which can be used to access agent state or other tools if needed.

    Returns:
        str: A formatted response indicating the SQL query and whether it is valid against the loaded schema for the specified vertical.
    """
    # Check if the required skill has been loaded
    skills_loaded = runtime.state.get("skills_loaded", [])

    if vertical not in skills_loaded:
        return (
            f"Error: You must load the '{vertical}' skill first "
            f"to understand the database schema before writing queries. "
            f"Use load_skill('{vertical}') to load the schema."
        )

    # Validate and format the query
    return (
        f"SQL Query for {vertical}:\n\n"
        f"```sql\n{query}\n```\n\n"
        f"✓ Query validated against {vertical} schema \n"
        "Ready to execute against the database."
    )


class SkillMiddleware(AgentMiddleware[CustomState]):
    """Middleware that injects skill descriptions into the system prompt."""

    # Register the load_skill tool as a class variable
    state_schema = CustomState
    tools = [load_skill, write_sql_query]

    def __init__(self):
        """Initialize and generate the skills prompt from SKILLS."""
        # Build skills prompt from the SKILLS list
        skills_list = []
        for skill in SKILLS:
            skills_list.append(f"- **{skill['name']}**: {skill['description']}")
        self.skills_prompt = "\n".join(skills_list)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Sync: Inject skill descriptions into system prompt."""
        # Build the skills addendum
        skills_addendum = (
            f"\n\n## Available Skills\n\n{self.skills_prompt}\n\n"
            "Use the load_skill tool when you need detailed information "
            "about handling a specific type of request."
        )

        # Append to system message content blocks
        new_content = list(request.system_message.content_blocks) + [
            {"type": "text", "text": skills_addendum}
        ]
        new_system_message = SystemMessage(new_content)
        modified_request = request.override(system_message=new_system_message)
        return handler(modified_request)


OLLAMA_BASE_URL = "http://209.121.195.118:13037"
model = ChatOllama(
    model="qwen3.5:9b",
    base_url=OLLAMA_BASE_URL,
    reasoning=False,
    client_kwargs={"timeout": 120},
)

# Create the agent with skill support
agent = create_agent(
    model,
    system_prompt=(
        "You are a SQL query assistant that helps users "
        "write queries against business databases."
    ),
    middleware=[SkillMiddleware()],
    checkpointer=InMemorySaver(),
)


# Example usage
if __name__ == "__main__":
    # Configuration for this conversation thread
    thread_id = str(uuid7())
    config = {"configurable": {"thread_id": thread_id}}

    # Ask for a SQL query
    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    "Write a SQL query to find all customers "
                    "who made orders over $1000 in the last month."
                )
            ]
        },
        config=config,
    )

    # Print the conversation
    for message in result["messages"]:
        if hasattr(message, "pretty_print"):
            message.pretty_print()
        else:
            print(f"{message.type}: {message.content}")
