import os
from typing import NotRequired, Literal, Callable
from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain.tools import tool, ToolRuntime
from langchain.messages import ToolMessage, HumanMessage
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import (
    wrap_model_call,
    ModelRequest,
    ModelResponse,
    SummarizationMiddleware,
)
from langchain_core.utils.uuid import uuid7
from langgraph.types import Command
from langgraph.checkpoint.memory import InMemorySaver


load_dotenv()


model = ChatOllama(
    model=os.getenv("OLLAMA_LLM_MODEL"),
    base_url=os.getenv("OLLAMA_BASE_URL"),
    validate_model_on_init=True,
    client_kwargs={"timeout": 120},
)


# Define the possible workflow steps
SupportStep = Literal["warranty_collector", "issue_classifier", "resolution_specialist"]


class SupportState(AgentState):
    """State for customer support workflow."""

    current_step: NotRequired[SupportStep]
    warranty_status: NotRequired[Literal["in_warranty", "out_of_warranty"]]
    issue_type: NotRequired[Literal["hardware", "software"]]


@tool(parse_docstring=True)
def record_warranty_status(
    status: Literal["in_warranty", "out_of_warranty"],
    runtime: ToolRuntime[None, SupportState],
) -> Command:
    """Record the customer's warranty status and transition to issue classification.

    Args:
        status (Literal["in_warranty", "out_of_warranty"]): The warranty status of the customer's product.
        runtime (ToolRuntime[None, SupportState]): The tool runtime for managing state and messages.

    Returns:
        Command: A command to update the state and send a message about the recorded warranty status.
    """
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=f"Warranty status recorded as: {status}",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
            "warranty_status": status,
            "current_step": "issue_classifier",
        }
    )


@tool(parse_docstring=True)
def record_issue_type(
    issue_type: Literal["hardware", "software"],
    runtime: ToolRuntime[None, SupportState],
) -> Command:
    """Record the type of issue and transition to resolution specialist.

    Args:
        issue_type (Literal["hardware", "software"]): The type of issue reported by the customer.
        runtime (ToolRuntime[None, SupportState]): The tool runtime for managing state and messages.

    Returns:
        Command: A command to update the state and send a message about the recorded issue type.
    """
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=f"Issue type recorded as: {issue_type}",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
            "issue_type": issue_type,
            "current_step": "resolution_specialist",
        }
    )


@tool(parse_docstring=True)
def escalate_to_human(reason: str) -> str:
    """Escalate the case to a human support specialist.

    Args:
        reason (str): The reason for escalation.

    Returns:
        str: A message indicating that the case has been escalated to a human support specialist.
    """
    return f"Escalating to human support. Reason: {reason}"


@tool(parse_docstring=True)
def provide_solution(solution: str) -> str:
    """Provide a solution to the customer's issue.

    Args:
        solution (str): The proposed solution for the customer's issue.

    Returns:
        str: A message indicating the proposed solution for the customer's issue.
    """
    return f"Solution provided: {solution}"


@tool(parse_docstring=True)
def go_back_to_warranty(runtime: ToolRuntime[None, SupportState]) -> Command:
    """Go back to warranty verification step.

    Args:
        runtime (ToolRuntime[None, SupportState]): The tool runtime for managing state and messages.

    Returns:
        Command: A command to update the state and send a message about going back to warranty verification.
    """
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=f"Back to warranty verification step.",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
            "current_step": "warranty_collector",
        }
    )


@tool(parse_docstring=True)
def go_back_to_classification(runtime: ToolRuntime[None, SupportState]) -> Command:
    """Go back to issue classification step.

    Args:
        runtime (ToolRuntime[None, SupportState]): The tool runtime for managing state and messages.

    Returns:
        Command: A command to update the state and send a message about going back to issue classification.
    """
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=f"Back to issue classification step.",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
            "current_step": "issue_classifier",
        }
    )


# Collect all tools from all step configurations
all_tools = [
    record_warranty_status,
    record_issue_type,
    provide_solution,
    escalate_to_human,
    go_back_to_warranty,
    go_back_to_classification,
]


STEP_CONFIG = {
    "warranty_collector": {
        "prompt": (
            "You are a customer support agent helping with device issues.\n\n"
            "CURRENT STAGE: Warranty verification\n\n"
            "At this step, you need to:\n"
            "1. Greet the customer warmly\n"
            "2. Ask if their device is under warranty\n"
            "3. Use record_warranty_status to record their response and move to the next step\n\n"
            "Be Conversational and friendly. Don't ask multiple questions at once."
        ),
        "tools": [record_warranty_status],
        "requires": [],
    },
    "issue_classifier": {
        "prompt": (
            "You are a customer support agent helping with device issues.\n\n"
            "CURRENT STAGE: Issue classification\n"
            "CUSTOMER INFO: Warranty status is {warranty_status}\n\n"
            "At this step, you need to:\n"
            "1. Ask the customer to describe their issue\n"
            "2. Determin if it's a hardware issue (physical damage, broken parts) or software issue (app crashes, performance)\n"
            "3. Use record_issue_type to record the classification and move to the next step\n\n"
            "If the customer indicates any information was wrong, use go_back_to_warranty to correct warranty status\n\n"
            "If unclear, ask clarifying questions before classifying."
        ),
        "tools": [record_issue_type, go_back_to_warranty],
        "requires": ["warranty_status"],
    },
    "resolution_specialist": {
        "prompt": (
            "You are a customer support agent helping with device issues.\n\n"
            "CURRENT STAGE: Resolution\n"
            "CUSTOMER INFO: Warranty status is {warranty_status}, issue type is {issue_type}\n\n"
            "At this step, you need to:\n"
            "1. For SOFTWARE issues: provide troubleshooting steps using provide_solution\n"
            "2. For HARDWARE issues:\n"
            "  - If IN WARRANTY: explain warranty repair process using provide_solution\n"
            "  - If OUT OF WARRANTY: escalate_to_human for paid repair options\n\n"
            "If the customer indicates any information was wrong, use:\n"
            "- go_back_to_warranty to correct warranty status\n"
            "- go_back_to_classification to correct issue type\n\n"
            "Be specific and helpful in your solutions."
        ),
        "tools": [
            provide_solution,
            escalate_to_human,
            go_back_to_warranty,
            go_back_to_classification,
        ],
        "requires": ["warranty_status", "issue_type"],
    },
}


@wrap_model_call
def apply_step_config(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """Configure agent behavior based on the current step."""
    # Get current step (defaults to warranty_collector for first interaction)
    current_step = request.state.get("current_step", "warranty_collector")

    # Look up step configuration
    stage_config = STEP_CONFIG[current_step]

    # Validate required state exists
    for key in stage_config["requires"]:
        if request.state.get(key) is None:
            raise ValueError(f"{key} must be set before reaching {current_step}")

    # Format prompt with state values (supports {warranty_status}, {issue_type}, etc.)
    system_prompt = stage_config["prompt"].format(**request.state)

    # Inject system prompt and step-specific tools
    request = request.override(
        system_message=system_prompt,
        tools=stage_config["tools"],
    )

    return handler(request)


# Create the agent with step-based configuration
agent = create_agent(
    model,
    tools=all_tools,
    state_schema=SupportState,
    middleware=[
        apply_step_config,
        SummarizationMiddleware(
            model=model,
            trigger=[("tokens", 2000), ("messages", 8)],
            keep=("messages", 4), # Nguyên tắc: keep nên bằng ~50% trigger (tính theo messages), để có đủ buffer trước khi trigger lại.
        ),
    ],
    checkpointer=InMemorySaver(),
)


# if __name__ == "__main__":
    # thread_id = str(uuid7())
    # config = {"configurable": {"thread_id": thread_id}}

    # while True:
    #     user_input = input("Customer: ")
    #     if user_input.lower() in ["exit", "quit"]:
    #         print("Exiting support agent.")
    #         break

    #     for msg in agent.stream({"messages": [HumanMessage(user_input)]}, config, stream_mode="values"):
    #         # print(f"Agent: {msg.content}")
    #     # print(f"Current step: {result.get('current_step')}")


# Configuratuion for this conversation thread
thread_id = str(uuid7())
config = {"configurable": {"thread_id": thread_id}}

# Turn 1: Initial message - starts with warranty_collector step
print("++++++ Turn 1: Warranty Collector ++++++")
result = agent.invoke({"messages": [HumanMessage("Hi, my phone is having problems.")]}, config)
for msg in result["messages"]:
    msg.pretty_print()
print(f"Current step: {result.get('current_step')}")

# Turn 2: User responds about warranty
print("\n\n++++++ Turn 2: Warranty Response ++++++")
result = agent.invoke({"messages": [HumanMessage("Yes, it's still under warranty")]}, config)
for msg in result["messages"]:
    msg.pretty_print()
print(f"Current step: {result.get('current_step')}")

# Turn 3: Wrong warranty status - user indicates it's actually out of warranty
print("\n\n++++++ Turn 3: Correcting Warranty Status ++++++")
result = agent.invoke({"messages": [HumanMessage("Actually, I made a mistake - my device is out of warranty")]}, config)
for msg in result["messages"]:
    msg.pretty_print()
print(f"Current step: {result.get('current_step')}")

# Turn 4: User responds about warranty
print("\n\n++++++ Turn 4: Warranty Response ++++++")
result = agent.invoke({"messages": [HumanMessage("My device is out of warranty")]}, config)
for msg in result["messages"]:
    msg.pretty_print()
print(f"Current step: {result.get('current_step')}")

# Turn 5: User describes the issue
print("\n\n++++++ Turn 5: Issue Description ++++++")
result = agent.invoke({"messages": [HumanMessage("The screen is physically cracked from dropping it")]}, config)
for msg in result["messages"]:
    msg.pretty_print()
print(f"Current step: {result.get('current_step')}")

# Turn 6: Resolution
print("\n\n++++++ Turn 6: Resolution ++++++")
result = agent.invoke({"messages": [HumanMessage("What should I do?")]}, config)
for msg in result["messages"]:
    msg.pretty_print()
print(f"Current step: {result.get('current_step')}")
