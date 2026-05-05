from langchain.tools import tool, ToolRuntime
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

OLLAMA_BASE_URL = "http://192.168.88.179:11434"


model = ChatOllama(
    model="qwen3.5:9b",
    validate_model_on_init=True,
    base_url=OLLAMA_BASE_URL,
    client_kwargs={"timeout": 120},
)


@tool(parse_docstring=True)
def create_calendar_event(
    title: str,
    start_datetime: str,
    end_datetime: str,
    attendees: list[str],
    location: str = "",
) -> str:
    """Creates a calendar event. Requires exact ISO datetime format.

    Args:
        title (str): The title of the event.
        start_datetime (str): The start date and time of the event in ISO format.
        end_datetime (str): The end date and time of the event in ISO format.
        attendees (list[str]): A list of email addresses of the attendees.
        location (str, optional): The location of the event. Defaults to "".

    Returns:
        str: A confirmation message about the created event.
    """
    return f"Event created: {title} from {start_datetime} to {end_datetime} at {location} with {len(attendees)} attendees"


@tool(parse_docstring=True)
def send_email(
    to: list[str],
    subject: str,
    body: str,
    cc: list[str] = [],
) -> str:
    """Send an email via email API. Requires properly formatted addresses.

    Args:
        to (list[str]): A list of email addresses to send the email to.
        subject (str): The subject of the email.
        body (str): The body content of the email.
        cc (list[str], optional): A list of email addresses to CC. Defaults to [].

    Returns:
        str: A confirmation message about the sent email.
    """
    return f"Email sent to {', '.join(to)} - Subject: {subject}"


@tool(parse_docstring=True)
def get_available_time_slots(
    attendees: list[str],
    date: str,
    duration_minutes: int,
) -> list[str]:
    """Check calendar availability for given attendees on a specific date.

    Args:
        attendees (list[str]): A list of email addresses of the attendees.
        date (str): The date to check availability for in ISO format.
        duration_minutes (int): The required duration of the meeting in minutes.

    Returns:
        list[str]: A list of available time slots in ISO format.
    """
    return ["09:00", "14:00", "16:00"]


@tool(parse_docstring=True)
def get_current_datetime() -> str:
    """Get the current date and time in ISO format.

    Returns:
        str: The current date and time in ISO format.
    """
    from datetime import datetime

    return datetime.now().isoformat()


calendar_agent = create_agent(
    model,
    tools=[create_calendar_event, get_available_time_slots, get_current_datetime],
    system_prompt=(
        "You are a calendar scheduling assistant. "
        "Parse natural language scheduling requests (e.g., 'next Tuesday at 2pm') "
        "into proper ISO datetime formats. "
        "Use create_calendar_event to schedule events. "
        "Use get_current_datetime to get current date and time, and get_available_time_slots to "
        "check availability when needed. "
        "If there is no suitable time slot, stop and confirm unavailability in your response. "
        "Always confirm what was scheduled in your final response."
    ),
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={"create_calendar_event": True},
            description_prefix="Calendar event pending approval",
        )
    ],
)


email_agent = create_agent(
    model,
    tools=[send_email],
    system_prompt=(
        "You are an email assistant. "
        "Compose professional emails based on natural language requests. "
        "Extract recipient information and craft appropriate subject lines and body text. "
        "Use send_email to send the message. "
        "Always confirm what was sent in your final response."
    ),
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={"send_email": True},
            description_prefix="Outbound email pending approval",
        )
    ],
)


@tool(parse_docstring=True)
def schedule_event(request: str, runtime: ToolRuntime) -> str:
    """Schedule calendar events using natural language request.

    Use this when the user wants to create, modify, or check calendar appointments.
    Handles date/time parsing, avalability checking, and event creation.

    Args:
        request (str): Natural language scheduling request (e.g., 'meeting with design team next Tuesday at 2pm for one hour').

    Returns:
        str: Confirmation message about the scheduled event.
    """
    original_user_message = next(
        message for message in runtime.state["messages"] if message.type == "human"
    )
    prompt = (
        "You are assisting with the following user inquiry:\n\n"
        f"{original_user_message.content}\n\n"
        "You are tasked with the following sub-request:\n\n"
        f"{request}"
    )
    result = calendar_agent.invoke({"messages": [{"role": "human", "content": prompt}]})
    # Option 1: Return just the confirmation message
    return result["messages"][-1].content
    # Option 2: Return structured data
    # return json.dumps(
    #     {
    #         "status": "success",
    #         "event_id": "evt_123",
    #         "summary": result["messages"][-1].content,
    #     }
    # )


@tool(parse_docstring=True)
def manage_email(request: str) -> str:
    """Send emails using natural language request.

    Use this when the user wants to send notifications, reminders, or any email communication.
    Handles recipient extraction, subject generation, and email composition.

    Args:
        request (str): Natural language email request (e.g., 'send them a reminder about the meeting')

    Returns:
        str: Confirmation message about the sent email.
    """
    result = email_agent.invoke({"messages": [{"role": "human", "content": request}]})
    return result["messages"][-1].content


supervisor_agent = create_agent(
    model,
    tools=[schedule_event, manage_email],
    system_prompt=(
        "You are a helpful personal assistant. "
        "You can schedule calendar events and send emails. "
        "Break down user requests into appropriate tool calls and coordinate the results. "
        "When a request involves multiple actions, use multiple tools in sequence."
    ),
    checkpointer=InMemorySaver(),
)

query = (
    "Schedule a meeting with Alice and Bob next Tuesday at 2pm for 1 hour, "
    "and send them an email reminder about reviewing the new mockups."
)

config = {"configurable": {"thread_id": "6"}}

interrupts = []
for step in supervisor_agent.stream(
    {"messages": [{"role": "human", "content": query}]},
    config=config,
):
    for update in step.values():
        if isinstance(update, dict):
            for message in update.get("messages", []):
                message.pretty_print()
        else:
            interrupt_ = update[0]
            interrupts.append(interrupt_)
            print(f"\nINTERRUPTED: {interrupt_.id}")


for interrupt_ in interrupts:
    for request in interrupt_.value["action_requests"]:
        print(f"INTERRUPTED: {interrupt_.id}")
        print(f"{request['description']}\n")


resume = {}
for interrupt_ in interrupts:
    if interrupt_.id == "6c62d2b703b5ec79ccc12175416ac126":
        # Edit email
        edited_action = interrupt_.value["action_requests"][0].copy()
        edited_action["args"]["subject"] = "Mockups reminder"
        resume[interrupt_.id] = {
            "decisions": [{"type": "edit", "edited_action": edited_action}]
        }
    else:
        resume[interrupt_.id] = {"decisions": [{"type": "approve"}]}


interrupts = []
for step in supervisor_agent.stream(
    Command(resume=resume),
    config=config,
):
    for update in step.values():
        if isinstance(update, dict):
            for message in update.get("messages", []):
                message.pretty_print()
        else:
            interrupt_ = update[0]
            interrupts.append(interrupt_)
            print(f"\nINTERRUPTED: {interrupt_.id}")
