# # Download the Chinook.db file if it doesn't exist locally
# import requests, pathlib

# url = "https://storage.googleapis.com/benchmarks-artifacts/chinook/Chinook.db"
# local_path = pathlib.Path("Chinook.db")

# if local_path.exists():
#     print(f"{local_path} already exists, skipping download.")
# else:
#     response = requests.get(url)
#     if response.status_code == 200:
#         local_path.write_bytes(response.content)
#         print(f"File downloaded and saved as {local_path}")
#     else:
#         print(f"Failed to download the file. Status code: {response.status_code}")


import os
from uuid import uuid4
from typing import Literal
from dotenv import load_dotenv

from langchain_core.runnables import RunnableConfig
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_ollama import ChatOllama
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.graph import START, END, MessagesState, StateGraph
from langgraph.types import Command
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import InMemorySaver


load_dotenv()


db = SQLDatabase.from_uri("sqlite:///Chinook.db")
# print(f"Dialect: {db.dialect}")
# print(f"Available tables: {db.get_usable_table_names()}")
# print(f"Sample output: {db.run('SELECT * FROM Artist LIMIT 5;')}")

model = ChatOllama(
    model=os.getenv("OLLAMA_LLM_MODEL"),
    base_url=os.getenv("OLLAMA_BASE_URL"),
    validate_model_on_init=False,
    client_kwargs={"timeout": 120},
    temperature=0.0,
    seed=9999,
    num_ctx=4096,
    reasoning=False,
)
toolkit = SQLDatabaseToolkit(db=db, llm=model)
tools = toolkit.get_tools()

# for tool in tools:
#     print(f"{tool.name}: {tool.description}\n")

list_tables_tool = next(tool for tool in tools if tool.name == "sql_db_list_tables")
# list_tables = ToolNode([list_tables_tool], name="list_tables")

get_schemas_tool = next(tool for tool in tools if tool.name == "sql_db_schema")
# get_schemas = ToolNode([get_schemas_tool], name="get_schemas")

run_query_tool = next(tool for tool in tools if tool.name == "sql_db_query")
# run_query = ToolNode([run_query_tool], name="run_query")


def list_tables(state: MessagesState) -> dict:
    llm_with_tools = model.bind_tools([list_tables_tool])
    tool_call_message = llm_with_tools.invoke(state["messages"])

    tool_call = tool_call_message.tool_calls[0]
    tool_message = list_tables_tool.invoke(tool_call)

    return {"messages": [tool_call_message, tool_message]}


def get_schemas(state: MessagesState) -> dict:
    llm_with_tools = model.bind_tools([get_schemas_tool])
    tool_call_message = llm_with_tools.invoke(state["messages"])

    tool_call = tool_call_message.tool_calls[0]
    tool_message = get_schemas_tool.invoke(tool_call)

    return {"messages": [tool_call_message, tool_message]}


def generate_query(state: MessagesState) -> dict:
    system_message = SystemMessage(
        content=(
            "You are an agent designed to interact with a SQL database. "
            "Give an input question, create a syntactically correct {dialect} query to run, "
            "then look at the results of the query and return the answer. "
            "Unless the user specifies a specific number of examples they wish to obtain, "
            "always limit your query to at most {top_k} results.\n\n"
            "You can order the results by a relevant column to return the most interesting "
            "examples in the database. Never query for all the columns from a specific table, "
            "only ask for the relevant columns given the question.\n\n"
            "DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database."
        ).format(dialect=db.dialect, top_k=5)
    )
    llm_with_tools = model.bind_tools([run_query_tool])
    tool_call_message = llm_with_tools.invoke([system_message] + state["messages"])

    return {"messages": [tool_call_message]}


def should_continue(state: MessagesState) -> Literal["__end__", "check_query"]:
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        return END
    else:
        return "check_query"


def check_query(state: MessagesState) -> dict:
    system_message = SystemMessage(
        content=(
            "You are a SQL expert with a strong attention to detail. "
            "Double check the {dialect} query for common mistakes, including:\n"
            "- Using NOT IN with NULL values\n"
            "- Using UNION when UNION ALL should have been used\n"
            "- Using BETWEEN for exclusive ranges\n"
            "- Data type mismatch in predicates\n"
            "- Properly quoting identififiers\n"
            "- Using the correct number of arguments for functions\n"
            "- Casting to the correct data type\n"
            "- Using the proper columns for joins\n\n"
            "If there are any of the above mistakes, rewrite the query. If there are no mistakes, "
            "just reproduce the original query.\n\n"
            "You will call the appropriate tool to execute the query after running this check."
        ).format(dialect=db.dialect)
    )
    # Generate an artificial user message to check
    tool_call = state["messages"][-1].tool_calls[0]
    user_message = HumanMessage(content=tool_call["args"]["query"])
    llm_with_tools = model.bind_tools([run_query_tool])
    response = llm_with_tools.invoke([system_message, user_message])
    response.id = state["messages"][-1].id

    return {"messages": [response]}


def run_query(state: MessagesState) -> dict:
    tool_call_message = state["messages"][-1]

    tool_call = tool_call_message.tool_calls[0]
    tool_message = run_query_tool.invoke(tool_call)
    response = model.invoke(state["messages"] + [tool_message])

    return {"messages": [tool_message, response]}


workflow = (
    StateGraph(MessagesState)
    .add_node("list_tables", list_tables)
    .add_node("get_schemas", get_schemas)
    .add_node("generate_query", generate_query)
    .add_node("check_query", check_query)
    .add_node("run_query", run_query)
    .add_edge(START, "list_tables")
    .add_edge("list_tables", "get_schemas")
    .add_edge("get_schemas", "generate_query")
    .add_conditional_edges("generate_query", should_continue)
    .add_edge("check_query", "run_query")
    .add_edge("run_query", END)
    .compile()
)

import pprint
if __name__ == "__main__":
    question = "Which genre on average has the longest tracks?"

    for step in workflow.stream(
        {"messages": [HumanMessage(question)]}, stream_mode="values"
    ):
        if "messages" in step:
            for msg in step["messages"]:
                msg.pretty_print()
        print("+" * 50)
