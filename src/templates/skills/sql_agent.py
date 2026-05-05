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
from langchain_ollama import ChatOllama
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.messages import HumanMessage
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.types import Command
from langgraph.checkpoint.memory import InMemorySaver

db = SQLDatabase.from_uri("sqlite:///Chinook.db")
# print(f"Dialect: {db.dialect}")
# print(f"Available tables: {db.get_usable_table_names()}")
# print(f"Sample output: {db.run('SELECT * FROM Artist LIMIT 5;')}")

model = ChatOllama(
    model="qwen3.5:9b",
    base_url=os.getenv("OLLAMA_BASE_URL"),
    reasoning=False,
    client_kwargs={"timeout": 120},
)
toolkit = SQLDatabaseToolkit(db=db, llm=model)
tools = toolkit.get_tools()

# for tool in tools:
#     print(f"{tool.name}: {tool.description}\n")

agent = create_agent(
    model,
    tools=tools,
    system_prompt=(
        "You are an agent designed to interact with a SQL database.\n"
        "Give an input question, create a syntactically correct {dialect} query to run, "
        "then look at the results of the query and return the answer. Unless the user "
        "specifies a specific number of examples they wish to obtain, always limit your "
        "query to at most {top_k} results.\n\n"
        "You can order the results by a relevant column to return the most interesting "
        "examples in the database. Never query for all the columns from a specific table, "
        "only ask for the relevant columns given the question.\n\n"
        "You MUST double check your query before executing it. If you get an error while "
        "executing a query, rewrite the query and try again.\n\n"
        "DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.\n\n"
        "To start you should ALWAYS look at the tables in the database to see what you "
        "can query. Do NOT skip this step.\n\n"
        "Then you should query the schema of the most relevant tables."
    ).format(dialect=db.dialect, top_k=5),
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={"sql_db_query": True},
            description_prefix="Tool execution pending approval",
        )
    ],
    checkpointer=InMemorySaver(),
)


if __name__ == "__main__":
    config = {"configurable": {"thread_id": "1"}}

    for step in agent.stream(
        {"messages": [HumanMessage("Which genre on average has the longest tracks?")]},
        config=config,
        stream_mode="values",
    ):
        if "__interrupt__" in step:
            print("INTERRUPTED:")
            interrupt = step["__interrupt__"][0]
            for request in interrupt.value["action_requests"]:
                print(request["description"])
        elif "messages" in step:
            step["messages"][-1].pretty_print()
        else:
            pass

    for step in agent.stream(
        Command(resume={"decisions": [{"type": "approve"}]}),
        config=config,
        stream_mode="values",
    ):
        if "messages" in step:
            step["messages"][-1].pretty_print()
        if "__interrupt__" in step:
            print("INTERRUPTED:")
            interrupt = step["__interrupt__"][0]
            for request in interrupt.value["action_requests"]:
                print(request["description"])
        else:
            pass
