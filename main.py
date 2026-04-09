import asyncio

from langgraph_sdk import get_client

API_URL = "http://localhost:2024"

client = get_client(url=API_URL)


async def main():
    async for chunk in client.runs.stream(
        None,
        "chat",
        input={
            "messages": [
                {
                    "role": "user",
                    "content": "What is the capital of France?",
                },
            ]
        },
        stream_mode="messages",
    ):
        print(chunk.data)


asyncio.run(main())
