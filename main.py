import uuid

from langgraph_sdk import get_sync_client

API_URL = "http://localhost:2024"

client = get_sync_client(url=API_URL)
thread_id = str(uuid.uuid4())


def main():
    client.threads.create(
        thread_id=thread_id,
        graph_id="chat",
        if_exists="do_nothing",
        ttl={"strategy": "delete", "ttl": 1},
    )
    while True:
        try:
            prompt = (
                input(">>> Human: ").encode("utf-8", errors="ignore").decode("utf-8")
            )
        except (KeyboardInterrupt, EOFError):
            client.threads.delete(thread_id=thread_id)
            print("\nThoát.")
            break
        if not prompt.strip():
            continue
        print(">>> AI: ", end="", flush=True)
        for chunk in client.runs.stream(
            thread_id=thread_id,
            assistant_id="chat",
            input={"messages": [{"role": "human", "content": prompt}]},
            stream_mode="messages-tuple",
        ):
            if chunk.event == "messages":
                if chunk.data[0].get("chunk_position") == "last":
                    print()
                    break
                content = chunk.data[0].get("content", "")
                print(content, end="", flush=True)
        print("=" * 80)


if __name__ == "__main__":
    main()
