"""
client.py

`docagent connect`: a terminal chat that talks to a running docagent server
instead of loading any models itself. It looks like the normal chat but only
needs the `requests` library.
"""

import json
import uuid

import requests

HELP_TEXT = """Commands:
  /help      show this list
  /reindex   ask the server to rebuild its index from the notes now
  /sources   show the passages used for the last answer
  /clear     forget the conversation and start fresh
  /quit      exit (or type quit)
Anything else is treated as a question about the notes."""


def show_sources_line(sources):
    if sources:
        print("Sources: " + ", ".join(f"{s['source']} ({s['distance']:.2f})" for s in sources) + "\n")


def show_passages(passages):
    if not passages:
        print("No sources yet. Ask a question about the notes first.\n")
    for p in passages:
        snippet = p["text"].replace("\n", " ")
        snippet = snippet if len(snippet) <= 240 else snippet[:240] + "..."
        print(f"  {p['source']} (distance {p['distance']:.2f})\n    {snippet}\n")


def explain_http_error(response):
    if response.status_code == 401:
        return "The server says the API key is missing or wrong. Use --api-key or set DOCAGENT_API_KEY."
    if response.status_code == 503:
        return "The server couldn't reach Ollama."
    return f"The server returned an error ({response.status_code})."


def ask(url, headers, session_id, question):
    """Sends one question and prints the answer as it streams in. Returns the 'done' event, or None."""
    printed = False
    with requests.post(f"{url}/ask/stream", json={"question": question, "session_id": session_id},
                       headers=headers, stream=True, timeout=None) as response:
        if response.status_code != 200:
            print(explain_http_error(response) + "\n")
            return None
        for line in response.iter_lines():
            if not line:
                continue
            event = json.loads(line)
            if event["type"] == "tool":
                print(f"  [tool] search_docs(query={event['query']!r})")
            elif event["type"] == "token":
                if not printed:
                    print()
                print(event["text"], end="", flush=True)
                printed = True
            elif event["type"] == "error":
                print(f"\n{event['message']}\n")
                return None
            elif event["type"] == "done":
                # replies the server wrote itself (greetings, "not found") never streamed
                print("\n" if printed else f"\n{event['answer']}\n")
                show_sources_line(event["sources"])
                return event
    return None


def connect(url, api_key):
    url = url.rstrip("/")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        health = requests.get(f"{url}/health", timeout=5).json()
    except (requests.exceptions.RequestException, ValueError):
        raise SystemExit(f"Couldn't reach a docagent server at {url}.\nStart one with:  docagent serve")

    print()
    print(f"  docagent  -  connected to {url}")
    print(f"  {health['notes']} notes, {health['chunks']} chunks")
    print(f"  model: {health['model']} (runs on the server)")
    print("  /help for commands, /quit to exit")
    print()

    session_id = uuid.uuid4().hex  # the server keys the conversation memory on this
    last_passages = []

    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue

        if question.startswith("/") or question.lower() in ("quit", "exit"):
            command = question.lower().lstrip("/")
            if command in ("quit", "exit"):
                break
            elif command == "help":
                print(HELP_TEXT + "\n")
            elif command == "sources":
                show_passages(last_passages)
            elif command == "clear":
                requests.delete(f"{url}/sessions/{session_id}", headers=headers, timeout=10)
                session_id = uuid.uuid4().hex
                print("Started a fresh conversation.\n")
            elif command == "reindex":
                response = requests.post(f"{url}/reindex", headers=headers, timeout=None)
                if response.status_code == 200:
                    stats = response.json()
                    print(f"Reindexed: {stats['notes']} notes, {stats['chunks']} chunks.\n")
                else:
                    print(explain_http_error(response) + "\n")
            else:
                print(f"Unknown command '{question}'. Type /help.\n")
            continue

        try:
            done = ask(url, headers, session_id, question)
        except requests.exceptions.RequestException:
            print(f"Lost the connection to the server at {url}.\n")
            continue
        if done:
            last_passages = done["passages"]
