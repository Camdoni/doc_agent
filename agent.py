"""
WHAT THIS SCRIPT DOES:
Turns the RAG pipleine into a simple agent. Before, every question went through retrieval no matter what. Now the model gets a tool called search_docs and decides for itself whether to use it.

This is an agent because:
1. The model can request a tool call instead of answering right away.
2. Our code runs the tool and gives the result back to the model.
3. The model looks at the result and either answers or calls a tool again.
That back-and-forth loop is the whole idea. Everything else is detail.

HOW TOOL CALLING WORKS WITH OLLAMA:
We use Ollama's /api/chat endpoint (instead of /api/generate) because it 
supports a list of messages with roles, and "tools" field where we describe the tools the model is allowed to call. If the model wants a tool, its reply has a "tool_calls" field instead of normal text.
llama3.2 supports this.

LOGGING:
Every run is saved to agent_log.json1 (one JSON object per line). Later
we can turn good entries into fine-tuning data for the LoRA step.
"""

import json
import datetime
import requests
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# retrieval config
INDEX_FILE = "index.faiss"
META_FILE = "chunks_meta.json"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 3
DISTANCE_THRESHOLD = 1.5

# LLM config
OLLAMA_CHAT_URL = "http://localhost:11434/api.chat"
OLLAMA_MODEL = "llama3.2"

# agent config
MAX_STEPS = 4 # safety limit so the loop can never run forever
LOG_FILE = "agent_log.json1"

SYSTEM_PROMPT = """You are an assistant that answers questions about the user's lecture notes.
 
You have one tool: search_docs. Use it whenever the question could be answered by the notes.
Do not use it for greetings or small talk.
 
The notes are short fragments and bullet points, not full sentences. Treat a fragment like
"Node.js - runtime for async events" as a real answer and rewrite it into a clear sentence.
 
Answer only from what search_docs returns. If it returns nothing relevant, say you could not
find that in the notes. Keep answers to a few sentences."""

# This is the description of the tool we hand to the model. The model never
# sees our Python function. It only sees this description, so the name and
# description need to make clear when the tool should be used.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": "Search the user's lecture notes and return the most relevant passages.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A short search query describing what to look for.",
                    }
                },
                "required": ["query"],
            },
        },
    }
]

def load_resources():
    """Loads the FAISS index, chunk metadata, and embedding model once at startup."""
    index = faiss.read_index(INDEX_FILE)
    with open(META_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return index, chunks, embed_model

def search_docs(query, index, chunks, embed_model):
    """
    The actual tool. Same retrieval as before, but it returns one string
    (the model reads text, not Python objects) plus a list of sources for logging.
    """
    query_vector = embed_model.encode([query], normalize_embeddings=True)
    query_vector = np.array(query_vector, dtype="float32")
    k = min(TOP_K, index.ntotal)
    distances, indices = index.search(query_vector, k)

    parts = []
    sources = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == - 1 or dist > DISTANCE_THRESHOLD:
            continue
        chunk = chunks[idx]
        parts.append(f"[{chunk['source']}]\n{chunk['text']}")
        sources.append({"source": chunk["source"], "distance": float(dist)})

    if not parts:
        return "No relevant passages found.", sources
    return "\n\n".join(parts), sources

def call_llm(messages):
    """Sends the convo so far to Ollama and returns the model's reply message."""
    response = requests.post(
        OLLAMA_CHAT_URL,
        json={
            "model": OLLAMA_MODEL,
            "messages": messages,
            "tools": TOOLS,
            "stream": False,
            "options": {"temperature": 0.2},
        },
    )

def run_agent(question, index, chunks, embed_model):
    """
    The agent loop.

    `messages` is the full conversation: system prompt, user question,
    then each model reply and tool result as we go. The model has no memory of its own, so we resend this whole list on every call.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    tool_log = [] # what the agent did, saved to the log file

    for step in range(MAX_STEPS):
        reply = call_llm(messages)
        messages.append(reply) # keep the model's reply in the history

        tool_calls = reply.get("tool_calls")

        # no tool call means the model is giving its final answer
        if not tool_calls:
            return reply.get("content", ""), tool_log

        # the model asked for one or more tool calls. run
        # each on and add the result to the history so the model
        # can read it
        for call in tool_calls:
            name = call["function"]["name"]
            args = call["function"]["arguments"]

            if name == "search_docs":
                search_query = args.get("query", question)
                print(f"  [tool] search_docs(query={search_query!r})")
                result, sources = search_docs(search_query, index, chunks, embed_model)
                tool_log.append({"tool": name, "query": search_query, "sources": sources})
            else:
                result = f"Unknown tool: {name}"

            messages.append({"role": "tool", "tool_name": name, "content": result})

    # if we get here, the model kept calling tools and never answered.
    return "I could not finish answering that. Try rephrasing the question.", tool_log

def write_log(question, answer, tool_log):
    """Appends one line of JSON to the log file for this run."""
    entry = {
        "time": datetime.datetime.now().isoformat(timespec="seconds"),
        "question": question,
        "answer": answer,
        "tool_calls": tool_log,
    }

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

def main():
    index, chunks, embed_model = load_resources()
    print("Agent ready. Type a question (or 'quit' to exit).")

    while True:
        question = input("Question: ").strip()
        if question.lower() in ("quit", "exit"):
            break
        if not question:
            continue

        try:
            answer, tool_log = run_agent(question, index, chunks, embed_model)
        except requests.exceptions.ConnectionError:
            print("Couldn't reach Ollama. Make sure it is running.\n")
            continue

        if not tool_log:
            print("  [no tool used]")

        print(f"\nAnswer: {answer}\n")
        write_log(question, answer, tool_log)

if __name__ == "__main__":
    main()
