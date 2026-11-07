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
Every run is saved to agent_log.jsonl (one JSON object per line). Later
we can turn good entries into fine-tuning data for the LoRA step.
"""

import os
import re
import time
import json
import argparse
import datetime
import requests
import numpy as np
import faiss

from config import resolve_vault, data_dir_for, OLLAMA_URL
from ollama_setup import ensure_ollama
from ingest import ensure_index, load_embedder, INDEX_FILE, META_FILE, MANIFEST_FILE
from watcher import NotesWatcher

# retrieval config
TOP_K = 3
DISTANCE_THRESHOLD = 1.5

# LLM config
OLLAMA_CHAT_URL = f"{OLLAMA_URL}/api/chat"
DEFAULT_MODEL = "llama3.2"

# agent config
MAX_STEPS = 4 # safety limit so the loop can never run forever
LOG_FILE = "agent_log.jsonl"  # lives in the data folder, next to the index

SYSTEM_PROMPT = """You are an assistant that answers questions about the user's lecture notes.
 
You have one tool: search_docs. Use it whenever the question could be answered by the notes.
Do not use it for greetings or small talk.
 
The notes are short fragments and bullet points, not full sentences. Treat a fragment like
"Node.js - runtime for async events" as a real answer and rewrite it into a clear sentence.
 
Answer only from what search_docs returns. If it returns nothing relevant, say you could not
find that in the notes. Keep answers to a few sentences.

Earlier messages may be part of the same conversation. If the user asks a follow-up like
"what about the second one?", call search_docs again with a complete query that makes sense on its own."""

# Replies the code gives itself, so they never depend on the small model behaving.
NOT_FOUND = "I couldn't find that in your notes."
EMPTY_REPLY = "I'm not sure how to answer that. Try asking a question about your notes."
SMALL_TALK = {
    "hi": "Hi! Ask me anything about your notes.",
    "hello": "Hello! What would you like to know from your notes?",
    "hey": "Hey! Ask me a question about your notes.",
    "good morning": "Good morning! What would you like to look up in your notes?",
    "good evening": "Good evening! What would you like to look up in your notes?",
    "thanks": "You're welcome! Ask me anything else about your notes.",
    "thank you": "Happy to help. Ask me another question about your notes any time.",
    "who are you": "I'm docagent. I answer questions using only your own notes.",
    "what can you do": "I answer questions using your own notes. Ask about something you wrote down and I'll look it up.",
}
HISTORY_TURNS = 6  # how many past question/answer pairs the model sees for follow-ups

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

def load_index(data_dir):
    """Loads the FAISS index and chunk metadata. Cheap, so /reindex can call it again."""
    index = faiss.read_index(os.path.join(data_dir, INDEX_FILE))
    with open(os.path.join(data_dir, META_FILE), "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return index, chunks

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
        sources.append({"source": chunk["source"], "distance": float(dist), "text": chunk["text"]})

    if not parts:
        return "No relevant passages found.", sources
    return "\n\n".join(parts), sources

def canned_reply(question):
    """Returns a ready-made reply for greetings and thanks, else None. No model call needed."""
    return SMALL_TALK.get(re.sub(r"[^a-z ]", "", question.lower()).strip())

def call_llm(messages, model, on_token=None):
    """
    Sends the convo so far to Ollama and returns the model's reply message.
    If `on_token` is given, the reply is streamed: Ollama sends it in small
    pieces and we pass each piece to on_token as it arrives, so words appear
    on screen while the model is still writing.
    """
    streaming = on_token is not None
    response = requests.post(
        OLLAMA_CHAT_URL,
        json={
            "model": model,
            "messages": messages,
            "tools": TOOLS,
            "stream": streaming,
            "options": {"temperature": 0.2},
        },
        stream=streaming,
    )
    response.raise_for_status()
    if not streaming:
        return response.json()["message"]

    content = ""
    tool_calls = []
    started = False  # have we begun printing yet?
    for line in response.iter_lines():
        if not line:
            continue
        piece = json.loads(line).get("message", {})
        tool_calls.extend(piece.get("tool_calls") or [])
        token = piece.get("content", "")
        if not token:
            continue
        content += token
        if started:
            on_token(token)
        elif len(content.strip()) > 2:
            # Hold back the first few characters. The model sometimes replies
            # with just "{}", and we would rather print a real message than that.
            on_token(content)
            started = True
    if not started and content.strip() not in ("", "{}"):
        on_token(content)  # a very short but real answer, like "Yes."

    reply = {"role": "assistant", "content": content}
    if tool_calls:
        reply["tool_calls"] = tool_calls
    return reply

def run_agent(question, index, chunks, embed_model, model, history=None, on_token=None):
    """
    The agent loop.

    `messages` is the full conversation: system prompt, earlier turns (so
    follow-up questions work), the new question, then each model reply and
    tool result as we go. The model has no memory of its own, so we resend
    this whole list on every call.
    """
    small_talk = canned_reply(question)
    if small_talk:
        return small_talk, []

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += history or []
    messages.append({"role": "user", "content": question})
    tool_log = [] # what the agent did, saved to the log file

    for step in range(MAX_STEPS):
        reply = call_llm(messages, model, on_token)
        messages.append(reply) # keep the model's reply in the history

        tool_calls = reply.get("tool_calls")

        # no tool call means the model is giving its final answer
        if not tool_calls:
            answer = reply.get("content", "").strip()
            return (answer if answer not in ("", "{}") else EMPTY_REPLY), tool_log

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

        # Nothing relevant was found in the notes. Say so ourselves instead of
        # letting the model improvise an answer from its own knowledge.
        if not any(call["sources"] for call in tool_log):
            return NOT_FOUND, tool_log

    # if we get here, the model kept calling tools and never answered.
    return "I could not finish answering that. Try rephrasing the question.", tool_log

def write_log(data_dir, question, answer, tool_log):
    """Appends one line of JSON to the log file for this run."""
    entry = {
        "time": datetime.datetime.now().isoformat(timespec="seconds"),
        "question": question,
        "answer": answer,
        "tool_calls": tool_log,
    }

    with open(os.path.join(data_dir, LOG_FILE), "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

def choose_model(preferred=None):
    """
    Lists installed models. Pressing Enter picks the default.
    If `preferred` is given (from --model), skips the menu and uses it.
    """
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags")
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise SystemExit("Couldn't reach Ollama. Make sure it is running, then try again.")
    models = [m["name"] for m in response.json()["models"]]
    if not models:
        raise SystemExit("Ollama has no models installed. Try: ollama pull llama3.2")

    if preferred:
        for name in models:
            if name == preferred or name.startswith(preferred):
                return name
        raise SystemExit(f"Model '{preferred}' is not installed. Installed: {', '.join(models)}")

    for i, name in enumerate(models, start=1):
        print(f"  {i}. {name}")

    choice = input(f"Pick a model number (Enter for {DEFAULT_MODEL}): ").strip()

    if not choice:
        for name in models:
            if name.startswith(DEFAULT_MODEL):
                return name
        return models[0]

    try:
        return models[int(choice) - 1]
    except (ValueError, IndexError):
        print(f"Invalid choice, using {DEFAULT_MODEL} instead.\n")
        for name in models:
            if name.startswith(DEFAULT_MODEL):
                return name
        return models[0]

HELP_TEXT = """Commands:
  /help      show this list
  /model     pick a different Ollama model
  /reindex   rebuild the index from your notes now
  /sources   show the passages used for the last answer
  /clear     forget the conversation and start fresh
  /quit      exit (or type quit)
Anything else is treated as a question about your notes."""

def index_age(data_dir):
    """Human-readable time since the index was last built, e.g. '3 h ago'."""
    seconds = time.time() - os.path.getmtime(os.path.join(data_dir, MANIFEST_FILE))
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        return f"{int(seconds // 60)} min ago"
    if seconds < 86400:
        return f"{int(seconds // 3600)} h ago"
    return f"{int(seconds // 86400)} d ago"

def print_banner(vault, data_dir, chunks, model):
    """The header you see when the app starts: what it is reading and what is answering."""
    notes = len({c["source"] for c in chunks})
    print()
    print(f"  docagent  -  notes: {vault}")
    print(f"  {notes} notes, {len(chunks)} chunks, indexed {index_age(data_dir)}")
    print(f"  model: {model} (Ollama, runs locally; your notes stay on this machine)")
    print("  /help for commands, /quit to exit")
    print()

def unique_sources(tool_log):
    """Collapses every search in one answer into {note path: closest distance}."""
    best = {}
    for call in tool_log:
        for s in call["sources"]:
            best[s["source"]] = min(s["distance"], best.get(s["source"], s["distance"]))
    return dict(sorted(best.items(), key=lambda item: item[1]))

def sources_line(tool_log):
    """One line listing which notes the answer came from. Empty if no search ran."""
    best = unique_sources(tool_log)
    if not best:
        return ""
    return "Sources: " + ", ".join(f"{name} ({dist:.2f})" for name, dist in best.items())

def print_last_sources(tool_log):
    """/sources: show the actual passages the model was given for the last answer."""
    shown = False
    for call in tool_log:
        for s in call["sources"]:
            shown = True
            snippet = s["text"].replace("\n", " ")
            snippet = snippet if len(snippet) <= 240 else snippet[:240] + "..."
            print(f"  {s['source']} (distance {s['distance']:.2f})\n    {snippet}\n")
    if not shown:
        print("No sources yet. Ask a question about your notes first.\n")

class Streamer:
    """Prints the model's answer piece by piece and remembers whether it printed anything."""
    def __init__(self):
        self.printed = False

    def __call__(self, text):
        if not self.printed:
            print()  # blank line between the [tool] line and the answer
        print(text, end="", flush=True)
        self.printed = True

def main():
    parser = argparse.ArgumentParser(prog="docagent", description="Ask questions about your notes.")
    parser.add_argument("--notes", help="path to your notes folder (or set DOCAGENT_NOTES)")
    parser.add_argument("--model", help="Ollama model to use, skips the picker (e.g. llama3.2)")
    args = parser.parse_args()

    vault = resolve_vault(args.notes)
    ensure_ollama(args.model, DEFAULT_MODEL)  # fail fast, before the slow indexing
    data_dir = data_dir_for(vault)
    ensure_index(vault, data_dir)
    state = {"corpus": load_index(data_dir)}  # (index, chunks), swapped in one step when notes change
    embed_model = load_embedder()
    model = choose_model(args.model)

    print_banner(vault, data_dir, state["corpus"][1], model)

    def notes_updated(reason):
        state["corpus"] = load_index(data_dir)
        print(f"\n  [notes updated: {reason}. The new text is searchable now.]\n> ", end="", flush=True)

    watcher = NotesWatcher(vault, data_dir, embed_model, notes_updated)
    watcher.start()
    last_tool_log = []
    history = []  # earlier questions and answers, so follow-ups like "and the second one?" work

    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue

        # lines starting with "/" are app commands, not questions for the model
        if question.startswith("/") or question.lower() in ("quit", "exit"):
            command = question.lower().lstrip("/")
            if command in ("quit", "exit"):
                break
            elif command == "help":
                print(HELP_TEXT + "\n")
            elif command == "model":
                try:
                    model = choose_model()
                    print(f"Now using {model}\n")
                except SystemExit as e:  # e.g. Ollama not running; stay in the app
                    print(f"{e}\n")
            elif command == "reindex":
                watcher.rebuild(verbose=True)
                state["corpus"] = load_index(data_dir)
                print()
            elif command == "sources":
                print_last_sources(last_tool_log)
            elif command == "clear":
                history = []
                print("Started a fresh conversation.\n")
            else:
                print(f"Unknown command '{question}'. Type /help.\n")
            continue

        streamer = Streamer()
        index, chunks = state["corpus"]
        try:
            answer, tool_log = run_agent(question, index, chunks, embed_model, model, history, streamer)
        except requests.exceptions.ConnectionError:
            print("Couldn't reach Ollama. Make sure it is running.\n")
            continue

        last_tool_log = tool_log
        if not tool_log:
            print("  [no tool used]")

        # If the answer streamed in word by word it is already on screen.
        # Replies the code wrote itself (greetings, "not found") print here.
        print("\n" if streamer.printed else f"\n{answer}\n")
        history += [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
        history = history[-2 * HISTORY_TURNS:]
        line = sources_line(tool_log)
        if line:
            print(f"{line}\n")
        write_log(data_dir, question, answer, tool_log)

if __name__ == "__main__":
    main()
