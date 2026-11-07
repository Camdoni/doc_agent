"""
server.py

Runs docagent as a web server, so other programs or devices can ask questions
over the network instead of using this terminal. It is the same agent as the
terminal chat, with a thin web layer on top:

    GET    /health              is it up, how many notes, which model (no key needed)
    POST   /ask                 {"question": "...", "session_id": "..."} -> one JSON answer
    POST   /ask/stream          same, but the answer streams in as it is written
    POST   /reindex             rebuild the index from the notes now
    DELETE /sessions/{id}       forget one conversation

SESSIONS: the server remembers each conversation by its session_id, so a
client only sends the new question and follow-ups still work.

SECURITY: this exposes your notes. If you set an API key, every request except
/health must send the header  "Authorization: Bearer <key>". The server refuses
to listen on the network at all without a key.
"""

import json
import queue
import secrets
import threading

import requests
import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import agent
from config import resolve_vault, data_dir_for
from ingest import ensure_index, load_embedder
from ollama_setup import ensure_ollama
from watcher import NotesWatcher

LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")
MAX_SESSIONS = 200  # oldest conversations are dropped past this, so memory can't grow forever


def pick_model(installed, preferred):
    """Chooses an installed Ollama model without ever asking a question (a server has no keyboard)."""
    wanted = preferred or agent.DEFAULT_MODEL
    for name in installed:
        if name == wanted or name.startswith(wanted):
            return name
    return installed[0]


def sources_payload(tool_log):
    """What the caller needs to show citations: closest distance per note, plus the raw passages."""
    return {
        "sources": [{"source": s, "distance": d} for s, d in agent.unique_sources(tool_log).items()],
        "passages": [passage for call in tool_log for passage in call["sources"]],
    }


class Engine:
    """Everything the server needs to answer questions: the index, the models, the conversations."""

    def __init__(self, vault, model_name):
        self.data_dir = data_dir_for(vault)
        installed = ensure_ollama(model_name, agent.DEFAULT_MODEL, auto_pull=True)
        self.model = pick_model(installed, model_name)
        try:
            ensure_index(vault, self.data_dir)
            self.corpus = agent.load_index(self.data_dir)  # (index, chunks), swapped in one step on change
        except SystemExit:
            # No notes yet. Start anyway: the watcher indexes them as soon as they appear,
            # which is what you want when the notes are copied in after the server starts.
            print("No notes found yet. Waiting for notes to appear...", flush=True)
            self.corpus = None
        self.embedder = load_embedder()
        self.sessions = {}
        self._lock = threading.Lock()  # guards sessions and the log file; requests run on many threads
        self.watcher = NotesWatcher(vault, self.data_dir, self.embedder, self._notes_updated)

    def start(self):
        self.watcher.start()

    def _notes_updated(self, reason):
        self.corpus = agent.load_index(self.data_dir)
        print(f"[notes updated: {reason}]", flush=True)

    def stats(self):
        chunks = self.corpus[1] if self.corpus else []
        return {"status": "ok", "model": self.model,
                "notes": len({c["source"] for c in chunks}), "chunks": len(chunks)}

    def reindex(self):
        try:
            self.watcher.rebuild()
        except SystemExit:
            return  # still no notes; nothing to index
        self.corpus = agent.load_index(self.data_dir)

    def clear(self, session_id):
        with self._lock:
            self.sessions.pop(session_id, None)

    def ask(self, question, session_id, on_token=None, on_tool=None):
        if self.corpus is None:
            return ("I don't have any notes yet. Add some to the notes folder and I'll pick them up "
                    "within a few seconds."), []
        with self._lock:
            history = list(self.sessions.get(session_id, []))
        index, chunks = self.corpus
        answer, tool_log = agent.run_agent(question, index, chunks, self.embedder, self.model,
                                           history, on_token, on_tool or (lambda query: None))
        with self._lock:
            turns = history + [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
            self.sessions.pop(session_id, None)  # re-insert so recently used conversations sort last
            self.sessions[session_id] = turns[-2 * agent.HISTORY_TURNS:]
            while len(self.sessions) > MAX_SESSIONS:
                self.sessions.pop(next(iter(self.sessions)))
            agent.write_log(self.data_dir, question, answer, tool_log)
        return answer, tool_log


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    session_id: str = Field(default="default", max_length=100)


def create_app(engine, api_key=None):
    app = FastAPI(title="docagent")

    def check_key(authorization: str = Header(default="")):
        if api_key is None:
            return
        expected = f"Bearer {api_key}".encode("utf-8")
        # compare_digest takes the same time however many characters match, so the key can't be guessed by timing
        if not secrets.compare_digest(authorization.encode("utf-8"), expected):
            raise HTTPException(status_code=401, detail="Missing or wrong API key.")

    @app.get("/health")
    def health():
        return engine.stats()

    @app.post("/ask", dependencies=[Depends(check_key)])
    def ask(req: AskRequest):
        tools = []
        try:
            answer, tool_log = engine.ask(req.question, req.session_id, on_tool=tools.append)
        except requests.exceptions.RequestException:
            raise HTTPException(status_code=503, detail="The server couldn't reach Ollama.")
        return {"answer": answer, "tools": tools, **sources_payload(tool_log)}

    @app.post("/ask/stream", dependencies=[Depends(check_key)])
    def ask_stream(req: AskRequest):
        events = queue.Queue()

        def work():
            try:
                answer, tool_log = engine.ask(
                    req.question, req.session_id,
                    on_token=lambda text: events.put({"type": "token", "text": text}),
                    on_tool=lambda query: events.put({"type": "tool", "query": query}),
                )
                events.put({"type": "done", "answer": answer, **sources_payload(tool_log)})
            except requests.exceptions.RequestException:
                events.put({"type": "error", "message": "The server couldn't reach Ollama."})
            except Exception as error:
                events.put({"type": "error", "message": f"Something went wrong: {error}"})
            events.put(None)  # tells the sender below that the answer is finished

        threading.Thread(target=work, daemon=True).start()

        def send():
            while (event := events.get()) is not None:
                yield json.dumps(event) + "\n"  # one JSON object per line

        return StreamingResponse(send(), media_type="application/x-ndjson")

    @app.post("/reindex", dependencies=[Depends(check_key)])
    def reindex():
        engine.reindex()
        return engine.stats()

    @app.delete("/sessions/{session_id}", dependencies=[Depends(check_key)])
    def clear_session(session_id: str):
        engine.clear(session_id)
        return {"cleared": session_id}

    return app


def serve(notes, host, port, model_name, api_key):
    if host not in LOOPBACK_HOSTS and not api_key:
        raise SystemExit(
            f"Refusing to listen on {host} without an API key, because this server hands out your notes.\n"
            "Set one with  --api-key SECRET  or the DOCAGENT_API_KEY variable."
        )
    vault = resolve_vault(notes)
    engine = Engine(vault, model_name)
    engine.start()
    stats = engine.stats()
    print(f"docagent server on http://{host}:{port}   notes: {vault}")
    print(f"{stats['notes']} notes, {stats['chunks']} chunks, model {stats['model']}, "
          f"API key {'required' if api_key else 'not required (this computer only)'}", flush=True)
    uvicorn.run(create_app(engine, api_key), host=host, port=port, log_level="warning")
