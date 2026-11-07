"""
cli.py

The `docagent` command. It only decides WHICH mode to start and loads that
mode's code on demand, so `docagent connect` starts fast and doesn't need the
heavy search libraries.

    docagent                      chat in this terminal (default)
    docagent serve                run as a server so other devices can ask questions
    docagent connect --url URL    chat with a running server
"""

import os
import argparse


def main():
    parser = argparse.ArgumentParser(prog="docagent", description="Ask questions about your notes.")
    parser.add_argument("command", nargs="?", choices=["serve", "connect"],
                        help="serve: run as a server. connect: chat with a running server. "
                             "Leave out to chat in this terminal.")
    parser.add_argument("--notes", help="path to your notes folder (or set DOCAGENT_NOTES)")
    parser.add_argument("--model", default=os.environ.get("DOCAGENT_MODEL"),
                        help="Ollama model to use, skips the picker (e.g. llama3.2) (or set DOCAGENT_MODEL)")
    parser.add_argument("--host", default="127.0.0.1", help="serve: address to listen on (default: this computer only)")
    parser.add_argument("--port", type=int, default=8000, help="serve: port to listen on (default 8000)")
    parser.add_argument("--url", default=os.environ.get("DOCAGENT_URL", "http://127.0.0.1:8000"),
                        help="connect: address of the server (or set DOCAGENT_URL)")
    parser.add_argument("--api-key", default=os.environ.get("DOCAGENT_API_KEY"),
                        help="serve/connect: secret key (or set DOCAGENT_API_KEY)")
    args = parser.parse_args()

    if args.command == "connect":
        from client import connect
        connect(args.url, args.api_key)
    elif args.command == "serve":
        try:
            from server import serve
        except ImportError:
            raise SystemExit("Server mode needs a few extra packages. Install them with:  pip install \"docagent[server]\"")
        serve(args.notes, args.host, args.port, args.model, args.api_key)
    else:
        from agent import run_chat
        run_chat(args.notes, args.model)


if __name__ == "__main__":
    main()
