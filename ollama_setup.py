"""
ollama_setup.py

Makes sure Ollama (the program that runs the AI model on your computer) is
ready before docagent starts, and fixes what it can by itself:

  not installed        -> tells you where to get it
  installed, not running -> starts it for you
  model not downloaded -> offers to download it
"""

import os
import sys
import json
import time
import shutil
import subprocess

import requests

from config import OLLAMA_URL, OLLAMA_IS_REMOTE

NOT_INSTALLED_MESSAGE = """Ollama isn't installed. It's the free program that runs the AI model on your own computer.
Download it from https://ollama.com/download, install it, then run docagent again."""


def installed_models():
    """Names of the models Ollama has downloaded. Raises a requests error if Ollama isn't running."""
    response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
    response.raise_for_status()
    return [m["name"] for m in response.json()["models"]]


def start_ollama():
    """Starts `ollama serve` in the background and waits up to 10 seconds for it. Returns True if it came up."""
    extra = {}
    if os.name == "nt":
        # keep it alive after docagent exits, and don't pop up a console window
        extra["creationflags"] = subprocess.DETACHED_PROCESS | subprocess.CREATE_NO_WINDOW
    else:
        extra["start_new_session"] = True
    subprocess.Popen(
        ["ollama", "serve"],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        **extra,
    )
    for _ in range(20):
        time.sleep(0.5)
        try:
            installed_models()
            return True
        except requests.exceptions.RequestException:
            continue
    return False


def ask_yes(question):
    """Y/n question. Defaults to yes. Answers 'no' when nobody is at the keyboard."""
    if not sys.stdin.isatty():
        return False
    return input(f"{question} [Y/n] ").strip().lower() in ("", "y", "yes")


def pull_model(name):
    """
    Downloads a model by asking Ollama's server to do it. Going through the
    server (instead of the `ollama` command) means this also works when Ollama
    runs somewhere else, like another Docker container.
    """
    print(f"Downloading '{name}'. This can take a few minutes the first time...")
    try:
        with requests.post(f"{OLLAMA_URL}/api/pull", json={"model": name, "stream": True},
                           stream=True, timeout=None) as response:
            response.raise_for_status()
            status = ""
            for line in response.iter_lines():
                if not line:
                    continue
                event = json.loads(line)
                if "error" in event:
                    raise SystemExit(f"Couldn't download '{name}': {event['error']}")
                if event.get("status") != status:
                    if status:
                        print()  # finish the previous status line
                    status = event.get("status", "")
                total = event.get("total")
                percent = f" {int(100 * event.get('completed', 0) / total)}%" if total else ""
                print(f"\r  {status}{percent}   ", end="", flush=True)
            print()
    except requests.exceptions.RequestException:
        raise SystemExit(f"Couldn't download '{name}'. Check the model name and your internet connection.")


def ensure_ollama(wanted=None, default="llama3.2", auto_pull=False):
    """
    Call once at startup. Returns the list of installed models, or exits with
    a plain-English message explaining the single next step.
    `wanted` is the model asked for with --model, if any.
    `auto_pull` downloads a missing model without asking, for servers that
    have no keyboard to answer with.
    """
    try:
        models = installed_models()
    except requests.exceptions.RequestException:
        if OLLAMA_IS_REMOTE:
            # OLLAMA_HOST was set on purpose, so starting a local Ollama would be the wrong fix
            raise SystemExit(f"Couldn't reach Ollama at {OLLAMA_URL}. Make sure it is running and OLLAMA_HOST is correct.")
        if not shutil.which("ollama"):
            raise SystemExit(NOT_INSTALLED_MESSAGE)
        print("Ollama isn't running. Starting it...")
        if not start_ollama():
            raise SystemExit("Couldn't start Ollama. Open the Ollama app (or run 'ollama serve' in another window), then try again.")
        models = installed_models()

    name = wanted or default
    has_it = any(m == name or m.startswith(name) for m in models) if wanted else bool(models)
    if has_it:
        return models

    if auto_pull or ask_yes(f"The model '{name}' isn't downloaded yet. Download it now?"):
        pull_model(name)
        return installed_models()
    raise SystemExit(f"No problem. Download it later with:  ollama pull {name}")
