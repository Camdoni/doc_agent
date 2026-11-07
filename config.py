"""
config.py

Decides WHERE things live, so the app works from any folder instead of only
from inside the project folder.

  notes folder (the vault)  -> given by --notes or the DOCAGENT_NOTES env var
  data folder               -> ~/.docagent/<vault name>-<hash>/ (override the
                               base with DOCAGENT_HOME, which Docker will use)

The data folder holds everything the app generates: the FAISS index, the
chunk metadata, the manifest, and the run log. Keeping it OUT of the vault
means Obsidian and cloud sync never see it.
"""

import os
import sys
import json
import hashlib
from pathlib import Path
from urllib.parse import urlparse

SETTINGS_FILE = "config.json"


def ollama_url():
    """
    Where Ollama is listening. Honors the standard OLLAMA_HOST variable
    (e.g. "ollama:11434" when Ollama runs in another Docker container),
    otherwise the normal local address.
    """
    host = os.environ.get("OLLAMA_HOST", "").strip()
    if not host:
        return "http://localhost:11434"
    if "://" not in host:
        host = "http://" + host
    if urlparse(host).port is None:
        host += ":11434"
    return host.replace("0.0.0.0", "localhost").rstrip("/")


OLLAMA_URL = ollama_url()
OLLAMA_IS_REMOTE = bool(os.environ.get("OLLAMA_HOST", "").strip())  # someone pointed us at a specific Ollama


def base_dir():
    """Where docagent keeps everything it generates."""
    return os.environ.get("DOCAGENT_HOME") or str(Path.home() / ".docagent")


def load_settings():
    """Reads the remembered settings, or {} if there are none yet or the file is damaged."""
    try:
        with open(os.path.join(base_dir(), SETTINGS_FILE), "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_settings(settings):
    os.makedirs(base_dir(), exist_ok=True)
    with open(os.path.join(base_dir(), SETTINGS_FILE), "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)


def clean_path(raw):
    """Turns what a person typed or pasted into a real absolute path."""
    # Windows "Copy as path" and drag-and-drop wrap the path in quotes
    return os.path.abspath(os.path.expanduser(raw.strip().strip('"').strip("'")))


def prompt_for_vault():
    """First-run question. Only called when we can talk to a person."""
    print("Welcome to docagent. Let's point it at your notes.")
    print("(Your notes stay on this computer. You only have to do this once.)\n")
    for _ in range(3):
        try:
            raw = input("Path to your notes folder (the one you open in Obsidian): ")
        except EOFError:
            break
        if raw.strip() and os.path.isdir(clean_path(raw)):
            return clean_path(raw)
        print(f"  I can't find a folder at '{raw.strip()}'. Check the path and try again.\n")
    raise SystemExit("No valid notes folder given. Run docagent again when you have the path.")


def resolve_vault(arg=None):
    """
    Finds the notes folder, trying in order:
      1. --notes PATH               (explicit, so a wrong path is an error)
      2. DOCAGENT_NOTES env var     (explicit)
      3. the folder used last time  (remembered in config.json)
      4. asking the person          (first run, terminal only)
    Whatever works is remembered, so next time you can just type `docagent`.
    """
    explicit = arg or os.environ.get("DOCAGENT_NOTES")
    if explicit:
        vault = clean_path(explicit)
        if not os.path.isdir(vault):
            raise SystemExit(f"Notes folder not found: {vault}")
    else:
        remembered = load_settings().get("notes")
        if remembered and os.path.isdir(remembered):
            return remembered
        if not sys.stdin.isatty():
            raise SystemExit("No notes folder given. Use --notes PATH or set DOCAGENT_NOTES.")
        vault = prompt_for_vault()

    save_settings({**load_settings(), "notes": vault})
    return vault


def data_dir_for(vault):
    """
    One data folder per vault. The short hash of the full path keeps two
    vaults that share a folder name (e.g. both called 'notes') apart.
    """
    base = base_dir()
    key = hashlib.sha1(vault.encode("utf-8")).hexdigest()[:8]
    folder = os.path.join(base, f"{os.path.basename(vault)}-{key}")
    os.makedirs(folder, exist_ok=True)
    return folder
