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
import hashlib
from pathlib import Path


def resolve_vault(arg=None):
    """Returns the absolute notes folder, or exits with a clear message."""
    vault = arg or os.environ.get("DOCAGENT_NOTES")
    if not vault:
        raise SystemExit("No notes folder given. Use --notes PATH or set DOCAGENT_NOTES.")
    vault = os.path.abspath(os.path.expanduser(vault))
    if not os.path.isdir(vault):
        raise SystemExit(f"Notes folder not found: {vault}")
    return vault


def data_dir_for(vault):
    """
    One data folder per vault. The short hash of the full path keeps two
    vaults that share a folder name (e.g. both called 'notes') apart.
    """
    base = os.environ.get("DOCAGENT_HOME") or str(Path.home() / ".docagent")
    key = hashlib.sha1(vault.encode("utf-8")).hexdigest()[:8]
    folder = os.path.join(base, f"{os.path.basename(vault)}-{key}")
    os.makedirs(folder, exist_ok=True)
    return folder
