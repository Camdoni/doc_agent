"""
ingest.py

WHAT THIS SCRIPT DOES:
Reads a whole Obsidian vault (a folder of .md notes, including subfolders),
cleans the Obsidian-specific syntax, chunks each note, embeds the chunks,
and saves the FAISS index. It replaces running chunk_docs.py and then
build_index.py by hand.

USAGE:
    python ingest.py "C:/path/to/vault"   # or set DOCAGENT_NOTES and omit the path

WHAT IT SAVES (into the data folder from config.py, not the vault):
    chunks_meta.json   all chunks, row i here matches vector i in the index
    index.faiss        the vectors
    manifest.json      {note path: last-modified time}, used to detect edits
"""

import os
import re
import sys
import json
import hashlib

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from chunk_docs import chunk_text  # your paragraph-packing chunker, reused as is
from config import resolve_vault, data_dir_for

INDEX_FILE = "index.faiss"
META_FILE = "chunks_meta.json"
MANIFEST_FILE = "manifest.json"
CACHE_FILE = "embed_cache.npz"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # agent.py loads it through load_embedder()

NOTE_EXTENSIONS = (".md", ".txt")


def find_notes(vault):
    """
    Walks the vault and returns note paths relative to the vault, using
    forward slashes (e.g. "CS/Lecture 8.md"). Folders that start with a dot
    (.obsidian, .trash, .git) hold app settings, not notes, so we skip them.
    """
    notes = []
    for folder, subfolders, files in os.walk(vault):
        # editing subfolders in place tells os.walk not to enter them
        subfolders[:] = [d for d in subfolders if not d.startswith(".")]
        for name in files:
            if name.lower().endswith(NOTE_EXTENSIONS):
                full = os.path.join(folder, name)
                notes.append(os.path.relpath(full, vault).replace(os.sep, "/"))
    return sorted(notes)


def strip_frontmatter(text):
    """
    Obsidian notes can start with a YAML block between two '---' lines
    (tags, dates, aliases). That is metadata, not content, so drop it.
    """
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            return text[end + 4:].lstrip("\n")
    return text


def clean_obsidian(text):
    """
    Turns Obsidian syntax into plain text the embedding model understands.
      ![[image.png]]      -> removed (an embed, no text to search)
      [[Note|shown text]] -> shown text
      [[Note]]            -> Note
      %% hidden text %%  -> removed (Obsidian comments)
    """
    text = re.sub(r"%%.*?%%", "", text, flags=re.DOTALL)
    text = re.sub(r"!\[\[.*?\]\]", "", text)
    text = re.sub(r"\[\[([^\]|]*)\|([^\]]*)\]\]", r"\2", text)
    text = re.sub(r"\[\[([^\]]*)\]\]", r"\1", text)
    return text


def split_by_heading(text):
    """
    Splits a note into (heading, body) sections at markdown headings
    ('# ...' through '###### ...'). Text before the first heading gets
    the heading "". We chunk each section separately, so a chunk never
    mixes two headings, and we can label the chunk with its heading.
    """
    sections = []
    heading = ""
    lines = []
    for line in text.split("\n"):
        match = re.match(r"^#{1,6}\s+(.*\S)\s*$", line)
        if match:
            sections.append((heading, "\n".join(lines)))
            heading = match.group(1)
            lines = []
        else:
            lines.append(line)
    sections.append((heading, "\n".join(lines)))
    return [(h, body) for h, body in sections if body.strip()]


def read_note(vault, rel_path):
    """Reads one note. errors='replace' means a stray bad byte can't crash the run."""
    with open(os.path.join(vault, rel_path), "r", encoding="utf-8-sig", errors="replace") as f:
        return f.read()


def chunk_note(rel_path, text):
    """Cleans one note and returns its chunk dicts."""
    text = clean_obsidian(strip_frontmatter(text))
    title = os.path.splitext(os.path.basename(rel_path))[0]
    chunks = []
    for heading, body in split_by_heading(text):
        for piece in chunk_text(body):
            # prefix the note title and heading so a fragment like
            # "- runtime for async events" still says what it is about
            label = title if not heading else f"{title} > {heading}"
            chunks.append({
                "id": f"{rel_path}::chunk_{len(chunks)}",
                "source": rel_path,
                "heading": heading,
                "text": f"[{label}]\n{piece}",
            })
    return chunks


def scan_notes(vault):
    """{note path: last-modified time} for every note in the vault right now."""
    return {n: os.path.getmtime(os.path.join(vault, n)) for n in find_notes(vault)}


def build_chunks(vault, verbose=True):
    """Reads and chunks every note. Returns (chunks, manifest)."""
    chunks = []
    manifest = scan_notes(vault)
    for rel_path in manifest:
        note_chunks = chunk_note(rel_path, read_note(vault, rel_path))
        if verbose:
            print(f"  {rel_path}: {len(note_chunks)} chunks")
        chunks.extend(note_chunks)
    return chunks, manifest


def load_embedder():
    """
    Loads the embedding model from the local cache. Only if it has never been
    downloaded do we contact the Hugging Face Hub (first run only), which is
    also what triggers the "unauthenticated requests" warning.
    """
    try:
        return SentenceTransformer(EMBEDDING_MODEL_NAME, local_files_only=True)
    except OSError:
        print("Downloading the embedding model (first run only)...")
        return SentenceTransformer(EMBEDDING_MODEL_NAME)


def embed_texts(texts, data_dir, embedder=None, verbose=True):
    """
    Turns texts into vectors, but only computes the ones it has not seen before.
    Embedding is the slow part of indexing. Each vector is cached under a hash
    of its text, so when you edit one note only that note's chunks are
    re-embedded and everything else is reused from the cache file.
    """
    keys = [hashlib.sha1(f"{EMBEDDING_MODEL_NAME}\0{t}".encode("utf-8")).hexdigest() for t in texts]
    cache_path = os.path.join(data_dir, CACHE_FILE)

    cache = {}
    if os.path.exists(cache_path):
        try:
            with np.load(cache_path) as saved:
                cache = dict(zip(saved["keys"].tolist(), saved["vectors"]))
        except Exception:  # a damaged cache just means we recompute
            cache = {}

    missing = [i for i, k in enumerate(keys) if k not in cache]
    if missing:
        model = embedder or load_embedder()
        fresh = model.encode([texts[i] for i in missing], show_progress_bar=verbose and len(missing) > 50,
                             normalize_embeddings=True)
        for i, vector in zip(missing, fresh):
            cache[keys[i]] = np.asarray(vector, dtype="float32")

    vectors = np.array([cache[k] for k in keys], dtype="float32")
    np.savez(cache_path, keys=np.array(keys), vectors=vectors)  # keep only what is still in use
    if verbose:
        print(f"Embedded {len(missing)} new chunks, reused {len(keys) - len(missing)} from the cache.")
    return vectors


def build_index(chunks, data_dir, embedder=None, verbose=True):
    """
    Embeds every chunk and returns a FAISS index. Row i of the index is
    chunks[i], so the order of `chunks` must never change after this.
    """
    vectors = embed_texts([c["text"] for c in chunks], data_dir, embedder, verbose)
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    return index


def ingest(vault, data_dir, embedder=None, verbose=True):
    """Full rebuild: read notes -> chunk -> embed -> save into data_dir."""
    if not os.path.isdir(vault):
        raise SystemExit(f"Notes folder not found: {vault}")

    if verbose:
        print(f"Reading notes from '{vault}'")
    chunks, manifest = build_chunks(vault, verbose)
    if not chunks:
        raise SystemExit("No notes with text found. Nothing to index.")

    index = build_index(chunks, data_dir, embedder, verbose)

    faiss.write_index(index, os.path.join(data_dir, INDEX_FILE))
    with open(os.path.join(data_dir, META_FILE), "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)
    with open(os.path.join(data_dir, MANIFEST_FILE), "w", encoding="utf-8") as f:
        json.dump({"vault": os.path.abspath(vault), "notes": manifest}, f, indent=2)

    if verbose:
        print(f"\nIndexed {len(manifest)} notes into {len(chunks)} chunks.")
    return len(manifest), len(chunks)


def vault_changes(vault, data_dir):
    """
    Compares the vault to the manifest saved by the last ingest.
    Returns a short reason string if a rebuild is needed, or None if the
    index is up to date. We compare modified times (cheap) instead of
    re-reading every note.
    """
    files = [os.path.join(data_dir, n) for n in (INDEX_FILE, META_FILE, MANIFEST_FILE)]
    if not all(os.path.exists(p) for p in files):
        return "no index yet"

    with open(os.path.join(data_dir, MANIFEST_FILE), "r", encoding="utf-8") as f:
        saved = json.load(f)

    if saved.get("vault") != os.path.abspath(vault):
        return "notes folder changed"

    old = saved["notes"]
    new = scan_notes(vault)
    added = [n for n in new if n not in old]
    removed = [n for n in old if n not in new]
    edited = [n for n in new if n in old and new[n] != old[n]]
    if not (added or removed or edited):
        return None
    return f"{len(added)} added, {len(edited)} edited, {len(removed)} removed"


def ensure_index(vault, data_dir, embedder=None):
    """Called at startup and when switching notes: rebuilds the index only if the notes changed."""
    reason = vault_changes(vault, data_dir)
    if reason is None:
        return False
    print(f"Notes changed ({reason}). Rebuilding index...")
    ingest(vault, data_dir, embedder)
    return True


if __name__ == "__main__":
    vault_path = resolve_vault(sys.argv[1] if len(sys.argv) > 1 else None)
    ingest(vault_path, data_dir_for(vault_path))
