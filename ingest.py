"""
ingest.py

WHAT THIS SCRIPT DOES:
Reads a whole Obsidian vault (a folder of .md notes, including subfolders),
cleans the Obsidian-specific syntax, chunks each note, embeds the chunks,
and saves the FAISS index. It replaces running chunk_docs.py and then
build_index.py by hand.

USAGE:
    python ingest.py                      # uses $DOCAGENT_NOTES, else ./docs
    python ingest.py "C:/path/to/vault"   # explicit vault path

WHAT IT SAVES (same names agent.py already loads):
    chunks.json        all chunks
    chunks_meta.json   same chunks, row i here matches vector i in the index
    index.faiss        the vectors
    manifest.json      {note path: last-modified time}, used later to detect edits
"""

import os
import re
import sys
import json

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from chunk_docs import chunk_text  # your paragraph-packing chunker, reused as is

DEFAULT_VAULT = os.environ.get("DOCAGENT_NOTES", "docs")
CHUNKS_FILE = "chunks.json"
INDEX_FILE = "index.faiss"
META_FILE = "chunks_meta.json"
MANIFEST_FILE = "manifest.json"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # must match agent.py

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


def build_chunks(vault):
    """Reads and chunks every note. Returns (chunks, manifest)."""
    chunks = []
    manifest = {}
    for rel_path in find_notes(vault):
        manifest[rel_path] = os.path.getmtime(os.path.join(vault, rel_path))
        note_chunks = chunk_note(rel_path, read_note(vault, rel_path))
        print(f"  {rel_path}: {len(note_chunks)} chunks")
        chunks.extend(note_chunks)
    return chunks, manifest


def build_index(chunks):
    """
    Embeds every chunk and returns a FAISS index. Row i of the index is
    chunks[i], so the order of `chunks` must never change after this.
    """
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    texts = [c["text"] for c in chunks]
    vectors = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    vectors = np.array(vectors, dtype="float32")
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    return index


def ingest(vault):
    """Full rebuild: read notes -> chunk -> embed -> save everything."""
    if not os.path.isdir(vault):
        raise SystemExit(f"Notes folder not found: {vault}")

    print(f"Reading notes from '{vault}'")
    chunks, manifest = build_chunks(vault)
    if not chunks:
        raise SystemExit("No notes with text found. Nothing to index.")

    index = build_index(chunks)

    faiss.write_index(index, INDEX_FILE)
    for path in (CHUNKS_FILE, META_FILE):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2)
    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump({"vault": os.path.abspath(vault), "notes": manifest}, f, indent=2)

    print(f"\nIndexed {len(manifest)} notes into {len(chunks)} chunks.")
    return len(manifest), len(chunks)


def vault_changes(vault):
    """
    Compares the vault to the manifest saved by the last ingest.
    Returns a short reason string if a rebuild is needed, or None if the
    index is up to date. We compare modified times (cheap) instead of
    re-reading every note.
    """
    if not all(os.path.exists(p) for p in (INDEX_FILE, META_FILE, MANIFEST_FILE)):
        return "no index yet"

    with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
        saved = json.load(f)

    if saved.get("vault") != os.path.abspath(vault):
        return "notes folder changed"

    old = saved["notes"]
    new = {n: os.path.getmtime(os.path.join(vault, n)) for n in find_notes(vault)}
    added = [n for n in new if n not in old]
    removed = [n for n in old if n not in new]
    edited = [n for n in new if n in old and new[n] != old[n]]
    if not (added or removed or edited):
        return None
    return f"{len(added)} added, {len(edited)} edited, {len(removed)} removed"


def ensure_index(vault):
    """Called at startup: rebuilds the index only if the notes changed."""
    reason = vault_changes(vault)
    if reason is None:
        return False
    print(f"Notes changed ({reason}). Rebuilding index...")
    ingest(vault)
    return True


if __name__ == "__main__":
    ingest(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VAULT)
