"""
01_chunk_docs.py

WHAT THIS SCRIPT DOES:
Reads every .txt or .md file in a folder, splits each one into small
overlapping chunks of text, and saves all the chunks to a single JSON file.

WHY WE CHUNK:
The model can't usefully search or reason over a giant 20-page document
in one shot, and embedding models work best on short passages. So we
break each doc into pieces that are small enough to embed individually
and retrieve one at a time later.

WHY WE CHUNK ON PARAGRAPH BOUNDARIES (not just word count):
Cutting every N words regardless of content mixes unrelated topics into
one chunk (e.g. npm package management and character encoding history
ending up in the same chunk just because they happened to fall in the
same 300-word window). Instead, we split on blank lines -- your natural
paragraph/section breaks -- and pack whole paragraphs into a chunk up to
a word limit, so a chunk boundary always lands between topics, not in
the middle of one.
"""

import os
import json
import glob

# --- CONFIG: tweak these numbers to see how chunking behavior changes ---
DOCS_FOLDER = "docs"              # folder containing your raw .txt/.md files
OUTPUT_FILE = "chunks.json"       # where the resulting chunks get saved
CHUNK_SIZE_WORDS = 200             # max words per chunk (smaller than before,
                                   # since we're now packing whole paragraphs
                                   # rather than cutting at an arbitrary word count)


def read_docs(folder):
    """
    Loads every .txt and .md file in `folder`.
    Returns a list of (filename, full_text) tuples.
    """
    filepaths = glob.glob(os.path.join(folder, "*.txt")) + \
                glob.glob(os.path.join(folder, "*.md"))

    docs = []
    for path in filepaths:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        docs.append((os.path.basename(path), text))
    return docs


def split_into_paragraphs(text):
    """
    Splits raw text into paragraphs, wherever there's a blank line.
    This is our proxy for "topic boundary" -- your notes naturally have
    blank lines between distinct sections (Node.js intro, single-threading,
    npm packages, character encoding aside, etc), so this respects that
    structure instead of ignoring it.
    """
    import re
    # \n\s*\n matches a blank line (possibly with stray whitespace on it)
    paragraphs = re.split(r"\n\s*\n", text)
    # Strip whitespace and drop any empty paragraphs left over from splitting
    return [p.strip() for p in paragraphs if p.strip()]


def split_oversized_paragraph(paragraph, chunk_size=CHUNK_SIZE_WORDS):
    """
    Fallback for the rare case where a single paragraph is itself longer
    than chunk_size (e.g. that big "Node packages:" block with lots of
    sub-bullets and no blank lines inside it). We fall back to plain
    word-count slicing just for that one paragraph, since we have no
    finer-grained boundary to split on.
    """
    words = paragraph.split()
    pieces = []
    for start in range(0, len(words), chunk_size):
        pieces.append(" ".join(words[start:start + chunk_size]))
    return pieces


def chunk_text(text, chunk_size=CHUNK_SIZE_WORDS):
    """
    Groups paragraphs into chunks, keeping each chunk under chunk_size
    words, WITHOUT splitting a paragraph across two chunks unless that
    single paragraph alone is already too big.

    How the packing works:
    - Go through paragraphs one at a time.
    - Keep adding them to the "current chunk" as long as we're still
      under the word limit.
    - The moment adding the next paragraph would push us over the limit,
      close off the current chunk and start a fresh one.
    - This means a chunk boundary always falls between two topics
      (a paragraph break), never in the middle of one.
    """
    paragraphs = split_into_paragraphs(text)
    chunks = []
    current_chunk_paragraphs = []
    current_word_count = 0

    for paragraph in paragraphs:
        paragraph_word_count = len(paragraph.split())

        # Paragraph is bigger than a whole chunk on its own -- flush
        # whatever we've built up so far, then split this paragraph
        # by itself using the word-count fallback.
        if paragraph_word_count > chunk_size:
            if current_chunk_paragraphs:
                chunks.append("\n\n".join(current_chunk_paragraphs))
                current_chunk_paragraphs = []
                current_word_count = 0
            chunks.extend(split_oversized_paragraph(paragraph, chunk_size))
            continue

        # Adding this paragraph would overflow the current chunk --
        # close the current chunk off first, then start a new one.
        if current_word_count + paragraph_word_count > chunk_size and current_chunk_paragraphs:
            chunks.append("\n\n".join(current_chunk_paragraphs))
            current_chunk_paragraphs = []
            current_word_count = 0

        current_chunk_paragraphs.append(paragraph)
        current_word_count += paragraph_word_count

    # Don't forget whatever's left in progress after the loop ends
    if current_chunk_paragraphs:
        chunks.append("\n\n".join(current_chunk_paragraphs))

    return chunks


def main():
    docs = read_docs(DOCS_FOLDER)
    print(f"Found {len(docs)} documents in '{DOCS_FOLDER}/'")

    all_chunks = []
    for filename, text in docs:
        doc_chunks = chunk_text(text)
        print(f"  {filename}: {len(doc_chunks)} chunks")

        for i, chunk in enumerate(doc_chunks):
            all_chunks.append({
                "id": f"{filename}::chunk_{i}",  # unique id, e.g. "notes.md::chunk_3"
                "source": filename,               # which file this came from
                "text": chunk,                    # the actual chunk text
            })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2)

    print(f"\nSaved {len(all_chunks)} total chunks to '{OUTPUT_FILE}'")


if __name__ == "__main__":
    main()