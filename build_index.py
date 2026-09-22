"""
02_build_index.py
 
WHAT THIS SCRIPT DOES:
Loads the chunks.json file from step 1, converts each chunk of text into
a vector (an "embedding"), and stores all those vectors in a FAISS index
so we can search them later by meaning instead of exact keyword match.
 
WHAT AN EMBEDDING IS:
An embedding model turns a piece of text into a list of numbers (a
vector), e.g. 384 numbers for the model used here. Texts with similar
meaning end up with vectors that are close together in that 384-
dimensional space. "The dog ran fast" and "The puppy sprinted" will
land near each other even though they share almost no words, because
the model captures meaning, not just word overlap.
 
WHAT FAISS DOES:
FAISS (Facebook AI Similarity Search) is a library that stores vectors
and lets you efficiently find the "nearest" ones to a query vector, i.e.
the chunks whose meaning is closest to your question. We're using the
simplest possible index type here (IndexFlatL2), which just compares
your query to every stored vector directly. That's fine for a few
thousand chunks. If this were millions of chunks you'd want a fancier
index, but flat search is the right place to start.
"""

import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

CHUNKS_FILE = "chunks.json"
INDEX_FILE = "index.faiss"
META_FILE = "chunks.meta.json" # keeps the text/source tied to each vector

# small, fasdt, well-regarded embedding model. Turns text into 384-dimensional vectors.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

def main():
    # Load the chunks
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Loaded {len(chunks)} chunks from '{CHUNKS_FILE}'")

    texts = [c["text"] for c in chunks]

    print(f"Loading embedding model '{EMBEDDING_MODEL_NAME}'...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("Embedding all chunks (slow)...")
    # show_progress_bar gives you a sense of how long this takes on your
    # machine. normalize_embedding=True makes cosine-similarity-style
    # search behave correctly with the L2 index below.
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    embeddings = np.array(embeddings, dtype="float32")
    print(f"Produced embeddings with shape {embeddings.shape}")
    # shape is (num_chunks, 384) -- one 384-number vector per chunk

    # Build the FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension) # flat = brute-force nearest neighbor
    index.add(embeddings)
    print(f"Built FAISS index with {index.ntotal} vectors")

    # Save everything to disk
    faiss.write_index(index, INDEX_FILE)
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    print(f"Saved index to '{INDEX_FILE}' and metadata to '{META_FILE}'")

if __name__ == "__main__":
    main()