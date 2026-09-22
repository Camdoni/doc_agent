"""
03_query.py
 
WHAT THIS SCRIPT DOES:
Takes a question, embeds it with the SAME embedding model used in build_index.py,
and asks FAISS for the chunks whose vectors are closest to the question's
vector. This is retrieval -- the "R" in RAG. There's no LLM call in this
script yet.
 
WHY THE QUERY USES THE SAME MODEL:
The whole idea only works if the query and the chunks live in the same
vector space. If you embedded chunks with one model and the query with a
different model, the distances between them would be meaningless -- like
comparing GPS coordinates to Cartesian coordinates and expecting the
numbers to line up.
"""
 
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
 
INDEX_FILE = "index.faiss"
META_FILE = "chunks_meta.json"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # must match 02_build_index.py
 
TOP_K = 5  # how many chunks to retrieve per query
 
 
def retrieve(query, model, index, chunks, k=TOP_K):
    """
    Embeds `query`, searches the FAISS index, and returns the top-k
    matching chunks along with their distance scores.
 
    Lower distance = more similar (we're using L2 / Euclidean distance
    between normalized vectors, so smaller numbers mean "closer meaning").
    """
    query_vector = model.encode([query], normalize_embeddings=True)
    query_vector = np.array(query_vector, dtype="float32")
 
    # Never ask FAISS for more matches than actually exist in the index.
    # If you request k=5 but only have 1 vector stored, FAISS pads the
    # extra slots with index -1 and a max-float distance as placeholders.
    k = min(k, index.ntotal)
 
    # index.search returns two arrays:
    #   distances: how far each match is from the query
    #   indices: the row number of each match in the original chunks list
    distances, indices = index.search(query_vector, k)
 
    results = []
    for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        if idx == -1:
            # Padding slot, FAISS ran out of real matches. Skip it.
            continue
        chunk = chunks[idx]
        results.append({
            "rank": rank + 1,
            "distance": float(dist),
            "source": chunk["source"],
            "text": chunk["text"],
        })
    return results
 
 
def main():
    # --- Load the saved index and metadata from step 2 ---
    index = faiss.read_index(INDEX_FILE)
    with open(META_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)
 
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
 
    print("Index loaded. Type a question (or 'quit' to exit).\n")
    while True:
        query = input("Question: ").strip()
        if query.lower() in ("quit", "exit"):
            break
        if not query:
            continue
 
        results = retrieve(query, model, index, chunks)
 
        print(f"\nTop {len(results)} matches:")
        for r in results:
            print(f"  [{r['rank']}] (distance={r['distance']:.3f}) "
                  f"from {r['source']}")
            print(f"      {r['text'][:200]}...")
        print()
 
 
if __name__ == "__main__":
    main()