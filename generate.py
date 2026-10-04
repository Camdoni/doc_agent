"""
04_generate.py

WHAT THIS SCRIPT DOES:
This is where retrieval becomes full RAG. We take a question, retrieve
the most relevant chunks (same as 03_query.py), but instead of just
printing them, we stuff them into a prompt and send that prompt to an
actual LLM running locally via Ollama. The model then writes a real
answer, grounded in your documents instead of just its own training
data.

WHY WE'RE USING OLLAMA:
Ollama is a local server that runs an LLM for you and exposes a simple
HTTP API on your machine (http://localhost:11434). We just send it a
prompt as JSON over HTTP and get text back, no need to manually load
model weights or manage GPU memory ourselves in this script.

REQUIREMENT: Ollama must be installed and running, with a model pulled.
See the README for setup steps. Run `ollama pull llama3.2` first if you
haven't.
"""

import json
import requests
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# --- Retrieval config (same as 03_query.py) ---
INDEX_FILE = "index.faiss"
META_FILE = "chunks_meta.json"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 3  # fewer chunks than before -- we're paying per-token now, keep the prompt lean

# --- Generation config ---
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"

# Distance above this is treated as "not actually relevant." Tune this
# based on what you saw while testing 03_query.py -- if your genuinely
# good matches were landing around 1.0-1.3 and garbage matches were
# landing around 1.6+, somewhere around 1.4-1.5 is a reasonable cutoff.
DISTANCE_THRESHOLD = 1.5


def retrieve(query, model, index, chunks, k=TOP_K):
    """Same retrieval logic as 03_query.py -- embed the query, search FAISS,
    map results back to chunk text. See that file's walkthrough for the
    line-by-line explanation."""
    query_vector = model.encode([query], normalize_embeddings=True)
    query_vector = np.array(query_vector, dtype="float32")
    k = min(k, index.ntotal)
    distances, indices = index.search(query_vector, k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        if dist > DISTANCE_THRESHOLD:
            continue  # too far to be a real match, don't pass it to the LLM
        chunk = chunks[idx]
        results.append({"distance": float(dist), "source": chunk["source"], "text": chunk["text"]})
    return results


def build_prompt(query, retrieved_chunks):
    """
    Builds the actual text we send to the LLM.

    The structure matters: we explicitly tell the model to only use the
    provided context, and to say so if the context doesn't answer the
    question. Without that instruction, the model will happily fall back
    on its own general knowledge, which defeats the point of grounding
    answers in YOUR documents specifically.
    """
    if not retrieved_chunks:
        context = "(No relevant document content was found.)"
    else:
        # Number each chunk and label its source, so the model can
        # reference where information came from if useful.
        context_parts = []
        for i, c in enumerate(retrieved_chunks):
            context_parts.append(f"[Source {i+1}: {c['source']}]\n{c['text']}")
        context = "\n\n".join(context_parts)

    prompt = f"""You are answering a question using ONLY the context below.
The context comes from raw lecture notes: short fragments, dashes, and
bullet points instead of full sentences. Treat a fragment like "Node.js -
runtime for async events" as a real, usable answer, not as insufficient
just because it isn't phrased as a complete sentence. Rewrite it into a
proper explanatory sentence for your answer.

Only say "I don't have enough information in the provided documents to answer that"
if the context truly does not mention the topic at all. Do not use outside knowledge
beyond what's in the context. Answer in a full sentence or two.

Context:
{context}

Question: {query}

Answer:"""
    return prompt


def generate_answer(prompt):
    """
    Sends the prompt to Ollama's local API and returns the generated text.

    stream=False means we wait for the full response in one go rather
    than receiving it word-by-word. Simpler to work with for now; a
    streaming version is a nice upgrade later once this works.
    """
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            # Lower temperature = less randomness. For fact-grounded RAG
            # answers we want consistency over creativity, so the same
            # question against the same docs gives (roughly) the same
            # answer each time, instead of varying run to run.
            "options": {"temperature": 0.2},
        },
    )
    response.raise_for_status()  # throws an error if Ollama returned a failure status
    data = response.json()
    return data["response"]


def main():
    index = faiss.read_index(INDEX_FILE)
    with open(META_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("RAG agent ready. Type a question (or 'quit' to exit).\n")
    while True:
        query = input("Question: ").strip()
        if query.lower() in ("quit", "exit"):
            break
        if not query:
            continue

        retrieved = retrieve(query, embed_model, index, chunks)
        prompt = build_prompt(query, retrieved)

        print("\nThinking...")
        try:
            answer = generate_answer(prompt)
        except requests.exceptions.ConnectionError:
            print("Couldn't reach Ollama. Is it installed and running? "
                  "Try running 'ollama pull llama3.2' in another terminal first.\n")
            continue

        print(f"\nAnswer: {answer}\n")

        if retrieved:
            print("Sources used:")
            for c in retrieved:
                print(f"  - {c['source']} (distance={c['distance']:.3f})")
        else:
            print("(No sources met the relevance threshold.)")
        print()


if __name__ == "__main__":
    main()