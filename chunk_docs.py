"""
chunk_docs.py
 
WHAT THIS SCRIPT DOES:
Reads every .txt or .md file in a folder, splits each one into small
overlapping chunks of text, and saves all the chunks to a single JSON file.
 
WHY WE CHUNK:
The model can't usefully search or reason over a giant 20-page document
in one shot, and embedding models work best on short passages. So we
break each doc into pieces (~e.g. 300 words) that are small enough to
embed individually and retrieve one at a time later.
 
WHY OVERLAP:
If we cut chunks with zero overlap, a sentence that explains something
important can get split in half at a chunk boundary, and neither half
makes sense on its own. Overlapping the end of one chunk with the start
of the next (e.g. 50 words of overlap) makes it much less likely that a
key idea gets orphaned.
"""

import os
import json
import glob

DOCS_FOLDER = "docs"
OUTPUT_FILE = "chunks.json"
CHUNK_SIZE_WORDS = 300
CHUNK_OVERLAP_WORDS = 50

def read_docs(folder):
	filepaths = glob.glob(os.path.join(folder, "*.txt")) + glob.glob(os.path.join(folder, "*.md"))

	docs = []
	for path in filepaths:
		with open(path, "r", encoding="utf-8") as f:
			text = f.read()
		docs.append((os.path.basename(path), text))

	return docs

def chunk_text(text, chunk_size=CHUNK_SIZE_WORDS, overlap=CHUNK_OVERLAP_WORDS):
	# breaks document into a list of individual words, split on
	# whitespace (e.g. if doc is 1000 words, words is a list of 1000 strings)
	words = text.split()
	chunks = []
	step = chunk_size - overlap
	for start in range(0, len(words), step):
		end = start + chunk_size
		chunk_words = words[start:end]
		if not chunk_words:
			break
		chunks.append(" ".join(chunk_words))
		if end >= len(words):
			break
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
				"id": f"{filename}::chunk_{i}",
				"source": filename,
				"text": chunk,
			})

	with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
		json.dump(all_chunks, f, indent=2)

	print(f"\nSaved {len(all_chunks)} total chunks to '{OUTPUT_FILE}'")

if __name__ == "__main__":
	main()