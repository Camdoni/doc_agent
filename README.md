# doc_agent
A RAG agent over your own notes

Ask questions about your Obsidian (or any markdown) notes from the terminal. Answers come only from your notes, with the sources shown. Everything runs on your own computer. No GPU needed.

## Setup
1. Install [Ollama](https://ollama.com/download) (it runs the AI model locally).
2. Install docagent: `pip install .` (from this folder)
3. Run `docagent`

The first run asks where your notes are and remembers it. If Ollama isn't running, docagent starts it. If the model isn't downloaded, docagent offers to download it.

## Using it
Ask questions in plain English. Follow-up questions work. Edit a note in Obsidian and it is searchable a few seconds later.

Commands: `/help`, `/model`, `/reindex`, `/sources`, `/clear`, `/quit`.

## With Docker (nothing to install except Docker)
```
NOTES_DIR=/path/to/your/notes docker compose run --rm docagent
```
The first run downloads the AI model (about 2 GB) into a Docker volume. Your notes are mounted read-only. Have an NVIDIA GPU? Uncomment the GPU lines in `docker-compose.yml`.

## Optional: fine-tune on your notes (needs an NVIDIA GPU)
`make_dataset.py` builds training data from your notes. This is optional and the app works fully without it.
