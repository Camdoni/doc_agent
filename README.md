# doc_agent
A RAG agent over your own notes

Ask questions about your Obsidian (or any markdown) notes from the terminal. Everything runs on your own computer. No GPU needed.

## Setup
1. Install [Ollama](https://ollama.com/download) (it runs the AI model locally).
2. Install docagent: `pip install .` (from this folder)
3. Run `docagent`

The first run asks where your notes are and remembers it. If Ollama isn't running, docagent starts it. If the model isn't downloaded, docagent offers to download it.

## Commands
Inside docagent: `/help`, `/model`, `/reindex`, `/sources`, `/quit`.
