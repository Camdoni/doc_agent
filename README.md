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

## Server mode (use your notes from other programs or devices)
```
pip install ".[server]"
docagent serve --api-key PICK-A-SECRET
```
Then, from any terminal that can reach it: `docagent connect --api-key PICK-A-SECRET` (add `--url http://host:8000` for another computer). Programs can call it directly: `POST /ask` for a JSON answer, `POST /ask/stream` for a streamed one, `POST /reindex`, and `GET /health`. Send the key as `Authorization: Bearer PICK-A-SECRET`.

By default it only listens on this computer. To listen on the network (`--host 0.0.0.0`) an API key is required. It also works in Docker: `DOCAGENT_API_KEY=PICK-A-SECRET NOTES_DIR=/path/to/notes docker compose --profile server up server`.

## Kubernetes
The `k8s/` folder runs docagent in server mode next to its own Ollama.
```
docker build -t docagent .
kubectl create secret generic docagent --from-literal=api-key=PICK-A-SECRET
kubectl apply -f k8s/
kubectl port-forward service/docagent 8000:8000
```
Copy your notes in with `kubectl cp ./my-notes <docagent-pod>:/notes/`. docagent notices them within seconds. The first start downloads the AI model, which takes a few minutes. With minikube, run `minikube image load docagent:latest` after building.

## Optional: fine-tune on your notes (needs an NVIDIA GPU)
`make_dataset.py` builds training data from your notes. This is optional and the app works fully without it.
