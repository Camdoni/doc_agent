# doc_agent
A RAG agent over your own notes

Ask questions about your Obsidian (or any markdown) notes from the terminal. Answers come only from your notes, with the sources shown. Everything runs on your own computer. No GPU needed.

## Quick start

**Windows:** open PowerShell and paste this one line. It installs anything that's missing (Python, Ollama, docagent):
```
irm https://raw.githubusercontent.com/Camdoni/doc_agent/main/install.ps1 | iex
```
Then open a new PowerShell and type `docagent`. (Want to read the script first? It's `install.ps1` in this repo.)

**Mac or Linux:** install [Python 3.10+](https://www.python.org/downloads/) and [Ollama](https://ollama.com/download), then:
```
pip install pipx
pipx ensurepath
pipx install https://github.com/Camdoni/doc_agent/archive/refs/heads/main.zip
docagent
```

The first run asks where your notes are and remembers it. If Ollama isn't running, docagent starts it. If the model isn't downloaded, docagent offers to download it (about 2 GB). Without a GPU, answers take a few seconds to a minute depending on your computer.

**Developing docagent?** From a clone: `pip install -e .`

## Using it
Ask questions in plain English. Follow-up questions work. Edit a note in Obsidian and it is searchable a few seconds later.

Commands: `/help`, `/change` (switch to another notes folder), `/model`, `/reindex`, `/sources`, `/clear`, `/quit`.

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
Copy your notes in with `kubectl cp my-notes/. <docagent-pod>:/notes/` (find the pod name with `kubectl get pods`). docagent notices them within seconds. On Windows use a relative path like this one, because `kubectl cp` mistakes `C:` for a pod name. The first start downloads the AI model, which takes a few minutes. With minikube, run `minikube image load docagent:latest` after building.

## Optional: fine-tune on your notes (needs an NVIDIA GPU)
`make_dataset.py` builds training data from your notes. This is optional and the app works fully without it.
