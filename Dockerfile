FROM python:3.12-slim

WORKDIR /app

# Install CPU-only torch first. On Linux the default torch download bundles
# gigabytes of NVIDIA libraries that this image never uses.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY agent.py config.py ingest.py chunk_docs.py ollama_setup.py watcher.py ./
RUN pip install --no-cache-dir .

# Bake the embedding model into the image so the first run needs no download.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# /notes is where your notes get mounted, /data is where the index lives.
ENV DOCAGENT_HOME=/data \
    DOCAGENT_NOTES=/notes \
    PYTHONUNBUFFERED=1
VOLUME /data

ENTRYPOINT ["docagent"]
