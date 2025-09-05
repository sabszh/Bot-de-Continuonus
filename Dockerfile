FROM python:3.11-slim

# small, reliable system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /Bot-de-Continuonus

COPY requirements.txt .

# Upgrade pip, install **CPU** torch first from the official CPU index,
# then the rest (includes sentence-transformers)
RUN python -V && pip install --upgrade pip \
 && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.8.0 \
 && pip install --no-cache-dir -r requirements.txt

# Build-time sanity check (ensures s-transformers imports on CPU)
RUN python - <<'PY'
import torch, transformers, sentence_transformers
print("Torch:", torch.__version__, "Transformers:", transformers.__version__, "S-BERT:", sentence_transformers.__version__)
from sentence_transformers import SentenceTransformer
SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("S-BERT model load OK (CPU)")
assert not torch.cuda.is_available(), "This image should be CPU-only."
PY

COPY . /Bot-de-Continuonus

ENV HF_HUB_DISABLE_TELEMETRY=1

ENTRYPOINT ["streamlit", "run", "src/streamlit_app.py", \
            "--server.port=80", \
            "--server.headless=true", \
            "--server.address=0.0.0.0", \
            "--browser.gatherUsageStats=false", \
            "--server.enableStaticServing=true", \
            "--server.fileWatcherType=none", \
            "--client.toolbarMode=viewer"]
