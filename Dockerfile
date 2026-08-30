# Single stage. There is no frontend build to isolate (ADR-0029), so a
# multi-stage image would add complexity without removing anything.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# libgomp is LightGBM's OpenMP runtime. Without it the import succeeds and the
# first fit fails, which is a confusing way to discover a missing system
# library.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies before source, so a code change does not reinstall the world.
COPY pyproject.toml README.md ./
COPY src/recovery/__init__.py src/recovery/
RUN pip install --no-deps -e . && pip install -e .

COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
COPY data/external ./data/external

# Build the snapshot at image build time so `docker compose up` serves
# immediately rather than making the reviewer wait through a batch. It also
# proves the whole pipeline runs inside the container, not just the web layer.
RUN python -m recovery.cli console --build --cases 20000 --seed 42

EXPOSE 8000

# Bind to all interfaces: the CLI defaults to 127.0.0.1, which is correct on a
# workstation and unreachable from outside a container.
CMD ["uvicorn", "recovery.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
