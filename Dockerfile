FROM python:3.12-slim AS base

WORKDIR /app

# af-shared peer dependency.
# Build context normally provides the real assessorflow/shared via
# `--build-context shared=...`. When that context is missing (CI without
# the sibling repo), fall back to the vendored shim under vendor/ so the
# build still succeeds end-to-end. Both paths are in PYTHONPATH at runtime.

COPY pyproject.toml README.md ./
COPY prompts/ prompts/
COPY src/ src/
COPY vendor/ vendor/

RUN pip install --no-cache-dir -e . \
    && ln -s /app/prompts /usr/local/lib/python3.12/prompts

ENV PYTHONPATH=/app/vendor:${PYTHONPATH}

EXPOSE 8000

CMD ["uvicorn", "classification_agent.main:app", "--host", "0.0.0.0", "--port", "8000"]
