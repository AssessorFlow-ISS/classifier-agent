FROM python:3.12-slim AS base

WORKDIR /app

# Install af-shared (peer dependency for all agents)
COPY --from=shared . /tmp/af-shared/
RUN pip install --no-cache-dir "/tmp/af-shared[langfuse]" && rm -rf /tmp/af-shared


COPY pyproject.toml README.md .
COPY prompts/ prompts/
COPY src/ src/
COPY tests/ tests/

RUN pip install --no-cache-dir -e . \
    && ln -s /app/prompts /usr/local/lib/python3.12/prompts

EXPOSE 8000

CMD ["uvicorn", "classification_agent.main:app", "--host", "0.0.0.0", "--port", "8000"]
