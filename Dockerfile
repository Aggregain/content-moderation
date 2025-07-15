FROM python:3.9-slim

WORKDIR /app

COPY . /app

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:${PATH}"

RUN poetry install --no-root

RUN poetry run python -m spacy download en_core_web_sm
RUN poetry run python -m spacy download en_core_web_lg

RUN poetry run python -c "import stanza; stanza.download('ru', verbose=False)"

ENV HF_TOKEN="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"

EXPOSE 8000

CMD ["poetry", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]