FROM python:3.12-slim

WORKDIR /app

ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ENV PIP_DEFAULT_TIMEOUT=600
ENV PIP_RETRIES=15

RUN pip install --upgrade pip && pip install poetry

COPY pyproject.toml poetry.lock /app/

RUN pip install torch --extra-index-url https://download.pytorch.org/whl/cpu

RUN poetry install --no-root --no-interaction --no-ansi

COPY . /app/

EXPOSE 8000

CMD ["poetry", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
