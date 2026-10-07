FROM python:3.13

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-install-project 

COPY . .

RUN uv sync --frozen 

# mlflow.db stores absolute host paths; point them at /app
RUN python docker/relocate_mlflow.py /app/mlflow.db /app

EXPOSE 8888 5000

CMD ["sh", "docker/start.sh"]
