FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml .
COPY src/ ./src/
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -e .
ENTRYPOINT ["repo-arch"]
