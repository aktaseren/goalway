FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY gateway ./gateway
COPY config ./config
COPY static ./static

# SQLite RAG/feedback stores live on a mounted volume in OpenShift
ENV RAG_VECTOR_DB_PATH=/app/data/vectors.db

RUN mkdir -p /app/data

EXPOSE 8080

CMD ["uvicorn", "gateway.main:app", "--host", "0.0.0.0", "--port", "8080"]
