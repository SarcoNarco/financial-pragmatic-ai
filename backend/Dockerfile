ARG RUNTIME_PLATFORM=linux/amd64
FROM --platform=${RUNTIME_PLATFORM} python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/tmp/hf_cache \
    PORT=7860

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["sh", "-c", "uvicorn api.server:app --host 0.0.0.0 --port ${PORT:-7860}"]
