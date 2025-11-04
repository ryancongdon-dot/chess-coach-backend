FROM python:3.11-slim

# Install Stockfish inside the container
RUN apt-get update && \
    apt-get install -y --no-install-recommends stockfish && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py /app/app.py

ENV PORT=8000 \
    HOST=0.0.0.0 \
    STOCKFISH_PATH=/usr/bin/stockfish \
    STOCKFISH_SECONDS=1.4 \
    USE_CLOUD=0 \
    USE_EXPLORER=1

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
