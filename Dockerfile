FROM python:3.12-slim

WORKDIR /app

# Install build tools for fuzzy
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m nltk.downloader punkt stopwords

COPY . .

EXPOSE 5000

CMD ["python", "backend.py"]