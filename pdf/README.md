# Generative AI PDF Extractor — Setup Guide

## What You Need Before Starting

- Python 3.11 or 3.12 from python.org
- Docker Desktop from docker.com (for Redis)
- An OpenAI API key from platform.openai.com
- A Pinecone API key and index from app.pinecone.io (create a free Serverless index, region: aws us-east-1, name it anything e.g. `docs`)

---

## Step 1 — Create your .env file

Inside the `pdf/` folder, create a file called `.env` with the following content:

```
OPENAI_API_KEY=your-openai-key
PINECONE_API_KEY=your-pinecone-key
PINECONE_INDEX_NAME=docs
SECRET_KEY=any-random-string-here
SQLALCHEMY_DATABASE_URI=sqlite:///app.db
UPLOAD_URL=http://localhost:8050
REDIS_URI=redis://localhost:6379
```

---

## Step 2 — Create a virtual environment and install dependencies

Run these commands from inside the `pdf/` folder:

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

---

## Step 3 — Initialize the database

```
.venv\Scripts\flask --app app.web init-db
```

---

## Step 4 — Start everything (4 terminals needed)

Open 4 separate terminal windows. In each one, navigate to the project folder first.

**Terminal 1 — Redis (requires Docker Desktop to be running):**
```
docker run -d --name pdf-redis -p 6379:6379 redis
```
Run this once. If it says the container already exists, run `docker start pdf-redis` instead.

**Terminal 2 — File server (stores uploaded PDFs):**
```
cd local-do-files
python app.py
```

**Terminal 3 — Flask main server:**
```
cd pdf
.venv\Scripts\flask --app app.web run --debug
```

**Terminal 4 — Celery worker (processes PDFs into embeddings):**
```
cd pdf
.venv\Scripts\celery -A app.celery.worker worker --loglevel=info --pool=solo
```

The `--pool=solo` flag is required on Windows.

---

## Step 5 — Open the app

Go to http://127.0.0.1:5000 in your browser.

1. Register an account
2. Upload a PDF
3. Wait a few seconds for embeddings to be created (watch Terminal 4 for logs)
4. Click on the PDF, start a new conversation, and ask questions

---

## How It Works

Your uploaded PDF gets split into chunks and converted to embeddings by OpenAI, then stored in Pinecone. When you ask a question, the app searches Pinecone for relevant chunks, passes them to the LLM with your question, and streams back an answer grounded in your document.

The thumbs up/down feedback updates scores in Redis. Better-scoring LLM/retriever/memory combinations get selected more often in future chats.

---

## Troubleshooting

**Celery worker crashes on Windows** — make sure you use `--pool=solo`

**Redis not connecting** — make sure Docker Desktop is open and running before starting Redis

**PDF uploads fail** — make sure the file server (Terminal 2) is running on port 8050

**AI says it doesn't have the document** — the Celery worker may not have processed the PDF yet, check Terminal 4 for errors
