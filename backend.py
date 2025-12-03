import os
import re
import json
import hashlib
import requests
import uvicorn
from datetime import datetime, timedelta
import asyncio
from bs4 import BeautifulSoup

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
import ollama

WORDPRESS_API_URL = "https://www.montclair.edu/policies/wp-json/wp/v2/policies"
CATEGORY_ID = 5

EMBED_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
PINECONE_DIMENSION = 1536
PINECONE_METRIC = "cosine"

RERANK_MODEL = "llama3.2:1b"
ANSWER_MODEL = "llama3.2:3b"

SYSTEM_PROMPT = "You are a precise assistant. Use ONLY the provided context to answer. If not in context, say 'I don't know'."

AUTO_UPDATE_INTERVAL_HOURS = 168
LAST_UPDATE_FILE = "last_update.json"

scraping_in_progress = False
cfg = None


def clean_text(html):
    txt = BeautifulSoup(html, "html.parser").get_text(" ")
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    query: str
    answer: str


def get_last_update_time():
    if os.path.exists(LAST_UPDATE_FILE):
        try:
            with open(LAST_UPDATE_FILE, "r") as f:
                dt = json.load(f)["last_update"]
                return datetime.fromisoformat(dt)
        except:
            return None
    return None

def save_last_update_time():
    with open(LAST_UPDATE_FILE, "w") as f:
        json.dump({"last_update": datetime.now().isoformat()}, f)

def should_auto_update():
    last = get_last_update_time()
    if last is None:
        return True
    return (datetime.now() - last) > timedelta(hours=AUTO_UPDATE_INTERVAL_HOURS)


def load_env_and_clients():
    from dotenv import load_dotenv
    load_dotenv()

    openai_key = os.getenv("OPENAI_API_KEY")
    pinecone_key = os.getenv("PINECONE_API_KEY")
    pinecone_index = os.getenv("PINECONE_INDEX")

    if not openai_key or not pinecone_key or not pinecone_index:
        raise RuntimeError("Missing required env vars.")

    return {
        "openai": OpenAI(api_key=openai_key),
        "pinecone": Pinecone(api_key=pinecone_key),
        "pinecone_index": pinecone_index
    }


def create_index_if_missing(pc, index_name):
    existing = [x["name"] for x in pc.list_indexes().get("indexes", [])]
    if index_name not in existing:
        pc.create_index(
            name=index_name,
            dimension=PINECONE_DIMENSION,
            metric=PINECONE_METRIC,
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

def ensure_index(pc, index_name):
    return pc.Index(index_name)


def embed_texts(client, texts):
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in resp.data]

def embed_query(client, text):
    resp = client.embeddings.create(model=EMBED_MODEL, input=text)
    return resp.data[0].embedding

def fetch_policies_from_api():
    print("Fetching policies from WordPress API...")

    all_docs = []
    page = 1

    while True:
        url = f"{WORDPRESS_API_URL}?categories={CATEGORY_ID}&page={page}&per_page=100&orderby=title&order=asc"
        res = requests.get(url)

        if res.status_code != 200:
            print("API status:", res.status_code)
            break

        try:
            raw = res.content.decode("utf-8-sig")
            data = json.loads(raw)
        except Exception as e:
            print("JSON decode failed:", e)
            print("Response snippet:", res.content[:200])
            break

        if not data:
            break

        for item in data:
            title = item.get("title", {}).get("rendered", "")
            html_content = item.get("content", {}).get("rendered", "")
            link = item.get("link", "")
            text = clean_text(html_content)

            all_docs.append({
                "title": title,
                "url": link,
                "content": text,
                "scraped_at": datetime.now().isoformat(),
            })

        page += 1

    print(f"Total policies fetched: {len(all_docs)}")
    return all_docs

def process_web_documents(web_docs, cfg):
    if not web_docs:
        return 0

    pc = cfg["pinecone"]
    index_name = cfg["pinecone_index"]
    create_index_if_missing(pc, index_name)
    index = ensure_index(pc, index_name)
    openai_client = cfg["openai"]

    total = 0

    for doc in web_docs:
        url = doc["url"]
        full_content = doc["content"].strip().lower()
        policy_hash = hashlib.md5(full_content.encode()).hexdigest()

        try:
            existing = index.query(
                vector=[0]*PINECONE_DIMENSION,
                filter={"source": url},
                top_k=1,
                include_metadata=True
            )

            old_hash = None
            if existing.matches:
                old_hash = existing.matches[0].metadata.get("policy_hash")

        except Exception as e:
            print(f"Hash check failed for {url}: {e}")
            old_hash = None

        if old_hash == policy_hash:
            continue

        try:
            index.delete(filter={"source": url})
        except Exception as e:
            print(f"Failed to delete old chunks for {url}: {e}")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        chunks = splitter.split_text(doc["content"])
        vecs = embed_texts(openai_client, chunks)
        vectors = []
        for v, chunk in zip(vecs, chunks):

            base = (url + chunk).strip().lower()
            chunk_id = hashlib.md5(base.encode()).hexdigest()

            vectors.append({
                "id": chunk_id,
                "values": v,
                "metadata": {
                    "source": url,
                    "title": doc["title"],
                    "text": chunk,
                    "policy_hash": policy_hash,
                    "scraped_at": doc["scraped_at"]
                }
            })

        index.upsert(vectors=vectors)
        total += len(vectors)

    save_last_update_time()
    print(f"Smart update complete. Upserted {total} new vectors.")
    return total

def pinecone_retrieve(cfg, query):
    pc = cfg["pinecone"]
    index = ensure_index(pc, cfg["pinecone_index"])
    qvec = embed_query(cfg["openai"], query)
    res = index.query(vector=qvec, top_k=24, include_metadata=True)
    return res.matches if hasattr(res, "matches") else res.get("matches", [])


def rerank_with_ollama(query, matches, top_n=6):
    if not matches:
        return []

    cand = matches[:18]
    docs = [
        f"[{i}] {cand[i-1]['metadata']['text'][:600]}"
        for i in range(1, len(cand) + 1)
    ]

    system = "You are a reranker. Return ONLY indices as JSON array."
    user = f"Query:\n{query}\nPassages:\n" + "\n".join(docs)

    content = ollama.chat(
        model=RERANK_MODEL,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}]
    )["message"]["content"]

    import re
    try:
        m = re.search(r"\[(.*?)\]", content, flags=re.S)
        idxs = json.loads("[" + m.group(1) + "]")
        return [cand[i-1] for i in idxs if 1 <= i <= len(cand)]
    except:
        return cand[:top_n]


def format_context(matches):
    return "\n\n".join(
        f"[{i}] {m['metadata']['text']}" for i, m in enumerate(matches, 1)
    )


def ask(query, cfg):
    matches = pinecone_retrieve(cfg, query)
    if not matches:
        return "I don't know."

    top = rerank_with_ollama(query, matches)
    context = format_context(top)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Q: {query}\nContext:\n{context}\nAnswer in 2-3 lines."}
    ]

    return ollama.chat(model=ANSWER_MODEL, messages=messages)["message"]["content"]

async def periodic_update_task():
    global scraping_in_progress
    while True:
        await asyncio.sleep(3600)
        if should_auto_update() and not scraping_in_progress:
            scraping_in_progress = True
            docs = fetch_policies_from_api()
            process_web_documents(docs, cfg)
            scraping_in_progress = False


def auto_update_database(cfg):
    global scraping_in_progress
    if scraping_in_progress:
        return 0
    scraping_in_progress = True
    docs = fetch_policies_from_api()
    count = process_web_documents(docs, cfg)
    scraping_in_progress = False
    return count

app = FastAPI(title="SmartKnow RAG")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "last_update": get_last_update_time(),
        "scraping_in_progress": scraping_in_progress
    }

@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    ans = ask(req.query, cfg)
    return ChatResponse(query=req.query, answer=ans)

@app.post("/api/update")
def update(background_tasks: BackgroundTasks):
    if scraping_in_progress:
        return {"status": "busy"}
    background_tasks.add_task(auto_update_database, cfg)
    return {"status": "started"}

@app.on_event("startup")
async def startup_event():
    global cfg
    cfg = load_env_and_clients()
    auto_update_database(cfg)
    asyncio.create_task(periodic_update_task())

if __name__ == "__main__":
    uvicorn.run("backend:app", host="127.0.0.1", port=8000, reload=True)


