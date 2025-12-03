# 🎓 SmartKnow – Student Policy Chatbot

## 📌 Description

**SmartKnow** is an AI-powered chatbot that helps students quickly find answers about university policies.  
Ask questions in plain English, and SmartKnow retrieves accurate information from official policy pages and generates a clear, short answer.

SmartKnow uses a complete **RAG (Retrieval-Augmented Generation)** pipeline, including:

- 🕷 Automatic WordPress API scraping  
- ✂️ Text chunking  
- 🔢 OpenAI vector embeddings  
- 🗂 Pinecone vector search  
- 🧠 LLaMA-based answer generation  

---

## ✨ Features

✔ **FastAPI backend** — scraping, embedding, Pinecone search, LLM answers  
✔ **Beautiful chat-style frontend** — fully local, modern UI  
✔ **Automatic weekly updates** — always current policies  
✔ **End-to-end RAG pipeline**

### 🔄 RAG Workflow:

1. Scrape WordPress API  
2. Clean and split text  
3. Generate embeddings  
4. Store vectors in Pinecone  
5. Retrieve + rerank using LLaMA  
6. Generate short, accurate answers  

---
I completed the entire development of this project in Visual Studio Code.

## 📁 Project Structure

```
smartknow/
│── backend.py           # FastAPI backend (RAG pipeline)
│── frontend.html        # Chat UI
│── .env                 # API keys (not included in GitHub)
│── requirements.txt     # Dependencies
```

## Installation 

#### 1️⃣  Clone the Project : 

```

git clone https://github.com/RAJASREE-GORREPATI/smartknow.git
cd smartknow

```

#### 2️⃣ Install Required Python Packages: 

```

pip install -r requirements.txt

```

If you don’t have a requirements file, install manually:

```

pip install fastapi uvicorn openai pinecone-client langchain python-dotenv beautifulsoup4 requests ollama

```

#### 3️⃣ Create a ```.env ```File:

Create a file named ```.env``` in the project folder and add:

```

OPENAI_API_KEY=your_key_here
PINECONE_API_KEY=your_key_here
PINECONE_INDEX=smartknow-index

```

This file is private so I didn't added in github.


## 🚀 Run the Backend : 

Start the FastAPI server:

```

uvicorn backend:app --reload --port 8000

```

Check if it works by opening this link in your browser:

```

http://localhost:8000/api/health

```

If you see 
```
{status": "ok" } 
```
Backend is running successfully.

## 💬 Open the Frontend : 

Just double-click:
```
frontend.html
```
The chatbot interface will open in your browser.


## 🧪 Questions you can try : 

1. Explain the attendance policy in 2–3 lines.
2. Describe the student code of conduct briefly.
3. What steps are required to file an academic appeal?
4. What is the policy on plagiarism?

## 🤝 Contributing : 

If you want to help improve the project:

1. Fork the repo

2. Create a branch

3. Add changes

4. Open a pull request


## 📬 Contact : 

👤 Raja Sree Gorrepati

📧 Email: rajasreegorrepati@gmail.com

