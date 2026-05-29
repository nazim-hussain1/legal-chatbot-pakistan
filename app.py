import os
import re
import warnings
from concurrent.futures import ThreadPoolExecutor

import faiss
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from langdetect import detect
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")

# =========================
# ENV LOAD
# =========================
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found in environment variables")

# =========================
# FLASK APP
# =========================
app = Flask(__name__)
executor = ThreadPoolExecutor(max_workers=4)

# =========================
# CONFIG
# =========================
file_path = r"E:\legal_chatbot\fyp_cleaned_dataset.csv"
INDEX_FILE = "faiss_index.bin"
CHUNKS_FILE = "chunks.npy"

TOP_K = 4
MAX_CONTEXT_CHARS = 2500
MAX_TOKENS = 400

# =========================
# OPENROUTER CLIENT
# =========================
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

MODEL_NAME = "openai/gpt-oss-120b"

# =========================
# LOAD DATASET
# =========================
def read_csv_dataset(path):
    print("Loading dataset...")

    df = pd.read_csv(path).fillna("")
    combined = df.astype(str).agg(" ".join, axis=1)

    return "\n".join(combined.tolist())

dataset_text = read_csv_dataset(file_path)
print(f"Dataset length: {len(dataset_text)}")

# =========================
# CLEAN TEXT
# =========================
def preprocess_text(text):
    text = re.sub(r"\bPage \d+\b", "", text)
    text = re.sub(r"\(i+?\)", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

cleaned_text = preprocess_text(dataset_text)

# =========================
# CHUNKING
# =========================
print("Creating chunks...")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1800,
    chunk_overlap=200
)

chunks = text_splitter.split_text(cleaned_text)

print(f"Chunks created: {len(chunks)}")

# =========================
# EMBEDDING MODEL
# =========================
print("Loading embedding model...")

embedder = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# =========================
# FAISS LOAD / BUILD
# =========================
def build_faiss(chunks):
    embeddings = embedder.encode(
        chunks,
        batch_size=32,
        convert_to_numpy=True,
        show_progress_bar=True
    ).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, INDEX_FILE)
    np.save(CHUNKS_FILE, np.array(chunks, dtype=object))

    return index

if os.path.exists(INDEX_FILE) and os.path.exists(CHUNKS_FILE):
    print("Loading FAISS index...")

    index = faiss.read_index(INDEX_FILE)
    chunks = np.load(CHUNKS_FILE, allow_pickle=True).tolist()

else:
    print("Building FAISS index...")
    index = build_faiss(chunks)

print("FAISS ready.")

# =========================
# UTILITIES
# =========================
def detect_lang_safe(text):
    try:
        return detect(text)
    except:
        return "en"

def retrieve_context(query):
    query_vec = embedder.encode([query], convert_to_numpy=True)[0].astype("float32")

    _, indices = index.search(np.array([query_vec]), k=TOP_K)

    results = [chunks[i] for i in indices[0] if i < len(chunks)]

    context = "\n\n".join(results)

    return context[:MAX_CONTEXT_CHARS]   # HARD LIMIT (important)

# =========================
# RAG CORE
# =========================
def rag_query(query):
    try:
        context = retrieve_context(query)

        prompt = f"""
You are a Pakistani legal assistant.

Answer strictly using context.

QUESTION:
{query}

CONTEXT:
{context}
"""

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=MAX_TOKENS
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print("RAG ERROR:", str(e))
        return "System error occurred while processing request."

# =========================
# ROUTES
# =========================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json(silent=True) or {}
        message = data.get("message", "").strip()

        if not message:
            return jsonify({"error": "Message is required"}), 400

        # run sync safely (no fake threading overhead)
        answer = rag_query(message)

        return jsonify({
            "reply": answer,
            "language": detect_lang_safe(message)
        })

    except Exception as e:
        print("CHAT ERROR:", str(e))
        return jsonify({"error": "Server error"}), 500

# =========================
# HEALTH
# =========================
@app.route("/health")
def health():
    return jsonify({"status": "ok"})

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    print("Starting Flask server...")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        use_reloader=False,   # 🔥 FIX DOUBLE LOAD
        threaded=True
    )