import os
import re
import warnings

warnings.filterwarnings("ignore")

# ── Safe imports with clear error messages ──────────────
try:
    from rank_bm25 import BM25Okapi
    print("[OK] rank_bm25 loaded")
except ImportError:
    raise ImportError("Run: pip install rank-bm25")

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    print("[OK] sentence_transformers loaded")
except ImportError:
    raise ImportError("Run: pip install sentence-transformers")

try:
    import faiss
    print("[OK] faiss loaded")
except ImportError:
    raise ImportError("Run: pip install faiss-cpu")

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from langdetect import detect
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found in .env file")

app = Flask(__name__)

# ── Config ──────────────────────────────────────────────
file_path   = r"D:\RAG-FYP\fyp_cleaned_dataset.csv"
INDEX_FILE  = "faiss_index.bin"
CHUNKS_FILE = "chunks.npy"
TOP_K       = 20
RERANK_TOP  = 5
MAX_TOKENS  = 800
MIN_SCORE   = 0.20

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)
MODEL_NAME = "gpt-oss-120b"

# ── Load dataset ─────────────────────────────────────────
def read_csv_dataset(path):
    print(f"Loading dataset from: {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")
    df = pd.read_csv(path).fillna("")
    combined = df.astype(str).agg(" ".join, axis=1)
    result = "\n".join(combined.tolist())
    print(f"[OK] Dataset loaded: {len(result):,} characters")
    return result

# ── Preprocessing ────────────────────────────────────────
def preprocess_text(text):
    text = re.sub(r"\bPage \d+\b", "", text)
    text = re.sub(r"_{3,}", " ", text)
    text = re.sub(r"-{3,}", " ", text)
    text = re.sub(r"\s{3,}", "  ", text)
    return text.strip()

# ── Chunking ─────────────────────────────────────────────
def create_legal_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1400,
        chunk_overlap=300,
        separators=[
            "\nArticle ",
            "\nPart ",
            "\nChapter ",
            "\n\n",
            "\n   (",
            "\n  (",
            "\n (",
            "\n",
            " ",
            ""
        ]
    )
    chunks = splitter.split_text(text)
    print(f"[OK] Created {len(chunks)} chunks")
    return chunks

# ── Initialize dataset and chunks ────────────────────────
try:
    dataset_text = read_csv_dataset(file_path)
    cleaned_text = preprocess_text(dataset_text)
    chunks_list  = create_legal_chunks(cleaned_text)
except Exception as e:
    print(f"[FATAL] Dataset loading failed: {e}")
    raise

# ── Embedding model ──────────────────────────────────────
print("Loading embedding model (may download on first run ~420MB)...")
try:
    embedder = SentenceTransformer(
        "sentence-transformers/multi-qa-mpnet-base-dot-v1"
    )
    print("[OK] Embedding model loaded")
except Exception as e:
    print(f"[WARN] Primary model failed: {e}")
    print("Falling back to multilingual MiniLM...")
    embedder = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    print("[OK] Fallback embedding model loaded")

# ── Reranker ─────────────────────────────────────────────
print("Loading reranker model (may download on first run ~85MB)...")
try:
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    USE_RERANKER = True
    print("[OK] Reranker loaded")
except Exception as e:
    print(f"[WARN] Reranker failed to load: {e}")
    print("Continuing without reranker...")
    USE_RERANKER = False

# ── FAISS index ──────────────────────────────────────────
def build_faiss(chunks):
    print(f"Encoding {len(chunks)} chunks (this may take several minutes)...")
    embeddings = embedder.encode(
        chunks,
        batch_size=16,
        convert_to_numpy=True,
        show_progress_bar=True
    ).astype("float32")

    print("Normalizing vectors for cosine similarity...")
    faiss.normalize_L2(embeddings)

    idx = faiss.IndexFlatIP(embeddings.shape[1])
    idx.add(embeddings)

    faiss.write_index(idx, INDEX_FILE)
    np.save(CHUNKS_FILE, np.array(chunks, dtype=object))
    print(f"[OK] FAISS index built and saved: {len(chunks)} vectors")
    return idx

# ── Delete old index if embedding model changed ──────────
# If you changed the embedding model, delete faiss_index.bin and chunks.npy
# before restarting so they are rebuilt with the new model dimensions.

try:
    if os.path.exists(INDEX_FILE) and os.path.exists(CHUNKS_FILE):
        print("Loading existing FAISS index...")
        index  = faiss.read_index(INDEX_FILE)
        chunks = np.load(CHUNKS_FILE, allow_pickle=True).tolist()
        print(f"[OK] FAISS index loaded: {len(chunks)} chunks")
    else:
        print("No existing index found. Building new index...")
        index  = build_faiss(chunks_list)
        chunks = chunks_list
except Exception as e:
    print(f"[WARN] Index load failed ({e}). Rebuilding...")
    if os.path.exists(INDEX_FILE):
        os.remove(INDEX_FILE)
    if os.path.exists(CHUNKS_FILE):
        os.remove(CHUNKS_FILE)
    index  = build_faiss(chunks_list)
    chunks = chunks_list

# ── BM25 index ───────────────────────────────────────────
print("Building BM25 index...")
try:
    tokenized_chunks = [c.lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    USE_BM25 = True
    print("[OK] BM25 index ready")
except Exception as e:
    print(f"[WARN] BM25 failed: {e}. Using vector-only retrieval.")
    USE_BM25 = False

# ── Legal query expansion ────────────────────────────────
LEGAL_SYNONYMS = {
    "land":          ["property", "immovable property", "acquisition"],
    "compensation":  ["payment", "compulsory acquisition", "indemnity"],
    "arrest":        ["detention", "custody", "safeguards"],
    "freedom":       ["fundamental rights", "liberty"],
    "acquire":       ["compulsory acquisition", "take possession"],
    "parliament":    ["majlis-e-shoora", "national assembly", "senate"],
    "court":         ["judicature", "high court", "supreme court"],
    "equality":      ["equal protection", "non-discrimination", "article 25"],
    "education":     ["right to education", "article 25a", "compulsory education"],
    "religion":      ["freedom of religion", "article 20", "religious freedom"],
    "speech":        ["freedom of speech", "article 19", "expression"],
}

def expand_query(query: str) -> str:
    expanded = query
    q_lower  = query.lower()
    for term, synonyms in LEGAL_SYNONYMS.items():
        if term in q_lower:
            expanded += " " + " ".join(synonyms[:2])
    art_match = re.search(r'article\s+(\d+[A-Za-z]*)', query, re.IGNORECASE)
    if art_match:
        expanded += f" Article {art_match.group(1)} constitution Pakistan law"
    return expanded

# ── Hybrid retrieval ─────────────────────────────────────
def hybrid_retrieve(query: str, k: int = TOP_K) -> list:
    expanded = expand_query(query)

    # Vector retrieval
    q_vec = embedder.encode([expanded], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_vec)
    vec_scores, vec_indices = index.search(q_vec, min(k * 2, len(chunks)))

    rrf_scores = {}

    # Add vector scores to RRF
    valid_vec_indices = []
    for rank, (score, idx) in enumerate(zip(vec_scores[0], vec_indices[0])):
        if idx < len(chunks) and score >= MIN_SCORE:
            valid_vec_indices.append(int(idx))
            rrf_scores[int(idx)] = rrf_scores.get(int(idx), 0) + 1 / (60 + rank + 1)

    # Add BM25 scores to RRF
    if USE_BM25:
        try:
            bm25_scores  = bm25.get_scores(expanded.lower().split())
            bm25_top_k   = np.argsort(bm25_scores)[::-1][:k * 2]
            for rank, idx in enumerate(bm25_top_k):
                rrf_scores[int(idx)] = rrf_scores.get(int(idx), 0) + 1 / (60 + rank + 1)
        except Exception as e:
            print(f"[WARN] BM25 retrieval error: {e}")

    if not rrf_scores:
        # Fallback: return whatever vector search found
        return [chunks[i] for i in valid_vec_indices[:k] if i < len(chunks)]

    top_indices = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:k]
    return [chunks[i] for i in top_indices if i < len(chunks)]

# ── Reranking ────────────────────────────────────────────
def rerank(query: str, candidates: list, top_n: int = RERANK_TOP) -> list:
    if not candidates:
        return []
    if not USE_RERANKER or len(candidates) <= top_n:
        return candidates[:top_n]
    try:
        pairs  = [(query, c) for c in candidates]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        return [text for _, text in ranked[:top_n]]
    except Exception as e:
        print(f"[WARN] Reranker error: {e}. Using top-{top_n} without reranking.")
        return candidates[:top_n]

# ── Context assembly ─────────────────────────────────────
def assemble_context(top_chunks: list) -> str:
    return "\n\n---\n\n".join(
        f"[Provision {i+1}]\n{chunk}"
        for i, chunk in enumerate(top_chunks)
    )

# ── RAG core ─────────────────────────────────────────────
def rag_query(query: str) -> str:
    try:
        print(f"\n[QUERY] {query}")

        candidates = hybrid_retrieve(query, k=TOP_K)
        print(f"[RETRIEVE] {len(candidates)} candidates found")

        if not candidates:
            return "No relevant legal provisions were found in the dataset for this query."

        top_chunks = rerank(query, candidates, top_n=RERANK_TOP)
        print(f"[RERANK] {len(top_chunks)} chunks selected for context")

        context = assemble_context(top_chunks)
        print(f"[CONTEXT] {len(context):,} characters sent to LLM")

        prompt = f"""You are a strict Pakistani legal assistant. Answer ONLY using the context below.

RULES:
1. Only use information explicitly present in the CONTEXT.
2. Quote the relevant Article or Section number if visible in the context.
3. If the context does not contain sufficient information, state:
   "The provided legal provisions do not directly address this query."
4. Do not add general legal knowledge not present in the context.
5. Structure your answer: Legal Basis → Applicable Provision → Conclusion.

CONTEXT:
{context}

QUERY:
{query}

ANSWER:"""

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You are a strict Pakistani legal retrieval assistant. Never fabricate legal information."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.0,
            max_tokens=MAX_TOKENS
        )

        answer = response.choices[0].message.content.strip()
        print(f"[RESPONSE] {len(answer)} characters returned")
        return answer

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"[ERROR] RAG query failed:\n{error_details}")
        return f"System error: {str(e) or 'Unknown error — check terminal for full traceback'}"

# ── Routes ───────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data    = request.get_json(silent=True) or {}
        message = data.get("message", "").strip()
        if not message:
            return jsonify({"error": "Message is required"}), 400
        answer = rag_query(message)
        try:
            lang = detect(message)
        except Exception:
            lang = "en"
        return jsonify({"reply": answer, "language": lang})
    except Exception as e:
        import traceback
        print(f"[ERROR] Chat route:\n{traceback.format_exc()}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route("/health")
def health():
    return jsonify({
        "status":     "ok",
        "chunks":     len(chunks),
        "bm25":       USE_BM25,
        "reranker":   USE_RERANKER,
        "index_type": type(index).__name__
    })

# ── Main ─────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*50)
    print("Pakistan Legal RAG System Starting...")
    print("="*50)
    print(f"Chunks loaded : {len(chunks)}")
    print(f"BM25 active   : {USE_BM25}")
    print(f"Reranker      : {USE_RERANKER}")
    print("="*50 + "\n")
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        use_reloader=False,
        threaded=True
    )
