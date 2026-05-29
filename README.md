# Pakistani Legal Chatbot

An AI-powered RAG-based legal assistant for Pakistani law.
Built with FAISS, Sentence Transformers, and OpenRouter API.

## Features
- Answers legal questions based on Pakistani law
- Multilingual support
- Fast semantic search using FAISS

## How to Run
1. Install requirements:
   pip install -r requirements.txt
2. Create a .env file and add:
   OPENROUTER_API_KEY=your_key_here
3. Run the app:
   python app.py
4. Open browser at:
   http://127.0.0.1:5000

## Note
faiss_index.bin and chunks.npy are auto-generated
when you run the app for the first time.
