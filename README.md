# YouTube Multi-Video Knowledge Base
### Powered by Gemma 2:9b · ChromaDB · Ollama

Ask questions across multiple YouTube videos. All processing is **100% local** — no OpenAI, no cloud, no cost.

---

## What it does

1. **Fetches transcripts** from any YouTube URL (no API key needed)
2. **Chunks** each video into 5-minute segments
3. **Embeds** each chunk into a local ChromaDB vector database
4. **Answers questions** by retrieving the most relevant chunks and passing them to Gemma
5. **Compares across videos** — ask "what do all these videos say about attention mechanisms?"
6. **Generates summaries** using map-reduce (handles videos of any length)
7. **Writes Googly AI episode scripts** from any indexed video

---

## Setup

### 1. Install Ollama
```
https://ollama.ai — download and install
```

### 2. Pull the models
```bash
ollama pull gemma2:9b          # main reasoning model
ollama pull nomic-embed-text   # embedding model (small, fast)
```

### 3. Clone and set up Python environment
```bash
git clone https://github.com/YOUR_USERNAME/AgenticAI.git
cd AgenticAI
python -m venv venv

# Activate virtual environment
# Windows: .\venv\Scripts\activate
# Mac/Linux: source venv/bin/activate

pip install -r requirements.txt
```

### 4. Copy environment variables
```bash
cp .env.example .env
```

---

## Run

### Web UI (recommended)
```bash
streamlit run app.py
```
Opens at `http://localhost:8501`

### Command line
```bash
python cli.py
```

---

## Project structure

```
AgenticAI/
├── app.py                 # Streamlit web UI (4 modes)
├── cli.py                 # Command-line interface
├── src/
│   ├── __init__.py
│   ├── transcript.py      # YouTube transcript fetching + chunking
│   ├── vector_store.py    # ChromaDB embedding + search
│   └── llm.py             # All Gemma calls (summarise, Q&A, episode gen)
├── tests/
│   └── test_transcript.py # Unit tests for transcript module
├── output/                # (gitignored) generated data
│   ├── chroma_db/         # Persisted vector database
│   └── summaries/         # Saved markdown summaries + episode drafts
├── .env.example           # Environment variable template
├── .gitignore
├── requirements.txt
└── README.md
```

---

## How it works (the architecture)

```
YouTube URL
    ↓
fetch_transcript()        — youtube-transcript-api, no key needed
    ↓
chunk_transcript()        — splits into 5-min segments with timestamps
    ↓
embed_text() × N chunks   — nomic-embed-text via Ollama (local embeddings)
    ↓
ChromaDB.add()            — stored on disk in output/chroma_db/

At query time:
question → embed_text() → ChromaDB.query() → top-K chunks
                                                    ↓
                                            Gemma 2:9b (local)
                                                    ↓
                                              Answer with sources
```

---

## Tips

- Videos need captions (auto-generated is fine — most YouTube videos have them)
- Hindi videos work too — add `'hi'` to language list in `transcript.py`
- First-time embedding is slow (30–60 sec per video). Subsequent runs reuse the DB.
- Gemma 2:9b needs ~8GB RAM. Use `gemma2:2b` if on limited hardware.
- The DB persists between runs — add videos once, chat forever.
