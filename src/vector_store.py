# vector_store.py — embeds transcript chunks and stores in ChromaDB
import chromadb
import ollama
import json
import os


EMBED_MODEL = "nomic-embed-text"   # pull with: ollama pull nomic-embed-text
DB_PATH = "./output/chroma_db"     # persists to disk between runs


def get_collection(collection_name: str = "yt_knowledge_base"):
    """Get or create a persistent ChromaDB collection."""
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}   # cosine similarity for text
    )
    return collection


def embed_text(text: str) -> list:
    """Get embedding vector for a piece of text using Ollama."""
    response = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    return response["embedding"]


def index_video(video_metadata: dict, chunks: list, title: str = "") -> int:
    """
    Embed all chunks of a video and store in ChromaDB.
    Returns number of chunks indexed.

    video_metadata: dict with video_id, url, etc.
    chunks: list of chunk dicts from transcript.py
    title: human-readable title for the video
    """
    collection = get_collection()
    video_id = video_metadata["video_id"]

    # Check if already indexed (avoid re-embedding)
    existing = collection.get(where={"video_id": video_id})
    if existing["ids"]:
        print(f"  Video {video_id} already indexed ({len(existing['ids'])} chunks). Skipping.")
        return len(existing["ids"])

    ids = []
    embeddings = []
    documents = []
    metadatas = []

    print(f"  Embedding {len(chunks)} chunks for '{title or video_id}'...")

    for i, chunk in enumerate(chunks):
        chunk_id = f"{video_id}_chunk_{i}"

        # Create a rich text for embedding (includes context)
        embed_text_content = f"Video: {title}\nTimestamp: {chunk['start_time']}\n\n{chunk['text']}"

        try:
            embedding = embed_text(embed_text_content)
        except Exception as e:
            print(f"  Warning: embedding failed for chunk {i}: {e}")
            continue

        ids.append(chunk_id)
        embeddings.append(embedding)
        documents.append(chunk["text"])
        metadatas.append({
            "video_id": video_id,
            "url": video_metadata["url"],
            "title": title,
            "start_time": chunk["start_time"],
            "end_time": chunk["end_time"],
            "chunk_index": i,
        })

        if (i + 1) % 5 == 0:
            print(f"    {i + 1}/{len(chunks)} chunks embedded")

    if ids:
        collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
        print(f"  Indexed {len(ids)} chunks successfully.")

    return len(ids)


def search(query: str, n_results: int = 5, video_ids: list = None) -> list:
    """
    Search the knowledge base for chunks relevant to a query.
    Optionally filter by specific video_ids.
    Returns list of result dicts with text, metadata, distance.
    """
    collection = get_collection()

    # Check if collection has anything
    if collection.count() == 0:
        return []

    query_embedding = embed_text(query)

    # Build optional filter
    where_filter = None
    if video_ids and len(video_ids) == 1:
        where_filter = {"video_id": video_ids[0]}
    elif video_ids and len(video_ids) > 1:
        where_filter = {"video_id": {"$in": video_ids}}

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(n_results, collection.count()),
        where=where_filter,
        include=["documents", "metadatas", "distances"]
    )

    # Format results cleanly
    formatted = []
    for i in range(len(results["ids"][0])):
        formatted.append({
            "chunk_text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "relevance_score": round(1 - results["distances"][0][i], 3),  # cosine → similarity
        })

    return formatted


def list_indexed_videos() -> list:
    """Return list of all videos currently in the knowledge base."""
    collection = get_collection()
    if collection.count() == 0:
        return []

    # Get all unique video metadata
    all_items = collection.get(include=["metadatas"])
    seen = {}
    for meta in all_items["metadatas"]:
        vid_id = meta["video_id"]
        if vid_id not in seen:
            seen[vid_id] = {
                "video_id": vid_id,
                "title": meta.get("title", "Untitled"),
                "url": meta.get("url", ""),
            }

    return list(seen.values())


def delete_video(video_id: str) -> int:
    """Remove all chunks for a video from the knowledge base."""
    collection = get_collection()
    existing = collection.get(where={"video_id": video_id})
    if existing["ids"]:
        collection.delete(ids=existing["ids"])
        return len(existing["ids"])
    return 0


def get_stats() -> dict:
    """Return stats about the knowledge base."""
    collection = get_collection()
    videos = list_indexed_videos()
    return {
        "total_chunks": collection.count(),
        "total_videos": len(videos),
        "videos": videos,
    }
