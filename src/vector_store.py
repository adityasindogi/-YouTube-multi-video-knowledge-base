# vector_store.py — embeds transcript chunks and stores in ChromaDB
import chromadb
import os
from fastembed import TextEmbedding


EMBED_MODEL = "BAAI/bge-small-en-v1.5"  # ~60MB ONNX model, runs in-process, no HTTP
DB_PATH = "./output/chroma_db"

_client: chromadb.PersistentClient = None
_collection: chromadb.Collection = None
_embedder: TextEmbedding = None


def get_embedder() -> TextEmbedding:
    global _embedder
    if _embedder is None:
        _embedder = TextEmbedding(model_name=EMBED_MODEL)
    return _embedder


def get_collection(collection_name: str = "yt_knowledge_base"):
    """Get or create a persistent ChromaDB collection (singleton)."""
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=DB_PATH)
        _collection = _client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    return _collection


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Batch embed a list of texts. Much faster than one-at-a-time."""
    embedder = get_embedder()
    return [vec.tolist() for vec in embedder.embed(texts)]


def index_video(video_metadata: dict, chunks: list, title: str = "") -> int:
    """
    Embed all chunks of a video and store in ChromaDB.
    Returns number of chunks indexed.
    """
    collection = get_collection()
    video_id = video_metadata["video_id"]

    existing = collection.get(where={"video_id": video_id})
    if existing["ids"]:
        print(f"  Video {video_id} already indexed ({len(existing['ids'])} chunks). Skipping.")
        return len(existing["ids"])

    texts = [
        f"Video: {title}\nTimestamp: {chunk['start_time']}\n\n{chunk['text']}"
        for chunk in chunks
    ]
    embeddings = embed_texts(texts)

    ids = [f"{video_id}_chunk_{i}" for i in range(len(chunks))]
    documents = [chunk["text"] for chunk in chunks]
    metadatas = [
        {
            "video_id": video_id,
            "url": video_metadata["url"],
            "title": title,
            "start_time": chunk["start_time"],
            "end_time": chunk["end_time"],
            "chunk_index": i,
        }
        for i, chunk in enumerate(chunks)
    ]

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

    if collection.count() == 0:
        return []

    query_embedding = embed_texts([query])[0]

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
