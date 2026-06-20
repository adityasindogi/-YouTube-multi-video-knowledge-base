# llm.py — all LLM calls go through here (Gemma 2:9b via Ollama)
import ollama

MODEL = "gemma2:9b"


def _chat(messages: list, system: str = None) -> str:
    """Base chat call. Returns response text."""
    if system:
        messages = [{"role": "system", "content": system}] + messages
    response = ollama.chat(model=MODEL, messages=messages)
    return response["message"]["content"]


# ── Summarisation ─────────────────────────────────────────────────────────────

def summarise_chunk(chunk_text: str, video_title: str, timestamp: str) -> str:
    """Summarise a single 5-minute chunk."""
    return _chat([{
        "role": "user",
        "content": f"Summarise this video segment in 3 concise bullet points.\nVideo: {video_title}\nTimestamp: {timestamp}\n\n{chunk_text}"
    }])


def summarise_video(chunk_summaries: list, video_title: str) -> str:
    """
    Map-reduce final summary.
    chunk_summaries: list of (timestamp, summary_text) tuples
    """
    combined = "\n\n".join([f"[{ts}]\n{text}" for ts, text in chunk_summaries])
    return _chat([{
        "role": "user",
        "content": f"""You have chunk-by-chunk summaries of a YouTube video titled "{video_title}".

Produce a structured summary with these exact sections:

## Overview
2-3 sentence overview of what the video covers.

## Key points
5 bullet points of the most important ideas.

## Notable quotes / moments
Top 3 moments worth re-watching, with timestamps.

## Action items / takeaways
What should someone DO after watching this?

## Googly AI angle
If this video covers an AI/tech concept, how would you explain the core idea using a cricket analogy for an Indian audience? (If not applicable, write "N/A")

---
CHUNK SUMMARIES:
{combined}"""
    }])


# ── Q&A ───────────────────────────────────────────────────────────────────────

def answer_question(question: str, context_chunks: list, chat_history: list = None) -> str:
    """
    Answer a question using retrieved transcript chunks as context.
    context_chunks: list of result dicts from vector_store.search()
    chat_history: list of previous (role, content) turns for conversational memory
    """
    # Build context string with source info
    context_parts = []
    for chunk in context_chunks:
        meta = chunk["metadata"]
        context_parts.append(
            f"[From: {meta.get('title', 'Unknown')} | {meta.get('start_time', '')}]\n{chunk['chunk_text']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    system = """You are a helpful assistant answering questions about YouTube videos.
You have been given relevant transcript excerpts to answer the question.
Always mention which video and timestamp your answer comes from.
If the answer isn't in the provided context, say so clearly."""

    # Build messages with history
    messages = []
    if chat_history:
        for role, content in chat_history[-6:]:  # last 3 turns = 6 messages
            messages.append({"role": role, "content": content})

    messages.append({
        "role": "user",
        "content": f"""CONTEXT FROM TRANSCRIPTS:
{context}

QUESTION: {question}"""
    })

    return _chat(messages, system=system)


def compare_across_videos(question: str, context_chunks: list) -> str:
    """
    Special mode: compare/contrast answers across multiple videos.
    Groups chunks by video and synthesises a cross-video answer.
    """
    # Group by video
    by_video = {}
    for chunk in context_chunks:
        title = chunk["metadata"].get("title", chunk["metadata"]["video_id"])
        if title not in by_video:
            by_video[title] = []
        by_video[title].append(chunk)

    context_parts = []
    for title, chunks in by_video.items():
        text = "\n".join([c["chunk_text"] for c in chunks[:2]])  # top 2 chunks per video
        context_parts.append(f"=== {title} ===\n{text}")

    context = "\n\n".join(context_parts)

    return _chat([{
        "role": "user",
        "content": f"""You have transcript excerpts from multiple YouTube videos. 
Compare and contrast how each video addresses the following question.
Structure your answer by video, then give a synthesis at the end.

QUESTION: {question}

TRANSCRIPTS:
{context}"""
    }])


def generate_episode_draft(video_summary: str, video_title: str) -> str:
    """Generate a Googly AI episode draft from a video summary."""
    return _chat([{
        "role": "system",
        "content": "You are the Googly AI content writer. You create engaging educational content that explains AI/tech concepts through cricket analogies for Indian audiences. Your style is fun, clear, and uses cricket naturally — not forced."
    }, {
        "role": "user",
        "content": f"""Based on this video summary, write a complete Googly AI episode script.

VIDEO: {video_title}

SUMMARY:
{video_summary}

---
Write the episode with these sections:

## Hook (30 seconds)
Open with a cricket scenario that mirrors the AI concept. Make it vivid.

## The concept
Explain the AI/tech concept using the cricket analogy. Keep it simple — target someone who knows cricket but is new to AI.

## Deep dive
3 key ideas from the video, each explained with a cricket parallel.

## The stumper
One surprising or counterintuitive insight from the video.

## Reel script (60 seconds)
A punchy short-form script for a YouTube Reel or Instagram video.

## Tags & description
5 YouTube tags + a 100-word video description."""
    }])
