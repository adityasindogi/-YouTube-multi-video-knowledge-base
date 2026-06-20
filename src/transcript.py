# transcript.py — fetches and chunks YouTube transcripts
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from datetime import timedelta
import re


def get_video_id(url: str) -> str:
    """Extract video ID from any YouTube URL format."""
    patterns = [
        r'youtube\.com/watch\?v=([^&]+)',
        r'youtu\.be/([^?]+)',
        r'youtube\.com/shorts/([^?]+)',
        r'youtube\.com/embed/([^?]+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError(f"Could not extract video ID from URL: {url}")


def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS string."""
    return str(timedelta(seconds=int(seconds)))


def fetch_transcript(url: str, languages: list = None) -> dict:
    """
    Fetch transcript for a YouTube video.
    Returns dict with video_id, full_text, and chunks.
    Tries English first, then falls back to available languages.
    """
    if languages is None:
        languages = ['en', 'en-US', 'en-GB', 'hi']  # Hindi fallback for Indian content

    video_id = get_video_id(url)

    try:
        entries = YouTubeTranscriptApi.get_transcript(video_id, languages=languages)
    except NoTranscriptFound:
        # Try auto-generated captions in any language
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        transcript = transcript_list.find_generated_transcript(['en', 'hi'])
        entries = transcript.fetch()
    except TranscriptsDisabled:
        raise RuntimeError(f"Transcripts are disabled for video: {video_id}")

    # Build full text with timestamps
    lines = []
    for entry in entries:
        ts = format_timestamp(entry['start'])
        lines.append(f"[{ts}] {entry['text']}")
    full_text = "\n".join(lines)

    # Chunk into 5-minute segments
    chunks = chunk_transcript(entries)

    return {
        "video_id": video_id,
        "url": url,
        "full_text": full_text,
        "chunks": chunks,
        "total_duration": format_timestamp(entries[-1]['start']) if entries else "0:00:00",
        "chunk_count": len(chunks),
    }


def chunk_transcript(entries: list, chunk_seconds: int = 300) -> list:
    """
    Split transcript entries into time-based chunks.
    Default: 5 minutes (300 seconds) per chunk.
    Returns list of dicts with start_time, end_time, text.
    """
    if not entries:
        return []

    chunks = []
    current_lines = []
    chunk_start = entries[0]['start']

    for entry in entries:
        ts = format_timestamp(entry['start'])
        current_lines.append(f"[{ts}] {entry['text']}")

        # New chunk every chunk_seconds
        if entry['start'] - chunk_start >= chunk_seconds:
            chunks.append({
                "start_time": format_timestamp(chunk_start),
                "end_time": format_timestamp(entry['start']),
                "start_seconds": chunk_start,
                "text": "\n".join(current_lines),
            })
            current_lines = []
            chunk_start = entry['start']

    # Don't forget the last chunk
    if current_lines:
        chunks.append({
            "start_time": format_timestamp(chunk_start),
            "end_time": format_timestamp(entries[-1]['start']),
            "start_seconds": chunk_start,
            "text": "\n".join(current_lines),
        })

    return chunks
