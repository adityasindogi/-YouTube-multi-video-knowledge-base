# tests/test_transcript.py
# Run with: python -m pytest tests/ -v
# Or simply: python tests/test_transcript.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.transcript import get_video_id, chunk_transcript


def test_video_id_standard_url():
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    assert get_video_id(url) == "dQw4w9WgXcQ"


def test_video_id_short_url():
    url = "https://youtu.be/dQw4w9WgXcQ"
    assert get_video_id(url) == "dQw4w9WgXcQ"


def test_video_id_shorts_url():
    url = "https://www.youtube.com/shorts/dQw4w9WgXcQ"
    assert get_video_id(url) == "dQw4w9WgXcQ"


def test_video_id_embed_url():
    url = "https://www.youtube.com/embed/dQw4w9WgXcQ"
    assert get_video_id(url) == "dQw4w9WgXcQ"


def test_video_id_with_params():
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=120&list=PL123"
    assert get_video_id(url) == "dQw4w9WgXcQ"


def test_video_id_invalid():
    try:
        get_video_id("https://www.google.com")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_chunk_transcript_empty():
    result = chunk_transcript([])
    assert result == []


def test_chunk_transcript_short_video():
    """A 2-minute video should produce 1 chunk."""
    entries = [
        {"start": i * 10, "text": f"Line at {i*10}s"}
        for i in range(12)  # 120 seconds
    ]
    chunks = chunk_transcript(entries, chunk_seconds=300)
    assert len(chunks) == 1
    assert chunks[0]["start_time"] == "0:00:00"


def test_chunk_transcript_long_video():
    """A 12-minute video should produce 2-3 chunks at 5-min intervals."""
    entries = [
        {"start": i * 10, "text": f"Line at {i*10}s"}
        for i in range(72)  # 720 seconds = 12 minutes
    ]
    chunks = chunk_transcript(entries, chunk_seconds=300)
    assert len(chunks) >= 2
    assert chunks[0]["start_time"] == "0:00:00"


def test_chunk_has_required_keys():
    entries = [
        {"start": 0, "text": "Hello"},
        {"start": 10, "text": "World"},
    ]
    chunks = chunk_transcript(entries, chunk_seconds=300)
    assert len(chunks) == 1
    chunk = chunks[0]
    assert "start_time" in chunk
    assert "end_time" in chunk
    assert "text" in chunk
    assert "start_seconds" in chunk


if __name__ == "__main__":
    tests = [
        test_video_id_standard_url,
        test_video_id_short_url,
        test_video_id_shorts_url,
        test_video_id_embed_url,
        test_video_id_with_params,
        test_video_id_invalid,
        test_chunk_transcript_empty,
        test_chunk_transcript_short_video,
        test_chunk_transcript_long_video,
        test_chunk_has_required_keys,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  PASS  {test.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {test.__name__}: {e}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed out of {len(tests)} tests")
