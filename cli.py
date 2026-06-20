#!/usr/bin/env python3
# cli.py — command-line version (no Streamlit needed)
# Usage: python cli.py

import os
import sys
from src.transcript import fetch_transcript
from src.vector_store import index_video, search, list_indexed_videos, get_stats, delete_video
from src.llm import summarise_chunk, summarise_video, answer_question, compare_across_videos

os.makedirs("output/summaries", exist_ok=True)


def print_header():
    print("\n" + "="*50)
    print("  YT Knowledge Base — Gemma 2:9b + ChromaDB")
    print("="*50)


def cmd_add():
    url = input("\nYouTube URL: ").strip()
    title = input("Title (press Enter to skip): ").strip() or f"Video_{url[-11:]}"

    print("\nFetching transcript...")
    try:
        transcript = fetch_transcript(url)
    except Exception as e:
        print(f"Error: {e}")
        return

    print(f"Got {transcript['chunk_count']} chunks ({transcript['total_duration']})")

    print("Embedding into ChromaDB...")
    n = index_video(
        {"video_id": transcript["video_id"], "url": url},
        transcript["chunks"],
        title=title
    )
    print(f"Indexed {n} chunks.")

    do_summary = input("\nGenerate summary? (y/n): ").lower() == "y"
    if do_summary:
        print("Summarising (map-reduce)...")
        chunk_sums = []
        for i, chunk in enumerate(transcript["chunks"]):
            print(f"  Chunk {i+1}/{len(transcript['chunks'])}...", end="\r")
            s = summarise_chunk(chunk["text"], title, chunk["start_time"])
            chunk_sums.append((chunk["start_time"], s))
        print()

        summary = summarise_video(chunk_sums, title)

        safe = "".join(c for c in title if c.isalnum() or c in " _-")[:50]
        path = f"output/summaries/{safe}.md"
        with open(path, "w") as f:
            f.write(f"# {title}\n\nURL: {url}\n\n---\n\n{summary}")
        print(f"\nSummary saved to {path}")
        print("\n" + summary[:500] + "...")


def cmd_chat():
    stats = get_stats()
    if stats["total_videos"] == 0:
        print("No videos indexed yet. Add some first.")
        return

    print(f"\nKnowledge base: {stats['total_videos']} videos, {stats['total_chunks']} chunks")
    print("Indexed videos:")
    for v in stats["videos"]:
        print(f"  - {v['title']}")

    mode = input("\nMode — (a)ll videos / (s)pecific video / (c)ompare: ").lower()
    filter_ids = None

    if mode == "s":
        vids = {str(i): v for i, v in enumerate(stats["videos"])}
        for k, v in vids.items():
            print(f"  [{k}] {v['title']}")
        idx = input("Pick number: ").strip()
        if idx in vids:
            filter_ids = [vids[idx]["video_id"]]

    print("\nAsk questions about your videos. Type 'quit' to exit.\n")
    history = []

    while True:
        q = input("You: ").strip()
        if q.lower() in ("quit", "exit", "q"):
            break
        if not q:
            continue

        results = search(q, n_results=4, video_ids=filter_ids)
        if not results:
            print("Bot: No relevant content found.\n")
            continue

        if mode == "c":
            answer = compare_across_videos(q, results)
        else:
            answer = answer_question(q, results, chat_history=history)

        print(f"\nBot: {answer}")
        print("\nSources:")
        for r in results[:2]:
            m = r["metadata"]
            print(f"  [{m.get('title','?')} | {m.get('start_time','')}] score={r['relevance_score']}")
        print()

        history.append(("user", q))
        history.append(("assistant", answer))


def cmd_list():
    stats = get_stats()
    print(f"\n{stats['total_videos']} videos, {stats['total_chunks']} total chunks\n")
    for v in stats["videos"]:
        print(f"  ID: {v['video_id']}")
        print(f"  Title: {v['title']}")
        print(f"  URL: {v['url']}\n")


def cmd_delete():
    stats = get_stats()
    for i, v in enumerate(stats["videos"]):
        print(f"  [{i}] {v['title']}")
    idx = int(input("Pick number to delete: "))
    vid = stats["videos"][idx]
    n = delete_video(vid["video_id"])
    print(f"Removed {n} chunks for '{vid['title']}'")


def main():
    print_header()
    while True:
        print("\nWhat would you like to do?")
        print("  [1] Add a video")
        print("  [2] Chat with knowledge base")
        print("  [3] List indexed videos")
        print("  [4] Delete a video")
        print("  [q] Quit")

        choice = input("\nChoice: ").strip().lower()
        if choice == "1":
            cmd_add()
        elif choice == "2":
            cmd_chat()
        elif choice == "3":
            cmd_list()
        elif choice == "4":
            cmd_delete()
        elif choice in ("q", "quit"):
            print("Bye!")
            sys.exit(0)
        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()
