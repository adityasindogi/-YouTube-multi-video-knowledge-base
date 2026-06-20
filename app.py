# app.py — Multi-Video YouTube Knowledge Base
# Run with: streamlit run app.py

import streamlit as st
import os
import json
from src.transcript import fetch_transcript, get_video_id
from src.vector_store import index_video, search, list_indexed_videos, delete_video, get_stats
from src.llm import summarise_chunk, summarise_video, answer_question, compare_across_videos, generate_episode_draft

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="YT Knowledge Base",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

os.makedirs("output", exist_ok=True)
os.makedirs("output/summaries", exist_ok=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("YT Knowledge Base")
    st.caption("Powered by Gemma 2:9b + ChromaDB")

    stats = get_stats()
    col1, col2 = st.columns(2)
    col1.metric("Videos", stats["total_videos"])
    col2.metric("Chunks", stats["total_chunks"])

    st.divider()

    mode = st.radio(
        "Mode",
        ["Add videos", "Chat with knowledge base", "Summaries", "Googly AI"],
        index=0
    )

    if stats["videos"]:
        st.divider()
        st.caption("Indexed videos")
        for vid in stats["videos"]:
            with st.expander(vid["title"][:40] + "..." if len(vid["title"]) > 40 else vid["title"]):
                st.caption(vid["url"])
                if st.button("Remove", key=f"del_{vid['video_id']}"):
                    n = delete_video(vid["video_id"])
                    st.success(f"Removed {n} chunks")
                    st.rerun()

# ── Main area ─────────────────────────────────────────────────────────────────

# ═══════════════════════════════════════════
# MODE 1: ADD VIDEOS
# ═══════════════════════════════════════════
if mode == "Add videos":
    st.header("Add videos to knowledge base")
    st.write("Paste one or more YouTube URLs. Each video's transcript will be fetched, chunked into 5-minute segments, and embedded into ChromaDB for semantic search.")

    urls_input = st.text_area(
        "YouTube URLs (one per line)",
        placeholder="https://www.youtube.com/watch?v=...\nhttps://youtu.be/...",
        height=120
    )

    title_input = st.text_input(
        "Video title (optional — leave blank to auto-detect from transcript)",
        placeholder="e.g. Andrew Ng — What is Machine Learning?"
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        also_summarise = st.checkbox("Also generate summary", value=True)
    with col2:
        generate_episode = st.checkbox("Generate Googly AI episode draft", value=False)

    if st.button("Add to knowledge base", type="primary"):
        urls = [u.strip() for u in urls_input.strip().split("\n") if u.strip()]
        if not urls:
            st.error("Please enter at least one URL.")
        else:
            for url in urls:
                st.divider()
                st.write(f"Processing: `{url}`")

                with st.status(f"Fetching transcript...", expanded=True) as status:
                    try:
                        transcript = fetch_transcript(url)
                        title = title_input or f"Video {transcript['video_id']}"
                        st.write(f"Got {transcript['chunk_count']} chunks ({transcript['total_duration']} total)")

                        status.update(label="Embedding chunks into ChromaDB...")
                        n_indexed = index_video(
                            {"video_id": transcript["video_id"], "url": url},
                            transcript["chunks"],
                            title=title
                        )
                        st.write(f"Indexed {n_indexed} chunks")

                        if also_summarise:
                            status.update(label="Generating summary (map-reduce)...")
                            chunk_summaries = []
                            progress = st.progress(0)
                            for i, chunk in enumerate(transcript["chunks"]):
                                chunk_sum = summarise_chunk(chunk["text"], title, chunk["start_time"])
                                chunk_summaries.append((chunk["start_time"], chunk_sum))
                                progress.progress((i + 1) / len(transcript["chunks"]))

                            full_summary = summarise_video(chunk_summaries, title)

                            # Save to file
                            safe_title = "".join(c for c in title if c.isalnum() or c in " _-")[:50]
                            summary_path = f"output/summaries/{safe_title}.md"
                            with open(summary_path, "w") as f:
                                f.write(f"# {title}\n\n")
                                f.write(f"**URL:** {url}\n\n")
                                f.write(f"**Duration:** {transcript['total_duration']}\n\n")
                                f.write("---\n\n")
                                f.write(full_summary)

                            st.session_state[f"summary_{transcript['video_id']}"] = full_summary
                            st.write("Summary saved to", summary_path)

                            if generate_episode:
                                status.update(label="Writing Googly AI episode...")
                                episode = generate_episode_draft(full_summary, title)
                                episode_path = f"output/summaries/{safe_title}_episode.md"
                                with open(episode_path, "w") as f:
                                    f.write(f"# Googly AI Episode: {title}\n\n")
                                    f.write(episode)
                                st.write("Episode draft saved to", episode_path)

                        status.update(label="Done!", state="complete")

                    except Exception as e:
                        status.update(label=f"Error: {e}", state="error")
                        st.error(str(e))

# ═══════════════════════════════════════════
# MODE 2: CHAT WITH KNOWLEDGE BASE
# ═══════════════════════════════════════════
elif mode == "Chat with knowledge base":
    st.header("Chat with your video library")

    if stats["total_videos"] == 0:
        st.info("No videos indexed yet. Go to 'Add videos' first.")
    else:
        # Filters
        col1, col2 = st.columns([2, 1])
        with col1:
            search_mode = st.radio(
                "Search scope",
                ["All videos", "Specific video", "Compare across videos"],
                horizontal=True
            )
        with col2:
            n_chunks = st.slider("Context chunks", 2, 8, 4)

        selected_video_ids = None
        if search_mode == "Specific video":
            video_options = {v["title"]: v["video_id"] for v in stats["videos"]}
            selected_title = st.selectbox("Select video", list(video_options.keys()))
            selected_video_ids = [video_options[selected_title]]

        # Chat history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Render chat history
        for role, content in st.session_state.chat_history:
            with st.chat_message(role):
                st.markdown(content)

        # Input
        if question := st.chat_input("Ask anything about your videos..."):
            st.session_state.chat_history.append(("user", question))
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Searching knowledge base..."):
                    # Retrieve relevant chunks
                    results = search(question, n_results=n_chunks, video_ids=selected_video_ids)

                    if not results:
                        answer = "No relevant content found. Try adding more videos first."
                    elif search_mode == "Compare across videos":
                        answer = compare_across_videos(question, results)
                    else:
                        answer = answer_question(
                            question, results,
                            chat_history=st.session_state.chat_history[:-1]
                        )

                st.markdown(answer)

                # Show sources
                with st.expander("Sources used"):
                    for r in results:
                        meta = r["metadata"]
                        st.caption(
                            f"**{meta.get('title', 'Unknown')}** | "
                            f"{meta.get('start_time', '')} | "
                            f"Relevance: {r['relevance_score']}"
                        )
                        st.text(r["chunk_text"][:200] + "...")

            st.session_state.chat_history.append(("assistant", answer))

        if st.button("Clear chat history"):
            st.session_state.chat_history = []
            st.rerun()

# ═══════════════════════════════════════════
# MODE 3: SUMMARIES
# ═══════════════════════════════════════════
elif mode == "Summaries":
    st.header("Video summaries")

    summary_dir = "output/summaries"
    if not os.path.exists(summary_dir) or not os.listdir(summary_dir):
        st.info("No summaries yet. Add a video with 'Also generate summary' checked.")
    else:
        files = [f for f in os.listdir(summary_dir) if f.endswith(".md") and "_episode" not in f]
        if not files:
            st.info("No summaries yet.")
        else:
            selected = st.selectbox("Select summary", files)
            with open(os.path.join(summary_dir, selected)) as f:
                content = f.read()
            st.markdown(content)
            st.download_button("Download", content, file_name=selected)

# ═══════════════════════════════════════════
# MODE 4: GOOGLY AI
# ═══════════════════════════════════════════
elif mode == "Googly AI":
    st.header("Googly AI episode generator")
    st.write("Generate a full Googly AI episode script from any indexed video.")

    if stats["total_videos"] == 0:
        st.info("No videos indexed yet. Go to 'Add videos' first.")
    else:
        video_options = {v["title"]: v["video_id"] for v in stats["videos"]}
        selected_title = st.selectbox("Select video", list(video_options.keys()))
        selected_vid_id = video_options[selected_title]

        # Check if summary exists
        safe_title = "".join(c for c in selected_title if c.isalnum() or c in " _-")[:50]
        summary_path = f"output/summaries/{safe_title}.md"

        if os.path.exists(summary_path):
            with open(summary_path) as f:
                summary_text = f.read()
        else:
            summary_text = None
            st.warning("No summary found for this video. Generate one first via 'Add videos'.")

        if summary_text and st.button("Generate Googly AI episode", type="primary"):
            with st.spinner("Writing episode..."):
                episode = generate_episode_draft(summary_text, selected_title)

            st.markdown(episode)
            episode_path = f"output/summaries/{safe_title}_episode.md"
            with open(episode_path, "w") as f:
                f.write(f"# Googly AI Episode: {selected_title}\n\n{episode}")

            st.download_button(
                "Download episode script",
                episode,
                file_name=f"googly_ai_{safe_title[:30]}.md"
            )
