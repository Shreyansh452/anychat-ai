import sys
import os
import uuid
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
from ingestion.ingestor import ingest_file
from vectorstore.store import VectorStore
from retrieval.rag import ask
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()


def create_session_store():
    """Create a unique VectorStore for this browser session."""
    session_id = str(uuid.uuid4())[:8]
    return VectorStore(collection_name=f"session_{session_id}")


def create_empty_set():
    return set()


def ingest_uploaded_file(file, store, files_set):
    if file is None:
        return "No file uploaded.", "No files indexed yet.", None, store, files_set

    filepath = file.name
    filename = os.path.basename(filepath)

    if filename in files_set:
        file_list = "\n".join(f"• {f}" for f in files_set)
        return f"'{filename}' is already indexed.", file_list, None, store, files_set

    try:
        chunks = ingest_file(filepath)
        store.add_chunks(chunks)
        files_set.add(filename)
        file_list = "\n".join(f"• {f}" for f in files_set)
        return f"✅ '{filename}' indexed — {len(chunks)} chunks added.", file_list, None, store, files_set
    except Exception as e:
        file_list = "\n".join(f"• {f}" for f in files_set) if files_set else "No files indexed yet."
        return f"❌ Error: {str(e)}", file_list, None, store, files_set


def clear_all(store, files_set):
    try:
        store.clear()
        files_set.clear()
        return "🗑️ All data cleared.", "No files indexed yet.", [], store, files_set
    except Exception as e:
        return f"❌ Clear failed: {str(e)}", "No files indexed yet.", [], store, files_set


def chat(message, history, source_filter, store):
    if not isinstance(history, list):
        history = []
    if not message.strip():
        return history, ""

    history = history + [{"role": "user", "content": message}]

    try:
        result = ask(message, source_filter=source_filter, store=store)
        answer = result["answer"]
    except Exception as e:
        answer = f"❌ Error: {str(e)}"

    history = history + [{"role": "assistant", "content": answer}]
    return history, ""


# ── UI ────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="AnyChat AI") as demo:

    # per-session state — unique to each browser session, invisible to other users
    session_store = gr.State(create_session_store)
    session_files = gr.State(create_empty_set)

    gr.Markdown("# 🧠 AnyChat AI\n### Chat with your documents, audio, and video files")

    with gr.Row():

        with gr.Column(scale=1):
            gr.Markdown("### 📁 Upload Files")
            upload = gr.File(
                label="PDF, DOCX, MP3, WAV, MP4",
                file_types=[".pdf", ".docx", ".mp3", ".wav", ".m4a", ".mp4", ".mkv"],
            )
            ingest_btn = gr.Button("Index File", variant="primary")
            ingest_status = gr.Textbox(label="Status", interactive=False, lines=2)

            gr.Markdown("### 📂 Indexed Files")
            file_list = gr.Textbox(value="No files indexed yet.", interactive=False, lines=5, label="")
            clear_btn = gr.Button("🗑️ Clear All Data", variant="stop")

            gr.Markdown("### 🔍 Filter by type")
            source_filter = gr.Radio(choices=["All", "pdf", "docx", "audio", "video"], value="All", label="")

        with gr.Column(scale=2):
            gr.Markdown("### 💬 Ask Questions")
            chatbot = gr.Chatbot(value=[], label="", height=460)
            with gr.Row():
                msg_input = gr.Textbox(placeholder="Ask anything about your uploaded files...", label="", scale=5, lines=1)
                send_btn = gr.Button("Send", variant="primary", scale=1)

            gr.Examples(
                examples=["What is the main topic?", "Summarize the key points", "What challenges are mentioned?"],
                inputs=msg_input,
                label="Try these"
            )

    # ── events ────────────────────────────────────────────────────────────────
    ingest_btn.click(
        ingest_uploaded_file,
        inputs=[upload, session_store, session_files],
        outputs=[ingest_status, file_list, upload, session_store, session_files]
    )

    clear_btn.click(
        clear_all,
        inputs=[session_store, session_files],
        outputs=[ingest_status, file_list, chatbot, session_store, session_files]
    )

    send_btn.click(
        chat,
        inputs=[msg_input, chatbot, source_filter, session_store],
        outputs=[chatbot, msg_input]
    )

    msg_input.submit(
        chat,
        inputs=[msg_input, chatbot, source_filter, session_store],
        outputs=[chatbot, msg_input]
    )


if __name__ == "__main__":
    demo.launch(
        share=True,
        theme=gr.themes.Soft(),
        max_file_size="500mb"
    )