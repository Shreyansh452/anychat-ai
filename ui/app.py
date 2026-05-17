import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
from ingestion.ingestor import ingest_file
from vectorstore.store import VectorStore
from retrieval.rag import ask
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

store = VectorStore()
ingested_files = set()


def ingest_uploaded_file(file):
    if file is None:
        return "No file uploaded.", get_file_list(), None

    filepath = file.name
    filename = Path(filepath).name

    if filename in ingested_files:
        return f"'{filename}' is already indexed.", get_file_list(), None

    try:
        chunks = ingest_file(filepath)
        store.add_chunks(chunks)
        ingested_files.add(filename)
        return f"✅ '{filename}' indexed — {len(chunks)} chunks added.", get_file_list(), None
    except Exception as e:
        return f"❌ Error: {str(e)}", get_file_list(), None


def clear_all():
    try:
        store.clear()
        ingested_files.clear()
        return "✅ All data cleared.", "No files indexed yet.", []
    except Exception as e:
        return f"❌ Clear failed: {str(e)}", get_file_list(), []


def get_file_list():
    if not ingested_files:
        return "No files indexed yet."
    return "\n".join(f"• {f}" for f in ingested_files)


def format_sources(sources: list) -> str:
    if not sources:
        return ""
    lines = ["\n\n---\n📎 **Sources:**"]
    for s in sources:
        line = f"- `{s['file']}` ({s['type']})"
        if "timestamp" in s:
            line += f" · ⏱ {s['timestamp']}"
        if "page" in s:
            line += f" · 📄 page {s['page']}"
        line += f" · relevance: {s['relevance']}"
        lines.append(line)
    return "\n".join(lines)


def chat(message, history, source_filter):
    if not isinstance(history, list):
        history = []

    if not message.strip():
        return history, ""

    history = history + [{"role": "user", "content": message}]

    if not ingested_files:
        history = history + [{"role": "assistant", "content": "⚠️ Please upload and index a file first."}]
        return history, ""

    try:
        filter_val = None if source_filter == "All" else source_filter.lower()
        result = ask(message, top_k=4, source_filter=filter_val)
        answer = result["answer"] + format_sources(result["sources"])
    except Exception as e:
        answer = f"❌ Error calling LLM: {str(e)}"

    history = history + [{"role": "assistant", "content": answer}]
    return history, ""


# ── UI ────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="AnyChat AI") as demo:

    gr.Markdown("# 🚀 AnyChat AI\n### Chat with your documents, audio, and video files")

    with gr.Row():

        with gr.Column(scale=1):
            gr.Markdown("### 📁 Upload Files")
            upload = gr.File(
                label="PDF, DOCX, MP3, WAV, MP4",
                file_types=[".pdf", ".docx", ".mp3", ".wav", ".m4a", ".mp4", ".mkv"],
            )
            ingest_btn = gr.Button("Index File", variant="primary")
            ingest_status = gr.Textbox(
                label="Status", interactive=False, lines=2
            )
            gr.Markdown("### 📂 Indexed Files")
            file_list = gr.Textbox(
                value="No files indexed yet.",
                interactive=False,
                lines=5,
                label=""
            )
            clear_btn = gr.Button("🗑️ Clear All Data", variant="stop")

            gr.Markdown("### 🔍 Filter by type")
            source_filter = gr.Radio(
                choices=["All", "pdf", "docx", "audio", "video"],
                value="All",
                label=""
            )

        with gr.Column(scale=2):
            gr.Markdown("### 💬 Ask Questions")
            chatbot = gr.Chatbot(
                value=[],
                label="",
                height=460,
            )
            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Ask anything about your uploaded files...",
                    label="",
                    scale=5,
                    lines=1,
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)

            gr.Examples(
                examples=[
                    "What is the main topic?",
                    "Summarize the key points",
                    "What challenges are mentioned?",
                ],
                inputs=msg_input,
                label="Try these"
            )

    # ── events ────────────────────────────────────────────────────────────────
    ingest_btn.click(
        ingest_uploaded_file,
        inputs=[upload],
        outputs=[ingest_status, file_list, upload]   # upload resets to None
    )

    clear_btn.click(
        clear_all,
        inputs=[],
        outputs=[ingest_status, file_list, chatbot]
    )

    send_btn.click(
        chat,
        inputs=[msg_input, chatbot, source_filter],
        outputs=[chatbot, msg_input]
    )

    msg_input.submit(
        chat,
        inputs=[msg_input, chatbot, source_filter],
        outputs=[chatbot, msg_input]
    )


if __name__ == "__main__":
    demo.launch(
        share=True,
        theme=gr.themes.Soft(),
        max_file_size="500mb"
    )
