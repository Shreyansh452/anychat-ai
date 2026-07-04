import os
import fitz  # pymupdf
import docx
import cv2
import base64
from groq import Groq
from faster_whisper import WhisperModel
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()

# ── data structure every chunk must follow ──────────────────────────────────
@dataclass
class Chunk:
    text: str
    source: str
    source_type: str          # "pdf" | "docx" | "audio" | "video"
    chunk_index: int
    page: Optional[int] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    chunk_id: str = ""

    def __post_init__(self):
        self.chunk_id = f"{Path(self.source).stem}_{self.chunk_index}"


# ── chunking helper ─────────────────────────────────────────────────────────
def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks by word count."""
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


# ── PDF ─────────────────────────────────────────────────────────────────────
def pdf_page_to_base64(page) -> str:
    """Convert a PDF page to base64 image for vision model."""
    mat = fitz.Matrix(2.0, 2.0)   # 2x zoom for better resolution
    pix = page.get_pixmap(matrix=mat)
    img_bytes = pix.tobytes("jpeg")
    return base64.b64encode(img_bytes).decode("utf-8")


def ingest_pdf(filepath: str) -> List[Chunk]:
    chunks = []
    doc = fitz.open(filepath)

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text().strip()
        for idx, chunk_text_part in enumerate(chunk_text(text)):
            chunks.append(Chunk(
                text=chunk_text_part,
                source=filepath,
                source_type="pdf",
                chunk_index=len(chunks),
                page=page_num,
            ))

    print(f"[PDF] {len(chunks)} chunks from {filepath}")
    return chunks


# ── DOCX ────────────────────────────────────────────────────────────────────
def ingest_docx(filepath: str) -> List[Chunk]:
    chunks = []
    doc = docx.Document(filepath)
    full_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    for idx, part in enumerate(chunk_text(full_text)):
        chunks.append(Chunk(
            text=part,
            source=filepath,
            source_type="docx",
            chunk_index=idx,
        ))
    print(f"[DOCX] {len(chunks)} chunks from {filepath}")
    return chunks


# ── AUDIO ───────────────────────────────────────────────────────────────────
_whisper_model = None

def get_whisper():
    global _whisper_model
    if _whisper_model is None:
        print("[Whisper] Loading model...")
        _whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
    return _whisper_model

def ingest_audio(filepath: str, chunk_duration: float = 30.0) -> List[Chunk]:
    """Transcribe audio and chunk by time window."""
    model = get_whisper()
    segments, _ = model.transcribe(filepath, word_timestamps=True)

    # collect all words with timestamps
    words = []
    for seg in segments:
        for word in seg.words:
            words.append((word.word, word.start, word.end))

    # group words into ~chunk_duration second windows
    chunks, current_words, chunk_start = [], [], None
    for word, start, end in words:
        if chunk_start is None:
            chunk_start = start
        current_words.append(word)
        if (end - chunk_start) >= chunk_duration:
            chunks.append(Chunk(
                text=" ".join(current_words).strip(),
                source=filepath,
                source_type="audio",
                chunk_index=len(chunks),
                start_time=round(chunk_start, 2),
                end_time=round(end, 2),
            ))
            current_words, chunk_start = [], None

    # flush remaining words
    if current_words:
        chunks.append(Chunk(
            text=" ".join(current_words).strip(),
            source=filepath,
            source_type="audio",
            chunk_index=len(chunks),
            start_time=round(chunk_start, 2),
            end_time=round(words[-1][2], 2),
        ))

    print(f"[Audio] {len(chunks)} chunks from {filepath}")
    return chunks


# ── VIDEO ───────────────────────────────────────────────────────────────────
def extract_audio_from_video(video_path: str) -> str:
    """Extract audio track to a temp wav file using opencv + ffmpeg."""
    audio_path = video_path.rsplit(".", 1)[0] + "_extracted.wav"
    os.system(f'ffmpeg -y -i "{video_path}" -ac 1 -ar 16000 "{audio_path}" -loglevel quiet')
    return audio_path

def get_frame_interval(fps: float, duration_seconds: float) -> int:
    """Adaptive interval — shorter videos get more frames."""
    if duration_seconds < 120:        # under 2 min → every 20s
        return int(fps * 20)
    elif duration_seconds < 600:      # under 10 min → every 30s
        return int(fps * 30)
    else:                             # longer → every 60s
        return int(fps * 60)


def describe_keyframes(filepath: str) -> List[Chunk]:
    """Send keyframes to Groq Vision to get scene descriptions."""
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    chunks = []

    cap = cv2.VideoCapture(filepath)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = total_frames / fps

    interval_frames = get_frame_interval(fps, duration)
    print(f"[Vision] Video duration: {duration:.0f}s → frame every {interval_frames/fps:.0f}s")

    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_num % interval_frames == 0:
            try:
                _, buffer = cv2.imencode(".jpg", frame)
                image_b64 = base64.b64encode(buffer).decode("utf-8")

                response = groq_client.chat.completions.create(
                    model="meta-llama/llama-4-scout-17b-16e-instruct",
                    messages=[{
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_b64}"
                                }
                            },
                            {
                                "type": "text",
                                "text": "Describe this video frame in detail: who is visible, what text appears on screen, what is happening, what is the setting."
                            }
                        ]
                    }],
                    max_tokens=300
                )

                description = response.choices[0].message.content.strip()
                timestamp = round(frame_num / fps, 2)

                if description:
                    chunks.append(Chunk(
                        text=f"[Visual scene at {timestamp}s] {description}",
                        source=filepath,
                        source_type="video",
                        chunk_index=len(chunks),
                        start_time=timestamp,
                    ))

            except Exception as e:
                print(f"[Vision] Frame skipped: {e}")

        frame_num += 1
    cap.release()
    return chunks

def ingest_video(filepath: str) -> List[Chunk]:
    chunks = []

    # 1. audio transcript via Whisper
    audio_path = extract_audio_from_video(filepath)
    if os.path.exists(audio_path):
        audio_chunks = ingest_audio(audio_path)
        for c in audio_chunks:
            c.source = filepath
            c.source_type = "video"
        chunks.extend(audio_chunks)
        os.remove(audio_path)

    # 2. visual scene understanding via Groq Vision (replaces pytesseract)
    vision_chunks = describe_keyframes(filepath)
    for i, c in enumerate(vision_chunks):
        c.chunk_index = len(chunks) + i
    chunks.extend(vision_chunks)

    print(f"[Video] {len(chunks)} total chunks from {filepath}")
    return chunks


# ── unified entry point ──────────────────────────────────────────────────────
def ingest_file(filepath: str) -> List[Chunk]:
    ext = Path(filepath).suffix.lower()
    if ext == ".pdf":
        return ingest_pdf(filepath)
    elif ext == ".docx":
        return ingest_docx(filepath)
    elif ext in (".mp3", ".wav", ".m4a", ".flac"):
        return ingest_audio(filepath)
    elif ext in (".mp4", ".mkv", ".avi", ".mov"):
        return ingest_video(filepath)
    else:
        raise ValueError(f"Unsupported file type: {ext}")