from groq import Groq
from vectorstore.store import VectorStore, expand_query
from dotenv import load_dotenv
import os
import re

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
store = VectorStore()

GENERIC_PATTERNS = [
    "hi", "hello", "hey", "hii", "helo",
    "how are you", "what are you", "who are you",
    "what can you do", "help", "thanks", "thank you",
    "bye", "goodbye", "ok", "okay", "cool", "nice",
    "what is this", "how does this work"
]

GENERIC_RESPONSES = {
    "greeting": "Hey! I'm AnyChat AI. Upload a PDF, audio, or video file and ask me anything about it — I'll answer with exact page numbers and timestamps.",
    "what_are_you": "I'm AnyChat AI — a multimodal RAG chatbot. I can read PDFs (including scanned ones), transcribe audio, and understand video. Upload any file and ask me questions about it.",
    "what_can_you_do": "I can:\n- 📄 Answer questions from PDFs and Word docs (with page numbers)\n- 🎵 Answer from audio files (with timestamps)\n- 🎥 Answer from videos (transcript + visual scene understanding)\n\nJust upload a file and start asking!",
    "thanks": "You're welcome! Feel free to upload more files or ask more questions.",
    "default": "Hi there! Upload a document, audio, or video file using the panel on the left, then ask me anything about it."
}


def is_generic_message(question: str) -> bool:
    """Detect greetings and small talk."""
    q = question.lower().strip().rstrip("!?.").strip()
    return q in GENERIC_PATTERNS or len(q.split()) <= 1 and len(q) < 6


def get_generic_response(question: str) -> str:
    q = question.lower().strip()
    if any(w in q for w in ["what are you", "who are you", "what is anychat"]):
        return GENERIC_RESPONSES["what_are_you"]
    if any(w in q for w in ["what can you do", "how does this work", "help"]):
        return GENERIC_RESPONSES["what_can_you_do"]
    if any(w in q for w in ["thanks", "thank you", "great", "nice", "cool"]):
        return GENERIC_RESPONSES["thanks"]
    if any(w in q for w in ["hi", "hello", "hey", "hii"]):
        return GENERIC_RESPONSES["greeting"]
    return GENERIC_RESPONSES["default"]


def format_timestamp(seconds: float) -> str:
    """Convert seconds to mm:ss format."""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"


def build_context(results: list) -> str:
    """Turn retrieved chunks into a readable context block for the LLM."""
    context_parts = []
    for i, r in enumerate(results, 1):
        meta = r["metadata"]
        source = os.path.basename(meta["source"])
        source_type = meta["source_type"]

        # build citation label
        if "start_time" in meta:
            ts = format_timestamp(meta["start_time"])
            citation = f"[Source {i}: {source} at {ts}]"
        elif "page" in meta:
            citation = f"[Source {i}: {source} page {meta['page']}]"
        else:
            citation = f"[Source {i}: {source}]"

        context_parts.append(f"{citation}\n{r['text']}")

    return "\n\n".join(context_parts)


def parse_timestamp(text: str) -> float | None:
    """Extract timestamp from query like '1:38', '2:05', '0:45'"""
    match = re.search(r'(\d+):(\d{2})', text)
    if match:
        minutes = int(match.group(1))
        seconds = int(match.group(2))
        return float(minutes * 60 + seconds)
    return None


def search_by_timestamp(timestamp_sec: float, window: float = 30.0) -> list:
    """Fetch chunks whose time range covers the requested timestamp."""
    all_chunks = store.collection.get(include=["documents", "metadatas"])
    
    results = []
    for i, meta in enumerate(all_chunks["metadatas"]):
        start = meta.get("start_time")
        end = meta.get("end_time")
        
        if start is None:
            continue
            
        # check if timestamp falls within chunk window
        chunk_end = end if end else start + window
        if start <= timestamp_sec <= chunk_end:
            results.append({
                "text": all_chunks["documents"][i],
                "metadata": meta,
                "distance": 0.0,   # exact match
                "id": all_chunks["ids"][i]
            })
    
    return results


def parse_page_number(question: str):
    """Detect queries like 'page 4', 'page four', 'pg 4'"""
    match = re.search(r'\bpage\s+(\d+)\b', question.lower())
    if match:
        return int(match.group(1))
    return None


def search_by_page(page_number: int) -> list:
    """Fetch all chunks from a specific page."""
    all_chunks = store.collection.get(include=["documents", "metadatas"])
    results = []
    for i, meta in enumerate(all_chunks["metadatas"]):
        if meta.get("page") == page_number:
            results.append({
                "text": all_chunks["documents"][i],
                "metadata": meta,
                "distance": 0.0,
                "id": all_chunks["ids"][i]
            })
    return results


def ask(question: str, top_k: int = 6, source_filter: str = None) -> dict:
    if is_generic_message(question):
        return {"answer": get_generic_response(question), "sources": []}

    try:
        if source_filter and source_filter.lower() == "all":
            source_filter = None

        # ── page number query ────────────────────────────────────────────
        page_num = parse_page_number(question)
        if page_num is not None:
            print(f"[RAG] Page query detected: page {page_num}")
            results = search_by_page(page_num)
            if not results:
                results = store.search(expand_query(question), top_k=top_k, source_filter=source_filter)

        # ── timestamp query ──────────────────────────────────────────────
        elif parse_timestamp(question) is not None:
            timestamp = parse_timestamp(question)
            results = search_by_timestamp(timestamp)
            if not results:
                results = store.search(expand_query(question), top_k=top_k, source_filter=source_filter)

        # ── semantic search ──────────────────────────────────────────────
        else:
            results = store.search(expand_query(question), top_k=top_k, source_filter=source_filter)

    except Exception as e:
        return {"answer": f"Search error: {str(e)}", "sources": []}

    if not results:
        return {
            "answer": "I couldn't find relevant information in the uploaded documents. Try rephrasing or upload a relevant file first.",
            "sources": []
        }

    context = build_context(results)

    system_prompt = """You are a helpful assistant that answers questions based on the provided context.

Rules:
- Use information from the context to answer
- For inferential questions (moral, theme, summary, conclusion),
  synthesize an answer from the overall context even if the exact
  word doesn't appear
- Always cite your sources using [Source N] labels
- If answering from video/audio mention the timestamp
- If answering from a document mention the page number
- Only say you don't have information if the context is
  completely unrelated to the question
- Be concise and direct"""

    user_prompt = f"""Context:
{context}

Question: {question}

Answer with citations:"""

    # step 4 — call LLM
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",    # free, fast on Groq
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2,                  # low temp = factual, less hallucination
        max_tokens=512
    )

    answer = response.choices[0].message.content

    # step 5 — build sources list for the UI
    sources = []
    for r in results:
        meta = r["metadata"]
        source_info = {
            "file": os.path.basename(meta["source"]),
            "type": meta["source_type"],
            "relevance": round(1 - r["distance"], 3)
        }
        if "start_time" in meta:
            source_info["timestamp"] = format_timestamp(meta["start_time"])
        if "page" in meta:
            source_info["page"] = meta["page"]
        sources.append(source_info)

    return {"answer": answer, "sources": sources}