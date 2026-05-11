from groq import Groq
from vectorstore.store import VectorStore
from dotenv import load_dotenv
import os
import re

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
store = VectorStore()


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


def ask(question: str, top_k: int = 4, source_filter: str = None) -> dict:
    
    """
    Full RAG pipeline:
    1. Embed question
    2. Retrieve top_k chunks
    3. Build prompt
    4. Call LLM
    5. Return answer + sources
    """

    # step 1 & 2 — retrieve
    try:
        if source_filter and source_filter.lower() == "all":
            source_filter = None

        # ── timestamp query detection ─────────────────────────────────────────────────────────────────────
        timestamp = parse_timestamp(question)
        if timestamp is not None:
            print(f"[RAG] Timestamp query detected: {timestamp}s")
            results = search_by_timestamp(timestamp)
            if not results:
                # fallback — find nearest chunk
                results = store.search(question, top_k=top_k, source_filter=source_filter)
        else:
            results = store.search(question, top_k=top_k, source_filter=source_filter)

        print(f"[RAG] Found {len(results)} results")

    except Exception as e:
        return {"answer": f"Search error: {str(e)}", "sources": []}

    if not results:
        return {
            "answer": "I couldn't find relevant information in the uploaded documents.",
            "sources": []
        }

    # step 3 — build prompt
    context = build_context(results)

    system_prompt = """You are a helpful assistant that answers questions based strictly on the provided context.

Rules:
- Only use information from the context below
- Always cite your sources using the [Source N] labels
- If answering from a video or audio, mention the timestamp (e.g. "at 02:14")
- If answering from a document, mention the page number
- If the context doesn't contain the answer, say "I don't have enough information to answer this"
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