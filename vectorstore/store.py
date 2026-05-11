import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List
from ingestion.ingestor import Chunk
import hashlib

# version hash for chunk config — auto-separates old data if params change
CHUNK_CONFIG_VERSION = hashlib.md5(
    b"chunk_size=400_overlap=50_model=MiniLM"
).hexdigest()[:8]

# global embedding model (loaded once)
_embed_model = None

def get_embedding_model():
    global _embed_model
    if _embed_model is None:
        print("[Embeddings] Loading all-MiniLM-L6-v2...")
        _embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _embed_model


class VectorStore:
    def __init__(self, persist_dir: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        # collection name includes version — auto-separates old data
        self.collection = self.client.get_or_create_collection(
            name=f"rag_chunks_{CHUNK_CONFIG_VERSION}",
            metadata={"hnsw:space": "cosine"}
        )
        self.embed_model = get_embedding_model()

    def add_chunks(self, chunks: List[Chunk]):
        """Embed and store chunks in ChromaDB."""
        if not chunks:
            return

        texts = [c.text for c in chunks]
        embeddings = self.embed_model.encode(texts, show_progress_bar=True).tolist()

        # prepare metadata for each chunk
        metadatas = []
        for c in chunks:
            meta = {
                "source": c.source,
                "source_type": c.source_type,
                "chunk_index": c.chunk_index,
            }
            if c.page is not None:
                meta["page"] = c.page
            if c.start_time is not None:
                meta["start_time"] = c.start_time
            if c.end_time is not None:
                meta["end_time"] = c.end_time
            metadatas.append(meta)

        ids = [c.chunk_id for c in chunks]

        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        print(f"[VectorStore] Added {len(chunks)} chunks to ChromaDB")

    def search(self, query: str, top_k: int = 5, source_filter: str = None):
        """Search for relevant chunks."""
        query_embedding = self.embed_model.encode([query]).tolist()

        try:
            if source_filter:
                results = self.collection.query(
                    query_embeddings=query_embedding,
                    n_results=top_k,
                    where={"source_type": {"$eq": source_filter}}  # explicit $eq operator
                )
            else:
                results = self.collection.query(
                    query_embeddings=query_embedding,
                    n_results=top_k,
                )
        except Exception as e:
            print(f"[VectorStore] Search error: {e}")
            return []

        chunks = []
        for i in range(len(results['ids'][0])):
            chunks.append({
                "text": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "distance": results['distances'][0][i],
                "id": results['ids'][0][i]
            })
        return chunks

    def clear(self):
        """Delete all chunks."""
        try:
            self.client.delete_collection(self.collection.name)
        except Exception:
            pass  # already gone, no problem
        
        # recreate fresh empty collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"}
        )
        print("[VectorStore] Cleared all chunks")