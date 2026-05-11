import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
from ingestion.ingestor import ingest_file
from vectorstore.store import VectorStore

# Initialize store
store = VectorStore()

# Ingest a file (change to your file)
chunks = ingest_file("data/SampleVideo.mp4")
print(f"\nIngested {len(chunks)} chunks")

# Add to vector store
store.add_chunks(chunks)

# Search test
query = "what is being discussed?"
print(f"\n🔍 Searching for: '{query}'")
results = store.search(query, top_k=3)

for i, result in enumerate(results, 1):
    print(f"\n--- Result {i} (distance: {result['distance']:.3f}) ---")
    print(f"Source: {result['metadata']['source']}")
    print(f"Type: {result['metadata']['source_type']}")
    if 'start_time' in result['metadata']:
        print(f"Timestamp: {result['metadata']['start_time']}s")
    if 'page' in result['metadata']:
        print(f"Page: {result['metadata']['page']}")
    print(f"Text: {result['text'][:150]}...")