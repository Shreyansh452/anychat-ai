import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
from ingestion.ingestor import ingest_file
# swap in any file you have locally
chunks = ingest_file("data/SampleVideo.mp4")   

for c in chunks[:3]:
    print(f"\n--- chunk {c.chunk_id} ---")
    print(f"  source_type : {c.source_type}")
    print(f"  page        : {c.page}")
    print(f"  start_time  : {c.start_time}")
    print(f"  text        : {c.text[:120]}...")