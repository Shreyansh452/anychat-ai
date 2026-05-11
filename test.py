from faster_whisper import WhisperModel

model = WhisperModel("base", device="cpu", compute_type="int8")

segments, info = model.transcribe("data/harvard.wav", word_timestamps=True)

for segment in segments:
    print(f"[{segment.start:.2f}s --> {segment.end:.2f}s] {segment.text}")
    for word in segment.words:
        print(f"  word: '{word.word}' at {word.start:.2f}s")