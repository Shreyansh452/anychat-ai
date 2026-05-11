from retrieval.rag import ask

questions = [
    "What is being discussed?",
    "What challenges are mentioned?",
    "Summarize the main points",
]

for q in questions:
    print(f"\n{'='*60}")
    print(f"Q: {q}")
    result = ask(q)
    print(f"\nA: {result['answer']}")
    print(f"\nSources:")
    for s in result['sources']:
        line = f"  - {s['file']} ({s['type']})"
        if 'timestamp' in s:
            line += f" at {s['timestamp']}"
        if 'page' in s:
            line += f" page {s['page']}"
        line += f" | relevance: {s['relevance']}"
        print(line)