from pypdf import PdfReader
import json

CHUNK_SIZE = 800
OVERLAP = 150

reader = PdfReader("docs.pdf")
chunks = []

for page_number, page in enumerate(reader.pages, start=1):
    text = page.extract_text() or ""
    text = " ".join(text.split())
    start = 0
    while start < len(text):
        piece = text[start:start + CHUNK_SIZE]
        if len(piece.strip()) > 50:
            chunks.append({"id": len(chunks), "page": page_number, "text": piece})
        start = start + CHUNK_SIZE - OVERLAP

print("Pages:", len(reader.pages))
print("Chunks:", len(chunks))

if not chunks:
    print("No text found. This PDF may be scanned images; try a different one.")
else:
    print("Example chunk:", chunks[0])
    with open("chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
