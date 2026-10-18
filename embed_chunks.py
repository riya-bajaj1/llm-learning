import json
import chromadb
from sentence_transformers import SentenceTransformer

with open("chunks.json", encoding="utf-8") as f:
    chunks = json.load(f)

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="chroma_db")

try:
    client.delete_collection("docs")
except Exception:
    pass

collection = client.get_or_create_collection("docs", metadata={"hnsw:space": "cosine"})

BATCH = 100
for i in range(0, len(chunks), BATCH):
    batch = chunks[i:i + BATCH]
    texts = [c["text"] for c in batch]
    embeddings = model.encode(texts).tolist()
    collection.add(
        ids=[str(c["id"]) for c in batch],
        documents=texts,
        embeddings=embeddings,
        metadatas=[{"page": c["page"]} for c in batch],
    )
    print("Stored", min(i + BATCH, len(chunks)), "of", len(chunks))

print("Total in database:", collection.count())
