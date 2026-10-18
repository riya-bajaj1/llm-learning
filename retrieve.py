import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection("docs")

TOP_K = 3

while True:
    question = input("\nAsk a question (or 'quit'): ")
    if question.lower() == "quit":
        break

    q_emb = model.encode(question).tolist()
    results = collection.query(query_embeddings=[q_emb], n_results=TOP_K)

    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        score = 1 - dist
        print(f"{score:.2f} | page {meta['page']} | {doc[:300]}")
