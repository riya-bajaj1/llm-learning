import chromadb
from sentence_transformers import SentenceTransformer

QUESTION = "What methods can be used to physically acquire data from mobile devices, such as chip-off and JTAG?"

model = SentenceTransformer("all-MiniLM-L6-v2")
collection = chromadb.PersistentClient(path="chroma_db").get_collection("docs")

emb = model.encode(QUESTION).tolist()
res = collection.query(query_embeddings=[emb], n_results=collection.count())
metas = res["metadatas"][0]
dists = res["distances"][0]

best = {}
for rank, (m, d) in enumerate(zip(metas, dists), start=1):
    if m["page"] not in best:
        best[m["page"]] = (rank, 1 - d)

print("Top 12 chunks (rank | page | score):")
for rank in range(1, 13):
    print(rank, "|", metas[rank - 1]["page"], "|", round(1 - dists[rank - 1], 2))

print()
print("Best chunk rank for the pages we care about:")
for p in (250, 252, 257, 258, 259, 260):
    rank, score = best[p]
    print("page", p, "-> rank", rank, "| score", round(score, 2))