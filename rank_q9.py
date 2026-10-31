import chromadb
from sentence_transformers import SentenceTransformer

QUERIES = {
    "plain question": "What methods can be used to physically acquire data from mobile devices, such as chip-off and JTAG?",
    "heading-style query": "chip-off in vitro acquisition JTAG in-system acquisition mobile device flash memory",
}
PAGES = (250, 252, 257, 258, 259, 260)

model = SentenceTransformer("all-MiniLM-L6-v2")
collection = chromadb.PersistentClient(path="chroma_db").get_collection("docs")

for name, query in QUERIES.items():
    emb = model.encode(query).tolist()
    res = collection.query(query_embeddings=[emb], n_results=collection.count())
    metas = res["metadatas"][0]
    dists = res["distances"][0]

    best = {}
    for rank, (m, d) in enumerate(zip(metas, dists), start=1):
        if m["page"] not in best:
            best[m["page"]] = (rank, 1 - d)

    print(f"\n=== {name} ===")
    print("Top 8 pages:", [m["page"] for m in metas[:8]])
    for p in PAGES:
        rank, score = best[p]
        print(f"page {p} -> best chunk rank {rank} | score {score:.2f}")