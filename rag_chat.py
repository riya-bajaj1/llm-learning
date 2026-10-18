import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq

model = SentenceTransformer("all-MiniLM-L6-v2")
db = chromadb.PersistentClient(path="chroma_db")
collection = db.get_collection("docs")
llm = Groq()

TOP_K = 4
THRESHOLD = 0.40

SYSTEM = """You answer questions using ONLY the provided context excerpts from a book.
Each excerpt is labeled with its page number.
Rules:
- Use only the context. Do not use outside knowledge.
- Cite pages in your answer like (p. 60).
- If the context does not contain the answer, reply exactly: "The document does not contain enough information to answer that."
- Keep the answer under 150 words, in plain text."""

while True:
    question = input("\nAsk a question (or 'quit'): ")
    if question.lower() == "quit":
        break

    q_emb = model.encode(question).tolist()
    res = collection.query(query_embeddings=[q_emb], n_results=TOP_K)
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    scores = [1 - d for d in res["distances"][0]]

    if scores[0] < THRESHOLD:
        print(f"[best score {scores[0]:.2f}] I don't have information about that in the document.")
        continue

    context = "\n\n".join(f"[Page {m['page']}] {d}" for d, m in zip(docs, metas))
    response = llm.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
        temperature=0,
    )
    print(response.choices[0].message.content.strip())
    print(f"[best score {scores[0]:.2f}] Retrieved pages:", sorted(set(m["page"] for m in metas)))
