import json
import time
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq

model = SentenceTransformer("all-MiniLM-L6-v2")
db = chromadb.PersistentClient(path="chroma_db")
collection = db.get_collection("docs")
llm = Groq()

TOP_K = 8
THRESHOLD = 0.40
REFUSAL = "does not contain enough information"

SYSTEM = """You answer questions using ONLY the provided context excerpts from a book.
Each excerpt is labeled with its page number.
Rules:
- Use only the context. Do not use outside knowledge.
- Cite pages in your answer like (p. 60).
- If the context does not contain the answer, reply exactly: "The document does not contain enough information to answer that."
- Keep the answer under 150 words, in plain text."""

with open("questions.json", encoding="utf-8") as f:
    questions = json.load(f)

results = []
for i, q in enumerate(questions, start=1):
    emb = model.encode(q["question"]).tolist()
    res = collection.query(query_embeddings=[emb], n_results=TOP_K)
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    scores = [1 - d for d in res["distances"][0]]
    pages = sorted(set(m["page"] for m in metas))

    if scores[0] < THRESHOLD:
        answer = "REFUSED (below threshold, LLM not called)"
        refused = True
    else:
        context = "\n\n".join(f"[Page {m['page']}] {d}" for d, m in zip(docs, metas))
        r = llm.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {q['question']}"},
            ],
            temperature=0,
        )
        answer = r.choices[0].message.content.strip()
        refused = REFUSAL in answer.lower()
        time.sleep(1)

    expected = q["expected_pages"]
    found = [p for p in expected if p in pages]
    hit = len(found) > 0

    if q["type"] == "none":
        passed = refused
    else:
        passed = hit and not refused

    results.append({
        "question": q["question"], "type": q["type"], "top_score": round(scores[0], 2),
        "expected_pages": expected, "retrieved_pages": pages, "found": found,
        "refused": refused, "passed": passed, "answer": answer,
    })

    hit_text = "-" if q["type"] == "none" else ("Y" if hit else "N")
    status = "REFUSED" if refused else "ANSWERED"
    verdict = "PASS" if passed else "FAIL"
    print(f"{i:>2} {q['type']:<6} score={scores[0]:.2f} hit={hit_text} {status:<8} {verdict}  {q['question'][:55]}")

answerable = [r for r in results if r["type"] != "none"]
unanswerable = [r for r in results if r["type"] == "none"]
print()
print("Retrieval hit rate (answerable):", sum(1 for r in answerable if r["found"]), "of", len(answerable))
print("Answered correctly (answerable):", sum(1 for r in answerable if r["passed"]), "of", len(answerable))
print("Correct refusals (not in book): ", sum(1 for r in unanswerable if r["passed"]), "of", len(unanswerable))

with open("results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("Saved full answers to results.json")