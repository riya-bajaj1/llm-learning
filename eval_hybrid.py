import json
import re
import time
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from groq import Groq

model = SentenceTransformer("all-MiniLM-L6-v2")
db = chromadb.PersistentClient(path="chroma_db")
collection = db.get_collection("docs")
llm = Groq()

TOP_K = 8
THRESHOLD = 0.40
RRF_K = 60
CANDIDATES = 50
REFUSAL = "does not contain enough information"

SYSTEM = """You answer questions using ONLY the provided context excerpts from a book.
Each excerpt is labeled with its page number.
Rules:
- Use only the context. Do not use outside knowledge.
- Cite pages in your answer like (p. 60).
- If the context does not contain the answer, reply exactly: "The document does not contain enough information to answer that."
- Keep the answer under 150 words, in plain text."""

with open("chunks.json", encoding="utf-8") as f:
    chunks = json.load(f)


def tokenize(text):
    text = text.lower().replace("\ufb01", "fi").replace("\ufb02", "fl")
    return re.findall(r"[a-z0-9]+", text)


bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])


def hybrid_search(question):
    emb = model.encode(question).tolist()
    res = collection.query(query_embeddings=[emb], n_results=CANDIDATES)
    vector_ids = [int(i) for i in res["ids"][0]]
    best_vector_score = 1 - res["distances"][0][0]

    keyword_scores = bm25.get_scores(tokenize(question))
    keyword_ids = sorted(range(len(chunks)), key=lambda i: -keyword_scores[i])[:CANDIDATES]

    fused = {}
    for rank, cid in enumerate(vector_ids, start=1):
        fused[cid] = fused.get(cid, 0) + 1 / (RRF_K + rank)
    for rank, cid in enumerate(keyword_ids, start=1):
        fused[cid] = fused.get(cid, 0) + 1 / (RRF_K + rank)

    top_ids = sorted(fused, key=lambda c: -fused[c])[:TOP_K]
    return [chunks[i] for i in top_ids], best_vector_score


with open("questions.json", encoding="utf-8") as f:
    questions = json.load(f)

results = []
for i, q in enumerate(questions, start=1):
    top_chunks, best_score = hybrid_search(q["question"])
    pages = sorted(set(c["page"] for c in top_chunks))

    if best_score < THRESHOLD:
        answer = "REFUSED (below threshold, LLM not called)"
        refused = True
    else:
        context = "\n\n".join(f"[Page {c['page']}] {c['text']}" for c in top_chunks)
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
    passed = refused if q["type"] == "none" else (hit and not refused)

    results.append({
        "question": q["question"], "type": q["type"], "top_score": round(best_score, 2),
        "expected_pages": expected, "retrieved_pages": pages, "found": found,
        "refused": refused, "passed": passed, "answer": answer,
    })

    hit_text = "-" if q["type"] == "none" else ("Y" if hit else "N")
    status = "REFUSED" if refused else "ANSWERED"
    verdict = "PASS" if passed else "FAIL"
    print(f"{i:>2} {q['type']:<6} score={best_score:.2f} hit={hit_text} {status:<8} {verdict}  {q['question'][:55]}")

answerable = [r for r in results if r["type"] != "none"]
unanswerable = [r for r in results if r["type"] == "none"]
print()
print("Retrieval hit rate (answerable):", sum(1 for r in answerable if r["found"]), "of", len(answerable))
print("Answered correctly (answerable):", sum(1 for r in answerable if r["passed"]), "of", len(answerable))
print("Correct refusals (not in book): ", sum(1 for r in unanswerable if r["passed"]), "of", len(unanswerable))

with open("results_hybrid.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("Saved full answers to results_hybrid.json")