import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq

TOP_K = 8
THRESHOLD = 0.40

SYSTEM = """You answer questions using ONLY the provided context excerpts from a book.
Each excerpt is labeled with its page number.
Rules:
- Use only the context. Do not use outside knowledge.
- Cite pages in your answer like (p. 60).
- If the context does not contain the answer, reply exactly: "The document does not contain enough information to answer that."
- Keep the answer under 150 words, in plain text."""


@st.cache_resource
def load():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    db = chromadb.PersistentClient(path="chroma_db")
    collection = db.get_collection("docs")
    llm = Groq()
    return model, collection, llm


model, collection, llm = load()

st.title("Digital Forensics Book Q&A")
st.caption("Answers come only from the book, with page numbers. If the book doesn't say, the bot says so.")

question = st.text_input("Ask a question about the book")

if question:
    q_emb = model.encode(question).tolist()
    res = collection.query(query_embeddings=[q_emb], n_results=TOP_K)
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    scores = [1 - d for d in res["distances"][0]]

    if scores[0] < THRESHOLD:
        st.warning("I don't have information about that in the document.")
        st.caption(f"Best match score: {scores[0]:.2f} (below the {THRESHOLD} threshold)")
    else:
        context = "\n\n".join(f"[Page {m['page']}] {d}" for d, m in zip(docs, metas))
        with st.spinner("Thinking..."):
            r = llm.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
                ],
                temperature=0,
            )
        st.write(r.choices[0].message.content.strip())
        with st.expander("Sources used"):
            for d, m, s in zip(docs, metas, scores):
                st.markdown(f"**Page {m['page']}** (similarity {s:.2f})")
                st.write(d)