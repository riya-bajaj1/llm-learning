from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

docs = [
    "Refunds are processed within 5-7 business days after we receive the returned item.",
    "Standard delivery takes 3 to 5 working days across India.",
    "You can reset your password from the login page by clicking 'Forgot password'.",
    "Our support team is available Monday to Saturday, 9 am to 6 pm.",
    "Orders can be cancelled before they are shipped from the My Orders page.",
]

doc_embeddings = model.encode(docs)

THRESHOLD = 0.45

while True:
    query = input("Ask a question (or 'quit'): ")
    if query.lower() == "quit":
        break

    query_embedding = model.encode(query)
    scores = util.cos_sim(query_embedding, doc_embeddings)[0]
    best_score, best_doc = max(zip(scores.tolist(), docs))

    if best_score < THRESHOLD:
        print(f"{best_score:.2f} | I don't have information about that.")
    else:
        print(f"{best_score:.2f} | {best_doc}")
