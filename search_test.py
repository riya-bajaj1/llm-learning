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

query = input("Ask a question: ")
query_embedding = model.encode(query)

scores = util.cos_sim(query_embedding, doc_embeddings)[0]
ranked = sorted(zip(scores.tolist(), docs), reverse=True)

for score, doc in ranked[:3]:
    print(f"{score:.2f} | {doc}")
