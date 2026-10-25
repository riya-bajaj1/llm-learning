from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "I want to return my order and get a refund.",
    "How do I send back an item to get my money back?",
    "The weather in Bengaluru is pleasant today.",
    "My package has not arrived yet.",
]

embeddings = model.encode(sentences)

print("Shape:", embeddings.shape)
print("First 5 numbers of sentence 1:", embeddings[0][:5])
print()

for i in range(len(sentences)):
    for j in range(i + 1, len(sentences)):
        score = util.cos_sim(embeddings[i], embeddings[j]).item()
        print(f"{score:.2f} | {sentences[i]}  <->  {sentences[j]}")
