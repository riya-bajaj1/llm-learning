from groq import Groq
import json

client = Groq()

system_prompt = """You are a support ticket classifier.
Read the customer message and return ONLY a JSON object with these keys:
- issue: short description of the problem
- order_id: the order number as a string, or null if there is none
- urgency: one of "low", "medium", "high"
- category: one of "shipping", "billing", "product", "other"
"""

messages_to_check = [
    "Hi, my order #4521 hasn't arrived and it's been 2 weeks. I'm really annoyed. Please fix this today!",
    "I was charged twice for my subscription this month. Can you check when you get time?",
    "Love the app! Just wondering if a dark mode is planned.",
]

for text in messages_to_check:
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content
    data = json.loads(raw)
    print(data["category"], "|", data["urgency"], "|", data["issue"])
