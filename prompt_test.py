from groq import Groq

client = Groq()

message = "Hi, my order #4521 hasn't arrived and it's been 2 weeks. I'm really annoyed. Please fix this today!"

def ask(system_prompt):
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message},
        ],
        temperature=0,
    )
    return response.choices[0].message.content.strip()

vague = "Help with this customer message."
specific = "You are a support assistant. Read the customer message and reply with exactly one sentence stating the problem, then one word for urgency (low, medium or high). No extra text."

print("VAGUE:")
print(ask(vague))
print()
print("SPECIFIC:")
print(ask(specific))
