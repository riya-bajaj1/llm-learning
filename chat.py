from groq import Groq

client = Groq()

messages = [
    {"role": "system", "content": "You are a helpful tutor. Answer in plain text only, no markdown, no tables, no emojis. Keep answers under 120 words."}
]

print("Chatbot ready. Type 'quit' to exit.")

while True:
    user_input = input("You: ")
    if user_input.lower() in ("quit", "exit"):
        break

    messages.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=messages,
    )

    reply = response.choices[0].message.content
    print("Bot:", reply)

    messages.append({"role": "assistant", "content": reply})
