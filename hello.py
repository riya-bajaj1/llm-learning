from groq import Groq

client = Groq()

response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
        {"role": "user", "content": "Explain what an LLM is in 3 sentences."}
    ],
)

print(response.choices[0].message.content)
