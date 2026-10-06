from groq import Groq

client = Groq()
prompt = "Give me a one-line slogan for a coffee shop."

for temp in (0.0, 1.0, 1.5):
    print(f"--- temperature {temp} ---")
    for i in range(3):
        r = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": prompt}],
            temperature=temp,
        )
        print(r.choices[0].message.content.strip())
