from groq import Groq
from pydantic import BaseModel, ValidationError
from typing import Literal, Optional
import json

client = Groq()

class Ticket(BaseModel):
    issue: str
    order_id: Optional[str] = None
    urgency: Literal["low", "medium", "high"]
    category: Literal["shipping", "billing", "product", "other"]
    sentiment: Literal["positive", "neutral", "negative"]

# Part A: test the validator on hand-written data
good = {"issue": "late order", "order_id": "4521", "urgency": "high", "category": "shipping", "sentiment": "negative"}
bad = {"issue": "late order", "order_id": "4521", "urgency": "super urgent", "category": "shipping", "sentiment": "negative"}

print("Good data:", Ticket(**good))
try:
    Ticket(**bad)
except ValidationError as e:
    print("Bad data rejected:")
    print(e)

# Part B: the LLM loop, now with validation
system_prompt = """You are a support ticket classifier.
Read the customer message and return ONLY a JSON object with these keys:
- issue: short description of the problem
- order_id: the order number as a string, or null if there is none
- urgency: one of "low", "medium", "high"
- category: one of "shipping", "billing", "product", "other"
- sentiment: one of "positive", "neutral", "negative"
"""

messages_to_check = [
    "Hi, my order #4521 hasn't arrived and it's been 2 weeks. I'm really annoyed. Please fix this today!",
    "I was charged twice for my subscription this month. Can you check when you get time?",
    "Love the app! Just wondering if a dark mode is planned.",
]

print()
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
    try:
        ticket = Ticket(**json.loads(raw))
        print("OK    ", ticket.category, "|", ticket.urgency, "|", ticket.sentiment, "|", ticket.issue)
    except (json.JSONDecodeError, ValidationError) as e:
        print("FAILED:", e)
