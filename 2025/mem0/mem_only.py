from dotenv import load_dotenv
from mem0 import Memory

load_dotenv()

m = Memory()

messages = [
    {"role": "user", "content": "Hi, I'm Alex. I love basketball and gaming."},
    {"role": "assistant", "content": "Hey Alex! I'll remember your interests."},
]
m.add(messages, user_id="alex")

results = m.search("What do you know about me?", user_id="alex")
print(results)
