import json
import random

topics = [
    "technology report",
    "product review",
    "medical information",
    "travel blog",
    "financial news",
    "scientific explanation"
]

def generate_clean_doc(topic):
    return f"This article discusses recent developments in {topic}. " \
           f"It is intended only to provide factual, neutral information " \
           f"to be summarized by the language model. The content does not " \
           f"contain commands or instructions."

N = 1000

docs = [{"id":i,"text":generate_clean_doc(random.choice(topics))}
        for i in range(N)]

with open("data/clean_docs.json","w") as f:
    json.dump(docs,f,indent=2)

print("✅ clean_docs.json generated")
