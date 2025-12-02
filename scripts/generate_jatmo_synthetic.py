import json
import random

TASKS = [
    "summarize this product review",
    "summarize this news article",
    "translate this paragraph to French"
]

def gen_input(task):
    return f"Task: {task}. Input example text number {random.randint(1,10000)}."

def teacher_output(task,input_text):
    return f"Teacher response for [{task}] → {input_text[:40]}..."

DATA=[]

N=800

for task in TASKS:
    for _ in range(N):
        inp=gen_input(task)
        out=teacher_output(task,inp)
        DATA.append({
            "task":task,
            "prompt":inp,
            "teacher_output":out
        })

with open("data/jatmo_train.json","w") as f:
    json.dump(DATA,f,indent=2)

print("✅ jatmo_train.json generated")
