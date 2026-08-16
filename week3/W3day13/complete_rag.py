# ============================================================
# 1. IMPORT REQUIRED LIBRARIES
# ============================================================

import os
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer


# ============================================================
# 2. LOAD SENTENCE TRANSFORMER MODEL
# ============================================================

# Sentence Transformer converts text into numerical vectors
# all-MiniLM-L6-v2 produces 384-dimensional embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")  # 384


# ============================================================
# 3. LOAD ENVIRONMENT VARIABLES AND API KEY
# ============================================================

load_dotenv()

my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("API key kaha hai bhai")


# ============================================================
# 4. INITIALIZE GROQ CLIENT
# ============================================================

client = Groq(api_key=my_api_key)
groqmodel = "llama-3.3-70b-versatile"


# ============================================================
# 5. KNOWLEDGE BASE / DOCUMENTS
# ============================================================

documents = [
    "Employees receive 24 days of paid leave per year.",

    "Employees work from the office on Tuesday, Wednesday and Thursday. "
    "Monday and Friday are optional work-from-home days.",

    "Employees receive Rs 3000 per month for gym reimbursement.",

    "Employees can claim Rs 2000 per month for home internet.",

    "Employees have a 90 day notice period."
]


# ============================================================
# 6. CREATE EMBEDDINGS FOR ALL DOCUMENTS
# ============================================================

# Convert every document into a vector embedding
document_embeddings = model.encode(documents)

# Check the size of the embeddings object in memory
print(sys.getsizeof(document_embeddings))


# ============================================================
# 7. COSINE SIMILARITY
# ============================================================

# Measures how similar two vectors are based on their direction.
# Higher cosine similarity = more semantically similar.
def cosine_similarity(a, b):
    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )


# ============================================================
# 8. RETRIEVAL
# ============================================================

# Compare the query embedding with every document embedding
# and return the document with the highest similarity score.
def retrieve(qembedding):
    scores = []  # 0.4

    for i, document in enumerate(document_embeddings):
        score = cosine_similarity(qembedding, document)
        scores.append((score, documents[i]))

    # Sort documents by similarity score in descending order
    scores.sort(reverse=True)

    return scores[0]  # line #0.9


# ============================================================
# 9. ASK LLM USING RETRIEVED CONTEXT
# ============================================================

def ask_llm(question, context):

    # Tell the LLM to answer only from the retrieved context
    # and avoid generating information outside the context.
    sys_prompt = f"""answer in one line only. Answer only based on this context. do not hallucinate. Context: {context}"""

    system_message = {
        "role": "system",
        "content": sys_prompt
    }

    message = {
        "role": "user",
        "content": question
    }

    messages = [system_message, message]

    response = client.chat.completions.create(
        model=groqmodel,
        messages=messages
    )

    answer = response.choices[0].message.content

    return answer


# ============================================================
# 10. QUERY
# ============================================================

query = "How much vacation do I get?"


# ============================================================
# 11. CREATE QUERY EMBEDDING
# ============================================================

# Convert the user's question into the same vector space
# as the document embeddings.
qembedding = model.encode(query)


# ============================================================
# 12. RETRIEVE MOST RELEVANT DOCUMENT
# ============================================================

score, context = retrieve(qembedding)


# ============================================================
# 13. GENERATE FINAL ANSWER USING THE LLM
# ============================================================

answer = ask_llm(query, context)

print(answer)