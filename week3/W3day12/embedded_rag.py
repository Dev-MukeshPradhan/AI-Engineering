import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
import numpy as np
from sentence_transformers import SentenceTransformer

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


model = SentenceTransformer("all-MiniLM-L6-v2") #384
text = "Machine learning is fun."

# embedding=model.encode(text)
# print(embedding.shape)
# print(embedding[:10])

t1="There are 24 vacation paid leaves."
t2="I like cute cats."

v1=model.encode(t1)
v2=model.encode(t2)
print(cosine_similarity(v1, v2))