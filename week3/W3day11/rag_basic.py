# ============================================================
# BASIC RAG (Retrieval-Augmented Generation) IMPLEMENTATION
# ============================================================
# This program demonstrates a very simple RAG pipeline:
#
# 1. Store information in a small knowledge base
# 2. Retrieve relevant information based on the user's question
# 3. Pass the retrieved information as context to the LLM
# 4. Ask the LLM to generate an answer using only that context
#
# This is a simplified example for understanding the core
# concept of RAG. Real-world RAG systems usually use
# embeddings + vector databases for retrieval.
# ============================================================


# -------------------- IMPORTS --------------------

import os
from dotenv import load_dotenv
from groq import Groq


# -------------------- API KEY SETUP --------------------

# Load environment variables from the .env file.
load_dotenv()

# Read the Groq API key from the environment variable.
my_api_key = os.getenv("GROQ_API_KEY")

# Stop the program if the API key is not available.
if not my_api_key:
    raise ValueError("API key kaha hai bhai")


# -------------------- GROQ CLIENT SETUP --------------------

# Create the Groq client using the API key.
client = Groq(api_key=my_api_key)

# Specify the LLM model that will generate the final answer.
model = "llama-3.3-70b-versatile"


# ============================================================
# STEP 1: KNOWLEDGE BASE
# ============================================================
# In RAG, we first need some external knowledge/data from which
# the system can retrieve information.
#
# Here we are using a simple Python dictionary as our
# knowledge base.
#
# In a real RAG system, this could contain documents,
# PDFs, webpages, databases, etc.
# ============================================================

knowledge_base = {
    "age": "The age of Pratyush is 25 years",
    "net worth": "The net worth of Pratyush is 2000"
}


# ============================================================
# STEP 2: RETRIEVAL
# ============================================================
# The retrieval step finds information relevant to the
# user's question from the knowledge base.
#
# This is a VERY simple retrieval method:
# - Convert the question to lowercase
# - Check whether "age" or "net worth" appears in it
# - Return the corresponding information
#
# Real-world RAG usually performs semantic retrieval using
# embeddings and a vector database.
# ============================================================

def retrieve_info(question):

    # Convert the question to lowercase so that matching
    # is not affected by uppercase/lowercase differences.
    question = question.lower()

    # If the question contains "age", retrieve the age information.
    if "age" in question:
        return knowledge_base["age"]

    # If the question contains "net worth", retrieve the
    # net worth information.
    elif "net worth" in question:
        return knowledge_base["net worth"]

    # If nothing relevant is found, return None.
    else:
        return None


# ============================================================
# STEP 3: AUGMENT THE LLM PROMPT + GENERATE ANSWER
# ============================================================
# This function combines:
#
#     User Question
#           +
#     Retrieved Context
#           ↓
#          LLM
#           ↓
#     Final Answer
#
# The retrieved information is inserted into the system prompt
# so that the LLM can use it as context when answering.
# ============================================================

def ask_llm(question):

    # First retrieve relevant information from the knowledge base.
    context = retrieve_info(question)

    # Create a system prompt that tells the LLM:
    # - Answer in one line
    # - Use only the retrieved context
    # - Do not make up information
    sys_prompt = f"""
    Answer in one line only.
    Answer only based on this context.
    Do not hallucinate.
    
    Context: {context}
    """

    # Create the system message.
    # The system message controls how the LLM should behave.
    system_message = {
        "role": "system",
        "content": sys_prompt
    }

    # Create the user message containing the original question.
    message = {
        "role": "user",
        "content": question
    }

    # Combine the system instruction and user's question
    # into the message list expected by the chat API.
    messages = [system_message, message]

    # Send the messages to the LLM and generate a response.
    response = client.chat.completions.create(
        model=model,
        messages=messages
    )

    # Extract the actual text answer from the API response.
    answer = response.choices[0].message.content

    return answer


# ============================================================
# STEP 4: ASK A QUESTION
# ============================================================

question = "What is Pratyush's age?"

# Run the complete RAG pipeline:
#
# Question
#    ↓
# Retrieval
#    ↓
# Retrieved Context
#    ↓
# LLM
#    ↓
# Answer
#
# Print the final answer.
print(ask_llm(question))