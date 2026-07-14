# -----------------------------
# Import Required Libraries
# -----------------------------

import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq


# -----------------------------
# Load Environment Variables
# -----------------------------

load_dotenv()
my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("API key kaha hai bhai")


# -----------------------------
# Initialize Groq Client
# -----------------------------

client = Groq(api_key=my_api_key)

model = "llama-3.3-70b-versatile"
role = "user"


# -----------------------------
# Output Structure
# -----------------------------

from pydantic import BaseModel

class Ticket(BaseModel):
    name: str
    email: str
    issue: str

# Generate JSON Schema from the Pydantic model
schema = Ticket.model_json_schema()

# Force the model to return a valid JSON object
response_format = {
    "type": "json_object"
}


# -----------------------------
# System Prompt
# -----------------------------

system_prompt = f"""
Extract the personal information from the ticket strictly based on this schema and give a json output.
{schema}
"""

message_system = {
    "role": "system",
    "content": system_prompt
}


# -----------------------------
# User Prompt
# -----------------------------

text = """
Hello My name is Pratyush.
Yesterday I broke up with my girlfriend Sheetal.
I have an iPhone which is not working at all.
My address is Delhi.
My email is abc@gmail.com.
My contact number is 82134.
"""

prompt = f"""
This is a customer ticket. Please extract the personal information from this.

{text}
"""

message = {
    "role": role,
    "content": prompt
}

messages = [message_system, message]


# -----------------------------
# Generate Response
# -----------------------------

response = client.chat.completions.create(
    model=model,
    messages=messages,
    response_format=response_format
)

answer = response.choices[0].message.content
print(answer)


# -----------------------------
# Parse JSON Response
# -----------------------------

import json

# Convert JSON string to Python dictionary
data_file = json.loads(answer)

# Validate the dictionary using the Pydantic model
ticket = Ticket(**data_file)


# -----------------------------
# Access Structured Data
# -----------------------------

print(ticket.name)
print(ticket.email)
print(ticket.issue)



#Homework

# take resume in pdf or word
# have hr give you a list of things like skill, experience, projects
# extract these from resume 
# match against the hr list
# generate a percentage of matching or not