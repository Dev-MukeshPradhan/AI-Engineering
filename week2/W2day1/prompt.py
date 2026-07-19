import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
my_api_key=os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("Api Key id=s missing")


client=Groq(api_key=my_api_key)

model="llama-3.3-70b-versatile"
role="user"

def get_response(prompt):
    message={
        "role": "user",
        "content": prompt
    }
    messages=[message]
    response=client.chat.completions.create(model=model, messages=messages, temperature=0.8, max_tokens=500)
    ans=response.choices[0].message.content
    return ans


# Define your user input variable to test across all prompts
test_complaint = "My laptop screen is flickering and it won't charge."

# -------------------------------------------------------------
# 1. THE "BAD" PROMPT
# -------------------------------------------------------------
bad_prompt = f"""
Here is a user complaint: "{test_complaint}". Handle this.
"""

# -------------------------------------------------------------
# 2. THE "OKAY" PROMPT
# -------------------------------------------------------------
okay_prompt = f"""
Role: You are a customer support assistant at a laptop company.
Task: Classify the user complaint below into a category.

Complaint: "{test_complaint}"
"""

# -------------------------------------------------------------
# 3. THE "GOOD" PROMPT
# -------------------------------------------------------------
good_prompt = f"""
Role: You are a customer support assistant at a laptop company.
Task: Classify the user complaint below.
Constraints: You must choose exactly one of these three categories: Billing, Technical, or Return.

Complaint: "{test_complaint}"
"""

# -------------------------------------------------------------
# 4. THE "BETTER" PROMPT
# -------------------------------------------------------------
better_prompt = f"""
Role: You are a customer support assistant at a laptop company.
Task: Classify the user complaint below.
Constraints: You must choose exactly one of these three categories: Billing, Technical, or Return.
Output Format: Your answer must be exactly one word. Only output the name of the category in plain text with no punctuation, explanation, or extra characters.

Complaint: "{test_complaint}"
"""

# -------------------------------------------------------------
# 5. THE "BEST" PROMPT (Production-Grade)
# -------------------------------------------------------------
best_prompt = f"""
Role: You are a customer support assistant at a laptop company.
Task: Classify the user complaint below.
Constraints: You must choose exactly one of these categories: Billing, Technical, Return, or Other.
Output Format: Your answer must be exactly one word. Only output the name of the category in plain text with no punctuation or extra characters.

Examples:
* Input: "I was overcharged on my last receipt." -> Output: Billing
* Input: "I want to send this device back for a refund." -> Output: Return

Fallback: If the user complaint is completely unrelated to a laptop company's customer support or does not fit Billing, Technical, or Return, you must output the word "Other".

Complaint: "{test_complaint}"
"""

# -------------------------------------------------------------
# TESTING AUTOMATION
# -------------------------------------------------------------
# Put them in a dictionary to easily print and compare their outputs
prompt_tests = {
    "1. Bad Prompt": bad_prompt,
    "2. Okay Prompt": okay_prompt,
    "3. Good Prompt": good_prompt,
    "4. Better Prompt": better_prompt,
    "5. Best Prompt": best_prompt
}

# Run the test loop
print(f"--- TESTING WITH INPUT: '{test_complaint}' ---\n")
for prompt_name, prompt_string in prompt_tests.items():
    print(f"=== Running {prompt_name} ===")
    # Calling your custom function to fetch the response
    response = get_response(prompt_string) 
    print(f"Output:\n{response}")
    print("-" * 100)