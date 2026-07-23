# ============================================================
# IMPORTS & SETUP
# ============================================================
import os
from pathlib import Path
from time import sleep
from dotenv import load_dotenv
from groq import Groq
import re

load_dotenv()
my_api_key=os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("API key kaha hai bhai")

# ============================================================
# CLIENT & MODEL CONFIG
# ============================================================
client=Groq(api_key=my_api_key)
model="llama-3.3-70b-versatile"

# ============================================================
# TOOL/ACTION FUNCTIONS (search, api, calculator, etc.)
# ============================================================

def get_product_price(product):
    if product == 'iPhone 17':
        return 1000
    elif product == "iPhone 15":
        return 500
    else:
        return 0
    
def calculator(expression):
    try:
        return eval(expression)
    except:
        return "calc error!"

# ============================================================
# TOOL REGISTRY (maps tool name -> actual function)
# ============================================================
tools = {
    "get_product_price": get_product_price,
    "calculator": calculator
}

# ============================================================
# SYSTEM PROMPT (defines ReAct format + rules for the LLM)
# ============================================================
system_prompt = """
You are a shopping assistant.

You have these tools:

get_product_price(product)
calculator(expression)
IMPORTANT:
Call tools exactly like these examples:

Action: get_product_price("iPhone 17")
Action: calculator("5000 - 1000")

Never write:
get_product_price(product="iPhone 17")

Never write:
calculator(expression="5000 - 1000")
Follow these rules:

1. Decide what you need to do next.
2. Call ONLY ONE tool at a time.
3. After writing an Action, STOP immediately.
4. Never guess or invent a tool result.
5. Wait until you receive an Observation.
6. Then decide your next action.
7. When the task is complete, give the Final Answer.

Format:

Thought: what you need to do
Action: tool_name(argument)

When finished:

Final Answer: your answer
"""

# ============================================================
# AGENT LOOP (Thought -> Action -> Observation cycle)
# ============================================================
def run_agent(question):

    # ---- initialize conversation memory with system + user msg ----
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": question
        }
    ]

    # ---- loop for max 5 steps (safety limit to avoid infinite loop) ----
    for step in range(5):

        print("\n------------------")
        print("STEP", step + 1)
        print("------------------")

        # ---- call LLM to get next Thought/Action ----
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0
        )

        answer = response.choices[0].message.content

        print(answer)

        # ---- check if agent has finished ----
        # Agent has finished
        if "Final Answer:" in answer:
            break


        # ---- parse the Action line using regex ----
        # Find the Action
        match = re.search(
            r"Action:\s*(\w+)\((.*?)\)",
            answer
        )


        if match:

            tool_name = match.group(1)

            tool_input = match.group(2)

            tool_input = tool_input.strip()

            tool_input = tool_input.strip('"')


            # ---- execute the matched tool ----
            # Run the tool
            if tool_name in tools:

                tool = tools[tool_name]

                observation = tool(tool_input)

            else:

                observation = "Tool not found"


            print(
                "Observation:",
                observation
            )


            # ---- update memory: save LLM's Thought/Action ----
            # Add LLM response to memory
            messages.append({
                "role": "assistant",
                "content": answer
            })


            # ---- update memory: feed Observation back to LLM ----
            # Give tool result back to LLM
            messages.append({
                "role": "user",
                "content":
                    "Observation: "
                    + str(observation)
            })
            sleep(5)


# ============================================================
# MAIN EXECUTION
# ============================================================
prompt="""
I have 5000 rupees. What is the price of an iphone 17?
and how much money will I have left?
"""
run_agent(prompt)