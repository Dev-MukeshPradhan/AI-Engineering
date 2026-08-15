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
prompt="Tell me about Groq"
prompt1="Hi"
prompt2="Explain time travle in detail"
prompt3="Wite an essay in machine learning in 100 words"
prompts=[prompt1, prompt2, prompt3]
for prompt in prompts:
    message={
    "role": role,
    "content": prompt
    }
    messages=[message]
    response=client.chat.completions.create(model=model, messages=messages, max_tokens=100, temperature=1.5)
    #print(response)

    #print("###################################################################################")
    
    answer=response.choices[0].message.content
    print(answer)
    usage=response.usage
    print(f"Prompt: {prompt} --> Prompt tokens : {usage.prompt_tokens} , Completion tokens : {usage.completion_tokens} , Total tokens : {usage.total_tokens} , Finish Reason : {response.choices[0].finish_reason}")