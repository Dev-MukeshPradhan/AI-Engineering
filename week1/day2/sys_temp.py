import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
my_api_key=os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("Api Key is missing")


client=Groq(api_key=my_api_key)

model="llama-3.3-70b-versatile"
role="user"
prompt="suggest me only 2 best names you feel with resopect to indo_chinese cuisine "
message_system={
    "role": "system",
    "content": "You are a brand manager who suggests me a name for food company"
}
# groq asks for a list of messages, each message is a dictionary with role and content keys
message={
    "role": role,
    "content": prompt
}
messages=[message_system, message]
# Temperature by default is 0(safe temperatrure), temperature range must be taken between 0 to 2, higher the temperature more creative the response will be

response=client.chat.completions.create(model=model, messages=messages, temperature=1.5)
#print(response)

print("###################################################################################")

answer=response.choices[0].message.content
print(answer)