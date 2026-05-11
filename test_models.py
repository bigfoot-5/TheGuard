import os
from dotenv import load_dotenv
load_dotenv()
from google import genai

client = genai.Client()

for model in ['gemini-1.5-flash', 'gemini-2.0-flash', 'gemini-2.5-flash']:
    try:
        response = client.models.generate_content(
            model=model,
            contents='Say hi',
        )
        print(f"{model} SUCCESS: {response.text}")
    except Exception as e:
        print(f"{model} FAILED: {e}")
