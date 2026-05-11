import os
from dotenv import load_dotenv
load_dotenv()
from google import genai
from google.genai import types
from pydantic import BaseModel

class MySchema(BaseModel):
    name: str

client = genai.Client()
response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents='Generate a name',
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=MySchema,
    ),
)
print(response.text)
