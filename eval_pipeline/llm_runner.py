# eval_pipeline/llm_runner.py
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

def generate_response(system_prompt: str, user_input: str, model: str = "gpt-4o-mini", temperature: float = 0.2) -> str:
    """Generates a text response from the target LLM."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            temperature=temperature
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating response: {e}")
        return ""