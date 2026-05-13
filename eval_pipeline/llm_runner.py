import os
import time
from dotenv import load_dotenv
from openai import OpenAI
from google import genai
import anthropic

load_dotenv()

oai_client = OpenAI()
gemini_client = genai.Client()
claude_client = anthropic.Anthropic()

def generate_response(system_prompt: str, user_input: str, model: str = "gpt-4o-mini", temperature: float = 0.2, max_retries: int = 3) -> str:
    """Generates a text response with automatic retries for API Rate Limits."""
    
    for attempt in range(max_retries):
        try:
            # 1. GOOGLE GEMINI ROUTER
            if "gemini" in model.lower():
                full_prompt = f"System Instructions:\n{system_prompt}\n\nUser Input:\n{user_input}"
                response = gemini_client.models.generate_content(
                    model=model,
                    contents=full_prompt,
                    config={"temperature": temperature}
                )
                return response.text

            # 2. ANTHROPIC CLAUDE ROUTER
            elif "claude" in model.lower():
                response = claude_client.messages.create(
                    model=model,
                    system=system_prompt,
                    max_tokens=1000,
                    temperature=temperature,
                    messages=[{"role": "user", "content": user_input}]
                )
                return response.content[0].text

            # 3. OPENAI ROUTER
            else:
                response = oai_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_input}
                    ],
                    temperature=temperature
                )
                return response.choices[0].message.content

        except Exception as e:
            error_msg = str(e)
            
            # Check if the error is a Rate Limit (429)
            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg or "rate_limit" in error_msg.lower():
                wait_time = (attempt + 1) * 20  # Waits 20s, then 40s, then 60s
                print(f"  -> ⏳ Rate limit hit on {model}. Waiting {wait_time}s to retry (Attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait_time)
                continue # Try the loop again
            else:
                # If it's a different error (like a bad API key), fail immediately
                print(f"🚨 Fatal API Error routing to {model}: {e}")
                return ""
                
    print(f"❌ Failed to get response from {model} after {max_retries} attempts.")
    return ""