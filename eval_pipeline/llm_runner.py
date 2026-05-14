import os
import time
from dotenv import load_dotenv
from openai import OpenAI
import anthropic
from google import genai
import groq

load_dotenv()

openai_client = OpenAI()
claude_client = anthropic.Anthropic()
google_client = genai.Client()
groq_client = groq.Groq()

PRICING = {
    "gpt-4o-mini": (0.15, 0.60),
    "claude-3-5-haiku-20241022": (1.00, 5.00),
    "gemini-1.5-flash": (0.075, 0.30),
    "llama3-8b-8192": (0.05, 0.08) # Groq pricing (often free tier, but good to estimate)
}

def _calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    rates = PRICING.get(model, (0.0, 0.0))
    return (input_tokens / 1_000_000 * rates[0]) + (output_tokens / 1_000_000 * rates[1])

def _execute_api_call(system_prompt: str, user_input: str, model: str, temperature: float) -> tuple[str, float]:
    """Makes the raw API call and returns (response_text, cost)."""
    
    if "claude" in model.lower():
        response = claude_client.messages.create(
            model=model,
            max_tokens=1000,
            system=system_prompt,
            messages=[{"role": "user", "content": user_input}],
            temperature=temperature
        )
        cost = _calculate_cost(model, response.usage.input_tokens, response.usage.output_tokens)
        return response.content[0].text, cost

    elif "gemini" in model.lower():
        response = google_client.models.generate_content(
            model=model,
            contents=system_prompt + "\n\n" + user_input
        )
        input_toks = response.usage_metadata.prompt_token_count if response.usage_metadata else 0
        output_toks = response.usage_metadata.candidates_token_count if response.usage_metadata else 0
        return response.text, _calculate_cost(model, input_toks, output_toks)

    elif "llama" in model.lower():
        response = groq_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            temperature=temperature
        )
        cost = _calculate_cost(model, response.usage.prompt_tokens, response.usage.completion_tokens)
        return response.choices[0].message.content, cost

    else:
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            temperature=temperature
        )
        cost = _calculate_cost(model, response.usage.prompt_tokens, response.usage.completion_tokens)
        return response.choices[0].message.content, cost

def generate_response(system_prompt: str, user_input: str, model: str = "gpt-4o-mini", temperature: float = 0.2) -> tuple[str, float]:
    """
    Sends the prompt to the selected LLM and returns the generated text along with its cost.
    If the primary model fails, it automatically retries with a reliable default model.
    """
    max_retries = 2
    for attempt in range(max_retries):
        try:
            return _execute_api_call(system_prompt, user_input, model, temperature)
        except Exception as e:
            print(f"  -> ⚠️ API Error on {model} (Attempt {attempt+1}): {e}")
            time.sleep(2 ** attempt)
            
    print(f"  -> 🚨 {model} failed completely. Triggering Fallback Router to GPT-4o-mini...")
    try:
        return _execute_api_call(system_prompt, user_input, "gpt-4o-mini", temperature)
    except Exception as e:
        print("  -> 💥 Fallback also failed. Pipeline broken.")
        raise e