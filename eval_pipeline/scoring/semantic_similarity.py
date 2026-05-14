import math
import os
from dotenv import load_dotenv
from openai import OpenAI
from google import genai

load_dotenv()

openai_client = OpenAI()
google_client = genai.Client()

def get_embedding(text: str, embedding_model: str) -> list[float]:
    """
    Gets the text embedding from either Google or OpenAI.
    """
    try:
        if "gemini" in embedding_model.lower() or "004" in embedding_model:
            response = google_client.models.embed_content(
                model=embedding_model,
                contents=text
            )
            return response.embeddings[0].values
            
        else:
            response = openai_client.embeddings.create(
                input=text,
                model=embedding_model
            )
            return response.data[0].embedding
            
    except Exception:
        return [0.0] * 768 

def calculate_cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """
    Compares two text embeddings to see how similar they are.
    """
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))
    
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
        
    return dot_product / (magnitude1 * magnitude2)

def score_semantic_similarity(expected_text: str, generated_text: str, embedding_model: str = "text-embedding-3-small") -> float:
    """
    Compares the generated text to the expected text and returns a similarity score from 0.0 to 1.0.
    """
    if not generated_text or not generated_text.strip():
        return 0.0
        
    expected_vector = get_embedding(expected_text, embedding_model)
    gen_vector = get_embedding(generated_text, embedding_model)
    
    score = calculate_cosine_similarity(expected_vector, gen_vector)
    
    return round(score, 4)
