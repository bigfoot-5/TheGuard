import math
import os
from dotenv import load_dotenv
from openai import OpenAI
from google import genai

# Load environment variables
load_dotenv()

# Initialize both clients
openai_client = OpenAI()
google_client = genai.Client()

def get_embedding(text: str, embedding_model: str) -> list[float]:
    """
    Fetches the embedding vector, dynamically routing to either Google or OpenAI.
    """
    try:
        # --- ROUTE 1: GOOGLE GEMINI ---
        if "gemini" in embedding_model.lower() or "004" in embedding_model:
            response = google_client.models.embed_content(
                model=embedding_model,
                contents=text
            )
            return response.embeddings[0].values
            
        # --- ROUTE 2: OPENAI (Default) ---
        else:
            response = openai_client.embeddings.create(
                input=text,
                model=embedding_model
            )
            return response.data[0].embedding
            
    except Exception as e:
        print(f"  -> ⚠️ Embedding API Error ({embedding_model}): {e}")
        # Return a generic zero-vector so the pipeline math doesn't crash.
        # It will safely result in a 0.0 cosine similarity.
        return [0.0] * 768 

def calculate_cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """
    Calculates the Cosine Similarity between two vectors.
    Formula: (A dot B) / (||A|| * ||B||)
    """
    # 1. Calculate Dot Product
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    
    # 2. Calculate Magnitudes (Euclidean distance from origin)
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))
    
    # Prevent division by zero just in case
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
        
    # 3. Return Cosine Similarity
    return dot_product / (magnitude1 * magnitude2)

def score_semantic_similarity(expected_text: str, generated_text: str, embedding_model: str = "text-embedding-3-small") -> float:
    """
    Orchestrates the API calls and math to return a 0.0 to 1.0 similarity score.
    """
    if not generated_text or not generated_text.strip():
        return 0.0
        
    # Vectors are dynamically fetched based on the YAML config
    expected_vector = get_embedding(expected_text, embedding_model)
    gen_vector = get_embedding(generated_text, embedding_model)
    
    # Calculate how close they are in vector space
    score = calculate_cosine_similarity(expected_vector, gen_vector)
    
    # Round to 4 decimal places for clean reporting
    return round(score, 4)

# ==========================================
# QUICK TEST RUNNER
# ==========================================
if __name__ == "__main__":
    print("🧪 Testing Semantic Similarity Scoring...\n")
    
    reference = "Get 50% off all Apple MacBooks this Friday. Use code MAC50."
    good_generation = "This Friday, take 50% off Apple MacBooks with promo code MAC50!"
    
    # Test OpenAI
    score_oai = score_semantic_similarity(reference, good_generation, embedding_model="text-embedding-3-small")
    print(f"OpenAI Similarity: {score_oai}")
    
    # Test Gemini
    score_gem = score_semantic_similarity(reference, good_generation, embedding_model="text-embedding-004")
    print(f"Gemini Similarity: {score_gem}")