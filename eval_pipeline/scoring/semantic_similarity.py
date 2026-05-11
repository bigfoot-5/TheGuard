import math
import os
from dotenv import load_dotenv
from google import genai

# Load environment variables
load_dotenv()
client = genai.Client()

def get_embedding(text: str) -> list[float]:
    """
    Calls the Gemini API to convert text into a high-dimensional vector.
    """
    response = client.models.embed_content(
        model='gemini-embedding-2-preview',
        contents=text
    )
    # Extract the raw float array from the response
    return response.embeddings[0].values

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

def score_semantic_similarity(reference_text: str, generated_text: str) -> float:
    """
    Orchestrates the API calls and math to return a 0.0 to 1.0 similarity score.
    """
    # Get vector representations of both texts
    ref_vector = get_embedding(reference_text)
    gen_vector = get_embedding(generated_text)
    
    # Calculate how close they are in vector space
    score = calculate_cosine_similarity(ref_vector, gen_vector)
    
    # Round to 4 decimal places for clean reporting
    return round(score, 4)

# ==========================================
# QUICK TEST RUNNER
# ==========================================
if __name__ == "__main__":
    print("🧪 Testing Semantic Similarity Scoring...\n")
    
    reference = "Get 50% off all Apple MacBooks this Friday. Use code MAC50."
    
    # Test 1: High Similarity (Paraphrased but same meaning)
    good_generation = "This Friday, take 50% off Apple MacBooks with promo code MAC50!"
    score_1 = score_semantic_similarity(reference, good_generation)
    print(f"High Similarity Test: {score_1} (Expected > 0.90)")
    
    # Test 2: Moderate Similarity (Changed a key detail)
    bad_generation = "Get 20% off all Windows Laptops this Friday. Use code WIN20."
    score_2 = score_semantic_similarity(reference, bad_generation)
    print(f"Moderate Similarity Test: {score_2} (Expected ~0.60 - 0.75)")
    
    # Test 3: Low Similarity (Completely off topic)
    terrible_generation = "GrabOn is a great place to work. We have an open office."
    score_3 = score_semantic_similarity(reference, terrible_generation)
    print(f"Low Similarity Test: {score_3} (Expected < 0.40)")