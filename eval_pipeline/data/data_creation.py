import os
import json
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import List
from tenacity import retry, wait_fixed, stop_after_attempt

# ==========================================
# SETUP
# ==========================================
load_dotenv()
client = genai.Client()
os.makedirs("eval_pipeline/data", exist_ok=True)

def print_retry_status(retry_state):
    print(f"  [!] ⏳ API Rate Limit Hit! Sleeping for 65 seconds before attempt {retry_state.attempt_number}/5...")

# ==========================================
# SCHEMAS
# ==========================================
class DealCopyCase(BaseModel):
    merchant_name: str
    raw_deal_extraction: str
    category: str
    target_channel: str = Field(description="Must be one of: Email, WhatsApp, Push Notification, Glance Lockscreen")
    localization_required: str = Field(description="E.g., 'None', 'Translate urgency to Hindi', 'Translate to Telugu'")
    expected_max_length: int = Field(description="Email: 500, WA: 160, Push: 150, Glance: 55")

class DealCopyDataset(BaseModel):
    cases: List[DealCopyCase]

class InsuranceIntentCase(BaseModel):
    cart_id: str
    cart_contents: List[str]
    user_device_type: str = Field(description="E.g., iOS, Android, Desktop")
    historical_purchase_flags: List[str] = Field(description="E.g., 'Frequent Traveler', 'High Returns'")
    expected_intent: str = Field(description="Must be one of: Electronics Damage Protection, Travel Cancellation, Fraud/Cyber Protection, Health/Accident, No Insurance Applicable")
    is_ambiguous_edge_case: bool = Field(description="True if the cart contains conflicting items (e.g., iPhone and Flight ticket)")

class InsuranceIntentDataset(BaseModel):
    cases: List[InsuranceIntentCase]

class CreditNarrativeCase(BaseModel):
    merchant_id: str
    msme_registration_status: str = Field(description="E.g., Registered, Unregistered, Pending")
    business_vintage_years: float = Field(description="Age of business. Use weird decimals for edge cases, e.g., 1.25, 0.5")
    historical_default_rate_percentage: float 
    yoy_gmv_growth_percentage: float = Field(description="Include conflicting signals, e.g., negative growth but 0 defaults")
    expected_decision: str = Field(description="Must be one of: Approve, Reject, Manual Review")

class CreditNarrativeDataset(BaseModel):
    cases: List[CreditNarrativeCase]

# ==========================================
# CORE API CALLER (The Fix is Here)
# ==========================================
# Notice the retry is ONLY on this small helper function now.
@retry(wait=wait_fixed(65), stop=stop_after_attempt(5), before_sleep=print_retry_status)
def generate_batch(prompt: str, schema: BaseModel) -> List[dict]:
    """Makes a single API call to generate 10 items."""
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
            temperature=0.7,
        ),
    )
    return json.loads(response.text)["cases"]

# ==========================================
# GENERATION FUNCTIONS
# ==========================================
def generate_deal_copy_data():
    print("\n🚀 [1/3] Starting: Deal Copy Cases...")
    all_cases = []
    for i in range(3):
        print(f"  -> Generating batch {i+1}/3 ({len(all_cases)}/30 cases done)...")
        prompt = """
        You are a QA data generator. Generate exactly 10 test cases for evaluating marketing copy generation.
        Create diverse deal copy scenarios. Ensure a mix of all target channels (Email, WhatsApp, Push, Glance). 
        Include edge cases requiring cultural localization (Hindi/Telugu) and difficult categories.
        """
        cases = generate_batch(prompt, DealCopyDataset)
        all_cases.extend(cases)
        
        # Mandatory sleep between successful batches to respect the 1-minute Free Tier limits
        if i < 2:
            print("  -> Sleeping for 65 seconds to clear API token limits...")
            time.sleep(65)
            
    with open("eval_pipeline/data/deal_copy_cases.json", "w") as f:
        json.dump(all_cases, f, indent=4)
    print("✅ Successfully saved 30 Deal Copy Cases.")

def generate_insurance_data():
    print("\n🚀 [2/3] Starting: Insurance Intent Cases...")
    all_cases = []
    for i in range(3):
        print(f"  -> Generating batch {i+1}/3 ({len(all_cases)}/30 cases done)...")
        prompt = """
        You are a QA data generator. Generate exactly 10 test cases for e-commerce cart insurance classification.
        Create 10 cart intent scenarios distributed across the 5 specific intents. 
        Include 'ambiguous edge cases' where cart items conflict.
        """
        cases = generate_batch(prompt, InsuranceIntentDataset)
        all_cases.extend(cases)
        
        if i < 2:
            print("  -> Sleeping for 65 seconds to clear API token limits...")
            time.sleep(65)
            
    with open("eval_pipeline/data/insurance_intent_cases.json", "w") as f:
        json.dump(all_cases, f, indent=4)
    print("✅ Successfully saved 30 Insurance Intent Cases.")

def generate_credit_data():
    print("\n🚀 [3/3] Starting: Credit Narrative Cases...")
    all_cases = []
    for i in range(3):
        print(f"  -> Generating batch {i+1}/3 ({len(all_cases)}/30 cases done)...")
        prompt = """
        You are a QA data generator. Generate exactly 10 test cases for MSME credit underwriting evaluation.
        Create 10 merchant profiles ranging from prime borrowers to high-risk rejections. 
        Include edge cases like weird decimal growths.
        """
        cases = generate_batch(prompt, CreditNarrativeDataset)
        all_cases.extend(cases)
        
        if i < 2:
            print("  -> Sleeping for 65 seconds to clear API token limits...")
            time.sleep(65)
            
    with open("eval_pipeline/data/credit_narrative_cases.json", "w") as f:
        json.dump(all_cases, f, indent=4)
    print("✅ Successfully saved 30 Credit Narrative Cases.")

if __name__ == "__main__":
    generate_deal_copy_data()
    
    # Rest between massive datasets
    print("\n[!] Pausing 65 seconds before starting the next massive dataset...")
    time.sleep(65)
    
    generate_insurance_data()
    
    print("\n[!] Pausing 65 seconds before starting the next massive dataset...")
    time.sleep(65)
    
    generate_credit_data()
    
    print("\n🎉 ALL DATASETS GENERATED SUCCESSFULLY!")