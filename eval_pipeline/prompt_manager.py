import hashlib
import json
import os
import difflib
from datetime import datetime

# Define where prompts and metadata will live
PROMPTS_DIR = "eval_pipeline/prompts"
METADATA_FILE = f"{PROMPTS_DIR}/metadata.json"

def compute_prompt_hash(prompt_content: str) -> str:
    """
    Generates a deterministic 8-character version identifier based on prompt content.
    """
    # Any microscopic change (even whitespace) creates a completely new hash
    return hashlib.sha256(prompt_content.encode('utf-8')).hexdigest()[:8]

def save_prompt_version(prompt_name: str, prompt_content: str, author: str, model_config: dict) -> str:
    """
    Hashes the prompt, saves it to a file, and updates the metadata tracking.
    Enforces immutability for deployed prompts.
    """
    os.makedirs(PROMPTS_DIR, exist_ok=True)
    prompt_hash = compute_prompt_hash(prompt_content)
    
    # Initialize metadata if it doesn't exist
    if not os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, "w") as f:
            json.dump({}, f)
            
    with open(METADATA_FILE, "r") as f:
        metadata = json.load(f)
        
    # ENFORCE IMMUTABILITY: If hash exists and is a "GO", do not overwrite it
    if prompt_hash in metadata and metadata[prompt_hash].get("eval_status") == "GO":
        print(f"⚠️ Prompt version {prompt_hash} is already locked and deployed. No changes made.")
        return prompt_hash
        
    # Update Metadata JSON
    metadata[prompt_hash] = {
        "prompt_name": prompt_name,
        "timestamp": datetime.now().isoformat(),
        "author": author,
        "model_config": model_config,
        "eval_status": "PENDING" # This will be flipped to GO/NO-GO by the CI/CD pipeline
    }
    
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=4)
        
    # Save the actual prompt text securely
    prompt_file_path = f"{PROMPTS_DIR}/{prompt_name}_{prompt_hash}.txt"
    with open(prompt_file_path, "w") as f:
        f.write(prompt_content)
        
    print(f"✅ Saved new prompt version: {prompt_hash}")
    return prompt_hash

def generate_prompt_diff(baseline_content: str, candidate_content: str) -> str:
    """
    Generates a programmatic diff between two prompt versions to inject into CI/CD reports.
    """
    diff = difflib.unified_diff(
        baseline_content.splitlines(), 
        candidate_content.splitlines(),
        fromfile="Baseline (Production)",
        tofile="Candidate (New PR)",
        lineterm=''
    )
    return '\n'.join(diff)

# ==========================================
# QUICK TEST RUNNER
# ==========================================
if __name__ == "__main__":
    print("🧪 Testing Git-Native Prompt Versioning...\n")
    
    baseline_prompt = "You are a helpful assistant. Keep answers short."
    candidate_prompt = "You are a helpful assistant. Keep answers extremely short and concise."
    
    # 1. Save Baseline
    print("Saving Baseline...")
    base_hash = save_prompt_version(
        prompt_name="system_assistant",
        prompt_content=baseline_prompt,
        author="karthik_t",
        model_config={"model": "gpt-4o-mini", "temperature": 0.2}
    )
    
    # 2. Save Candidate (Modified prompt)
    print("\nSaving Candidate...")
    cand_hash = save_prompt_version(
        prompt_name="system_assistant",
        prompt_content=candidate_prompt,
        author="karthik_t",
        model_config={"model": "gpt-4o-mini", "temperature": 0.2}
    )
    
    # 3. Generate Diff
    print(f"\n🔍 Generating Diff between {base_hash} and {cand_hash}...")
    print("--------------------------------------------------")
    print(generate_prompt_diff(baseline_prompt, candidate_prompt))
    print("--------------------------------------------------")