import hashlib
import json
import os
import difflib
from datetime import datetime

PROMPTS_DIR = "eval_pipeline/prompts"
METADATA_FILE = f"{PROMPTS_DIR}/metadata.json"

def compute_prompt_hash(prompt_content: str) -> str:
    """
    Creates a unique ID based on the text of the prompt.
    """
    return hashlib.sha256(prompt_content.encode('utf-8')).hexdigest()[:8]

def save_prompt_version(prompt_name: str, prompt_content: str, author: str, model_config: dict) -> str:
    """
    Saves the prompt to a file and records it. Prevents overwriting already approved prompts.
    """
    os.makedirs(PROMPTS_DIR, exist_ok=True)
    prompt_hash = compute_prompt_hash(prompt_content)
    
    if not os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, "w") as f:
            json.dump({}, f)
            
    with open(METADATA_FILE, "r") as f:
        metadata = json.load(f)
        
    if prompt_hash in metadata and metadata[prompt_hash].get("eval_status") == "GO":
        print(f"⚠️ Prompt version {prompt_hash} is already locked and deployed. No changes made.")
        return prompt_hash
        
    metadata[prompt_hash] = {
        "prompt_name": prompt_name,
        "timestamp": datetime.now().isoformat(),
        "author": author,
        "model_config": model_config,
        "eval_status": "PENDING"
    }
    
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=4)
        
    prompt_file_path = f"{PROMPTS_DIR}/{prompt_name}_{prompt_hash}.txt"
    with open(prompt_file_path, "w") as f:
        f.write(prompt_content)
        
    print(f"✅ Saved new prompt version: {prompt_hash}")
    return prompt_hash

def generate_prompt_diff(baseline_content: str, candidate_content: str) -> str:
    """
    Creates a readable comparison between two versions of a prompt.
    """
    diff = difflib.unified_diff(
        baseline_content.splitlines(), 
        candidate_content.splitlines(),
        fromfile="Baseline (Production)",
        tofile="Candidate (New PR)",
        lineterm=''
    )
    return '\n'.join(diff)
