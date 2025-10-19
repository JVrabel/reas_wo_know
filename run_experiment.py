#!/usr/bin/env python3
"""
Run the complete knowledge-free reasoner experiment.
"""

import subprocess
import sys

def run_step(script_name, description):
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")
    
    try:
        # Remove capture_output=True to see real-time output
        result = subprocess.run([sys.executable, script_name], check=True)
        print(f"\n✅ {script_name} completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ ERROR in {script_name}")
        print(f"Exit code: {e.returncode}")
        sys.exit(1)

if __name__ == "__main__":
    steps = [
        # ("create_data.py", "Creating synthetic knowledge base and QA pairs"),
        # ("build_retriever.py", "Building TF-IDF retriever and FAISS index"),
        # ("test_retrieval.py", "Testing retriever quality"),
        ("train_reasoner.py", "Training the reasoning model"),
        ("eval.py", "Evaluating the complete system")
    ]
    
    for script, desc in steps:
        run_step(script, desc)
    
    print(f"\n{'='*60}")
    print("EXPERIMENT COMPLETE!")
    print(f"{'='*60}")