#!/usr/bin/env python3
"""
Test script to validate 4-hop chain generation
"""
import random
from create_data import generate_entities, generate_reasoning_chains

def test_4hop_chains():
    print("Testing 4-hop chain generation...")
    
    # Generate entities
    entities = generate_entities()[0]  # Use training entities
    
    # Generate a small set of chains with 4-hop included
    chains = generate_reasoning_chains(entities, num_chains=100)
    
    # Count chains by type
    counts = {}
    for chain in chains:
        reasoning_type = chain['reasoning_type']
        if reasoning_type not in counts:
            counts[reasoning_type] = 0
        counts[reasoning_type] += 1
    
    print(f"Generated {len(chains)} chains:")
    for reasoning_type in sorted(counts.keys()):
        print(f"  {reasoning_type}: {counts[reasoning_type]} chains")
    
    # Show examples of each 4-hop type
    print("\n" + "="*60)
    print("4-HOP CHAIN EXAMPLES")
    print("="*60)
    
    hop4_chains = [c for c in chains if c['reasoning_type'] == '4-hop']
    
    for i, chain in enumerate(hop4_chains[:5]):  # Show first 5
        print(f"\nExample {i+1}:")
        print(f"Question: {chain['question']}")
        print(f"Answer: {chain['answer']}")
        print(f"Reasoning chain:")
        for j, fact in enumerate(chain['facts'], 1):
            print(f"  {j}. {fact}")
        
        # Verify the chain makes sense
        if len(chain['facts']) != 4:
            print(f"  ❌ ERROR: Expected 4 facts, got {len(chain['facts'])}")
        else:
            print(f"  ✅ Chain has correct length")
    
    if len(hop4_chains) == 0:
        print("❌ No 4-hop chains generated!")
    else:
        print(f"\n✅ Successfully generated {len(hop4_chains)} 4-hop chains")

if __name__ == "__main__":
    test_4hop_chains() 