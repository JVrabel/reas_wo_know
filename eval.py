import json
import random
import torch
import faiss
import pickle
import numpy as np
import re
import os
from datetime import datetime
from utils import build_alias, apply_alias, build_prompt
from model import TinyDec, tok, device
from config import CONFIG
import glob

print("[eval] Loading data and models...")

# Create results directory with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = f"results/eval_{timestamp}"
os.makedirs(results_dir, exist_ok=True)

# Load data
with open("corpus/docs.jsonl", "r") as f:
    docs = [json.loads(line.strip()) for line in f]

with open("corpus/qa.jsonl", "r") as f:
    QA = [json.loads(line.strip()) for line in f]

# Load retriever
with open("corpus/tfidf_vec.pkl", "rb") as f:
    vectorizer = pickle.load(f)
index = faiss.read_index("corpus/tfidf.index")

# Find latest checkpoint
checkpoint_files = glob.glob("corpus/checkpoint_step_*.pt")
if checkpoint_files:
    # Sort by step number and get latest
    latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    print(f"[eval] Loading latest checkpoint: {latest_checkpoint}")
    checkpoint = torch.load(latest_checkpoint, map_location=device)
else:
    # Fallback to best model
    checkpoint = torch.load("corpus/tiny_reasoner.pt", map_location=device)

# Load model
model = TinyDec().to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"[eval] Loaded {len(docs)} documents, {len(QA)} QA pairs")
print(f"[eval] Results will be saved to: {results_dir}")

# Move the function definitions before they're used

def analyze_reasoning_types(qa_list):
    """Analyze the distribution of reasoning chain lengths in the dataset."""
    chain_lengths = {}
    examples_by_type = {}
    
    for q, a, reasoning_chain in qa_list:
        chain_len = len(reasoning_chain)
        
        if chain_len not in chain_lengths:
            chain_lengths[chain_len] = 0
            examples_by_type[chain_len] = []
        
        chain_lengths[chain_len] += 1
        
        # Store first few examples of each type
        if len(examples_by_type[chain_len]) < 3:
            examples_by_type[chain_len].append({
                'question': q,
                'answer': a,
                'chain': reasoning_chain
            })
    
    print(f"\n=== REASONING CHAIN ANALYSIS ===")
    print(f"Total questions: {len(qa_list)}")
    
    for chain_len in sorted(chain_lengths.keys()):
        count = chain_lengths[chain_len]
        percentage = (count / len(qa_list)) * 100
        print(f"{chain_len}-hop questions: {count} ({percentage:.1f}%)")
        
        # Show examples
        for i, example in enumerate(examples_by_type[chain_len]):
            print(f"  Example {i+1}:")
            print(f"    Q: {example['question']}")
            print(f"    A: {example['answer']}")
            print(f"    Chain: {example['chain']}")
    
    return chain_lengths, examples_by_type

def detailed_accuracy_by_hops(qa_list, use_gold_reasoning=True):
    """Evaluate accuracy broken down by reasoning chain length."""
    results_by_hops = {}
    all_results = []
    
    for i, item in enumerate(qa_list):
        if i % 100 == 0:
            print(f"  Evaluating {i}/{len(qa_list)}...")
        
        q, a, reasoning_chain = item[0], item[1], item[2]
        chain_len = len(reasoning_chain)
        
        if chain_len not in results_by_hops:
            results_by_hops[chain_len] = {'correct': 0, 'total': 0, 'examples': []}
        
        if use_gold_reasoning:
            pred, generated_text, prompt, alias, inv = answer_with_reasoning_chain(q, reasoning_chain)
        else:
            pred, generated_text, prompt, alias, inv, retrieved_docs = answer_with_retrieval(q)
            
        is_correct = pred == a
        
        # Store result
        result = {
            "question": q,
            "expected": a,
            "predicted": pred,
            "generated_text": generated_text,
            "correct": is_correct,
            "reasoning_type": f"{chain_len}-hop",
            "reasoning_chain": reasoning_chain,
            "prompt": prompt
        }
        
        all_results.append(result)
        results_by_hops[chain_len]['total'] += 1
        if is_correct:
            results_by_hops[chain_len]['correct'] += 1
        
        # Store first few examples of each type for debugging
        if len(results_by_hops[chain_len]['examples']) < 5:
            results_by_hops[chain_len]['examples'].append(result)
    
    # Print detailed breakdown
    print(f"\n=== ACCURACY BY REASONING HOPS ===")
    overall_correct = sum(r['correct'] for r in all_results)
    overall_total = len(all_results)
    print(f"Overall: {overall_correct}/{overall_total} = {overall_correct/overall_total:.3f}")
    
    for chain_len in sorted(results_by_hops.keys()):
        stats = results_by_hops[chain_len]
        accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        print(f"{chain_len}-hop: {stats['correct']}/{stats['total']} = {accuracy:.3f}")
        
        # Show failure examples for multi-hop
        if chain_len > 1 and accuracy < 1.0:
            print(f"  Sample {chain_len}-hop failures:")
            failure_count = 0
            for example in stats['examples']:
                if not example['correct'] and failure_count < 2:
                    print(f"    Q: {example['question']}")
                    print(f"    Expected: {example['expected']}, Got: {example['predicted']}")
                    print(f"    Generated: '{example['generated_text']}'")
                    print(f"    Chain: {example['reasoning_chain']}")
                    failure_count += 1
    
    return all_results, results_by_hops

# Add this after loading the data, before the main evaluation
print("[eval] Analyzing reasoning types in dataset...")
chain_lengths, examples_by_type = analyze_reasoning_types(QA)

def retrieve(q, k=CONFIG["RETRIEVAL_K"]):
    """Retrieve top-k documents for a query."""
    qv = vectorizer.transform([q]).astype("float32").toarray()
    _, I = index.search(qv, k)
    return [docs[i] for i in I[0]]

def extract_entity_id(text):
    """Extract entity ID from generated text - simplified for raw IDs."""
    # Look for patterns like L0123, P0456, C0789
    import re
    match = re.search(r'[PLCD]\d+', text)
    if match:
        return match.group(0)
    return text.strip()

def answer_with_reasoning_chain(question, reasoning_chain):
    """Answer using gold reasoning chain - WITH ALIASING to match training."""
    # Apply aliasing like in training
    all_texts = [question] + reasoning_chain
    alias, inv_alias = build_alias(all_texts)
    
    # Build prompt with aliasing
    aliased_question = apply_alias(question, alias)
    aliased_chain = [apply_alias(sent, alias) for sent in reasoning_chain]
    
    prompt = f"<Q> {aliased_question}\n"
    for i, sent in enumerate(aliased_chain):
        prompt += f"<DOC_{i+1}> {sent}\n"
    prompt += "<REASON>"
    
    print(f"  [DEBUG] Prompt: {prompt[:200]}...")
    
    # Generate
    input_ids = tok(prompt, return_tensors="pt", truncation=True, max_length=512).input_ids.to(device)
    
    with torch.no_grad():
        output = model.generate(
            input_ids, 
            max_new_tokens=20,
            temperature=0.1,
            do_sample=False
        )
    
    generated_text = tok.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
    
    # Apply inverse aliasing to get back real entity IDs
    for pseudo, real in inv_alias.items():
        generated_text = generated_text.replace(pseudo, real)
    
    # Extract entity ID
    pred = generated_text.strip().split()[0] if generated_text.strip() else ""
    
    print(f"  [DEBUG] Generated: '{generated_text}' → Extracted: '{pred}'")
    
    return pred, generated_text, prompt, alias, inv_alias

def load_retriever():
    """Load the improved entity-based retriever."""
    try:
        with open("corpus/entity_retriever.pkl", "rb") as f:
            retriever_data = pickle.load(f)
        print("[eval] Using entity-based retriever")
        return retriever_data['entity_to_docs'], retriever_data['docs'], "entity"
    except FileNotFoundError:
        print("[eval] Entity retriever not found, falling back to TF-IDF...")
        # Fallback to old retriever
        with open("corpus/tfidf_vec.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        index = faiss.read_index("corpus/tfidf.index")
        
        with open("corpus/docs.jsonl", "r") as f:
            docs = [json.loads(line.strip()) for line in f]
        
        return vectorizer, index, "tfidf"

def retrieve_documents(question, retriever_data, retriever_type, k=4):
    """Retrieve documents using the appropriate method."""
    if retriever_type == "entity":
        entity_to_docs, docs = retriever_data
        return retrieve_by_entities(question, entity_to_docs, docs, k)
    else:
        vectorizer, index = retriever_data
        return retrieve_tfidf(question, vectorizer, index, docs, k)

def retrieve_by_entities(question, entity_to_docs, docs, k=4):
    """Retrieve documents using entity-based method."""
    import re
    
    question_entities = set(re.findall(r'[PLSCD]\d+', question))
    
    if not question_entities:
        return []
    
    # Find candidate documents
    candidate_docs = set()
    for entity in question_entities:
        if entity in entity_to_docs:
            candidate_docs.update(entity_to_docs[entity])
    
    # Score and rank documents
    doc_scores = []
    for doc_id in candidate_docs:
        doc = docs[doc_id]
        doc_entities = set(re.findall(r'[PLSCD]\d+', doc))
        
        # Score by entity overlap
        overlap = len(question_entities.intersection(doc_entities))
        
        # Bonus for relevant relationships
        relationship_bonus = 0
        if 'where_does' in question or 'where_is' in question:
            if any(rel in doc for rel in ['lives_in', 'resides_in', 'headquartered_in', 'located_in', 'is_based_in']):
                relationship_bonus = 0.5
        
        score = overlap + relationship_bonus
        doc_scores.append((score, doc_id))
    
    # Return top-k documents
    doc_scores.sort(reverse=True)
    return [docs[doc_id] for _, doc_id in doc_scores[:k]]

def retrieve_tfidf(question, vectorizer, index, docs, k=4):
    """Retrieve documents using TF-IDF method."""
    qv = vectorizer.transform([question]).astype("float32").toarray()
    _, I = index.search(qv, k)
    return [docs[i] for i in I[0]]

def answer_with_retrieval(question):
    """Answer using document retrieval."""
    # Retrieve relevant documents
    chunks = retrieve_documents(question, retriever_data, retriever_type, k=4)
    
    if not chunks:
        return "NO_DOCS", "", "", {}, {}, []
    
    # Apply aliasing
    all_texts = [question] + chunks
    alias, inv_alias = build_alias(all_texts)
    
    # Build prompt
    prompt = build_prompt(question, chunks, alias)
    
    # Generate answer
    input_ids = tok(prompt, return_tensors="pt", truncation=True, max_length=512).input_ids.to(device)
    
    with torch.no_grad():
        output = model.generate(
            input_ids, 
            max_new_tokens=20,
            temperature=0.1,
            do_sample=False
        )
    
    generated_text = tok.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
    
    # Apply inverse aliasing
    for pseudo, real in inv_alias.items():
        generated_text = generated_text.replace(pseudo, real)
    
    pred = generated_text.strip().split()[0] if generated_text.strip() else ""
    
    return pred, generated_text, prompt, alias, inv_alias, chunks

def detailed_accuracy(qa_list, use_gold_reasoning=True):
    """Evaluate accuracy on a QA list."""
    correct = 0
    hop1_correct = 0
    hop2_correct = 0
    hop1_total = 0
    hop2_total = 0
    
    # Add debugging for first few examples
    debug_examples = []
    
    for i, item in enumerate(qa_list):
        if i % 100 == 0:
            print(f"  Evaluating {i}/{len(qa_list)}...")
        
        q, a, reasoning_chain = item[0], item[1], item[2]
        
        if use_gold_reasoning:
            pred, generated_text, prompt, alias, inv = answer_with_reasoning_chain(q, reasoning_chain)
        else:
            pred, generated_text, prompt, alias, inv, retrieved_docs = answer_with_retrieval(q)
            
        is_correct = pred == a
        
        # Debug first 5 examples
        if i < 5:
            debug_examples.append({
                "question": q,
                "expected": a,
                "predicted": pred,
                "generated_text": generated_text,
                "correct": is_correct,
                "prompt": prompt,
                "alias_mapping": alias
            })
            print(f"  [DEBUG {i}] Q: {q}")
            print(f"  [DEBUG {i}] Expected: {a}, Got: {pred}")
            print(f"  [DEBUG {i}] Generated: '{generated_text}'")
            print(f"  [DEBUG {i}] Correct: {is_correct}")
        
        if is_correct:
            correct += 1
        
        # Classify by reasoning type
        is_2hop = any(x in q for x in ["owner_of", "manager_of", "colleague_of"])
        if is_2hop:
            hop2_total += 1
            if is_correct:
                hop2_correct += 1
        else:
            hop1_total += 1
            if is_correct:
                hop1_correct += 1
    
    # Print debug info
    print(f"\n[DEBUG] First 5 examples:")
    for i, ex in enumerate(debug_examples):
        print(f"  {i}: {ex['question']} -> Expected: {ex['expected']}, Got: {ex['predicted']}")
    
    overall_acc = correct / len(qa_list)
    hop1_acc = hop1_correct / hop1_total if hop1_total > 0 else 0
    hop2_acc = hop2_correct / hop2_total if hop2_total > 0 else 0
    
    results = {
        "overall_accuracy": overall_acc,
        "correct": correct,
        "total": len(qa_list),
        "hop1_accuracy": hop1_acc,
        "hop1_correct": hop1_correct,
        "hop1_total": hop1_total,
        "hop2_accuracy": hop2_acc,
        "hop2_correct": hop2_correct,
        "hop2_total": hop2_total,
        "use_gold_reasoning": use_gold_reasoning
    }
    
    print(f"  Overall accuracy: {overall_acc:.3f} ({correct}/{len(qa_list)})")
    if hop1_total > 0:
        print(f"  1-hop accuracy: {hop1_acc:.3f} ({hop1_correct}/{hop1_total})")
    if hop2_total > 0:
        print(f"  2-hop accuracy: {hop2_acc:.3f} ({hop2_correct}/{hop2_total})")
    
    return results

def evidence_recall(N=100):
    """Check if retrieved documents contain the necessary evidence."""
    recall_scores = []
    examples = []
    
    for i, item in enumerate(random.sample(QA, N)):
        q, a, evidence = item[0], item[1], item[2]
        retrieved = retrieve(q)
        retrieved_text = " ".join(retrieved)
        
        # Check if all evidence sentences are found in retrieved docs
        found_evidence = []
        for ev in evidence:
            if ev in retrieved_text:
                found_evidence.append(ev)
        
        recall = len(found_evidence) / len(evidence) if evidence else 0
        recall_scores.append(recall)
        
        # Save examples for analysis
        if i < 10 or recall < 0.5:
            examples.append({
                "question": q,
                "answer": a,
                "evidence_needed": evidence,
                "evidence_found": found_evidence,
                "evidence_missing": [ev for ev in evidence if ev not in found_evidence],
                "retrieved_docs": retrieved,
                "recall": recall
            })
    
    avg_recall = sum(recall_scores) / len(recall_scores)
    print(f"  Evidence recall: {avg_recall:.3f}")
    
    return avg_recall, examples

print("\n=== EVALUATION RESULTS ===")

# Initialize report
report = {
    "timestamp": timestamp,
    "config": CONFIG,
    "model_info": {
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "device": str(device)
    },
    "data_info": {
        "total_docs": len(docs),
        "total_qa_pairs": len(QA)
    }
}

print("\n1. Accuracy with GOLD reasoning chains (easier):")
gold_results, gold_by_hops = detailed_accuracy_by_hops(QA[:200], use_gold_reasoning=True)

print("\n2. Accuracy with RETRIEVED documents (harder):")
retrieved_results, retrieved_by_hops = detailed_accuracy_by_hops(QA[:100], use_gold_reasoning=False)

print("\n3. Evidence recall:")
recall_score, recall_examples = evidence_recall(100)
report["evidence_recall"] = {
    "average_recall": recall_score,
    "total_examples": 100
}

print("\n4. NOVEL ENTITY TEST (Key test for knowledge-free reasoning):")
try:
    with open("corpus/qa_test.jsonl", "r") as f:
        test_qa = [json.loads(line.strip()) for line in f]
    
    print(f"  Testing on {len(test_qa)} examples with completely novel entities...")
    
    # Test with gold reasoning on novel entities
    novel_results = detailed_accuracy(test_qa[:200], use_gold_reasoning=True)
    report["novel_entity_reasoning"] = novel_results
    
    print(f"  Novel entity accuracy: {novel_results['overall_accuracy']:.3f}")
    
    # Save novel entity examples
    with open(f"{results_dir}/novel_entity_examples.json", "w") as f:
        json.dump(novel_results, f, indent=2)
        
except FileNotFoundError:
    print("  ❌ No test data found - run create_data.py first")
    novel_results = {"overall_accuracy": 0.0, "note": "No test data"}
    report["novel_entity_reasoning"] = novel_results

# Save detailed results
print(f"\n[eval] Saving detailed results to {results_dir}...")

# Save main report
with open(f"{results_dir}/evaluation_report.json", "w") as f:
    json.dump(report, f, indent=2)

# Save detailed examples
with open(f"{results_dir}/gold_reasoning_examples.json", "w") as f:
    json.dump(gold_results, f, indent=2)

with open(f"{results_dir}/retrieved_reasoning_examples.json", "w") as f:
    json.dump(retrieved_results, f, indent=2)

with open(f"{results_dir}/evidence_recall_examples.json", "w") as f:
    json.dump(recall_examples, f, indent=2)

# Fix the summary generation - extract stats from the results
def extract_stats(results_list):
    """Extract statistics from results list."""
    if not results_list:
        return {'overall_accuracy': 0.0, 'correct': 0, 'total': 0, 'hop1_accuracy': 0.0, 'hop1_correct': 0, 'hop1_total': 0, 'hop2_accuracy': 0.0, 'hop2_correct': 0, 'hop2_total': 0}
    
    total = len(results_list)
    correct = sum(1 for r in results_list if r['correct'])
    
    hop1_results = [r for r in results_list if r['reasoning_type'] == '1-hop']
    hop2_results = [r for r in results_list if r['reasoning_type'] == '2-hop']
    
    hop1_correct = sum(1 for r in hop1_results if r['correct'])
    hop2_correct = sum(1 for r in hop2_results if r['correct'])
    
    return {
        'overall_accuracy': correct / total if total > 0 else 0.0,
        'correct': correct,
        'total': total,
        'hop1_accuracy': hop1_correct / len(hop1_results) if hop1_results else 0.0,
        'hop1_correct': hop1_correct,
        'hop1_total': len(hop1_results),
        'hop2_accuracy': hop2_correct / len(hop2_results) if hop2_results else 0.0,
        'hop2_correct': hop2_correct,
        'hop2_total': len(hop2_results)
    }

# Extract stats
gold_stats = extract_stats(gold_results)
retrieved_stats = extract_stats(retrieved_results)

# Update the summary text generation
summary_text = f"""KNOWLEDGE-FREE REASONER EVALUATION REPORT
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Results Directory: {results_dir}

=== MODEL INFO ===
Total Parameters: {report['model_info']['total_parameters']:,}
Device: {report['model_info']['device']}
Training Steps: {CONFIG['TRAIN_STEPS']}
Model Dimension: {CONFIG['MODEL_DIM']}

=== DATA INFO ===
Total Documents: {report['data_info']['total_docs']:,}
Total QA Pairs: {report['data_info']['total_qa_pairs']:,}

=== RESULTS SUMMARY ===
Gold Reasoning Accuracy:     {gold_stats['overall_accuracy']:.3f} ({gold_stats['correct']}/{gold_stats['total']})
  - 1-hop questions:         {gold_stats['hop1_accuracy']:.3f} ({gold_stats['hop1_correct']}/{gold_stats['hop1_total']})
  - 2-hop questions:         {gold_stats['hop2_accuracy']:.3f} ({gold_stats['hop2_correct']}/{gold_stats['hop2_total']})

Retrieved Reasoning Accuracy: {retrieved_stats['overall_accuracy']:.3f} ({retrieved_stats['correct']}/{retrieved_stats['total']})
  - 1-hop questions:         {retrieved_stats['hop1_accuracy']:.3f} ({retrieved_stats['hop1_correct']}/{retrieved_stats['hop1_total']})
  - 2-hop questions:         {retrieved_stats['hop2_accuracy']:.3f} ({retrieved_stats['hop2_correct']}/{retrieved_stats['hop2_total']})

Evidence Recall:             {recall_score:.3f}

=== INTERPRETATION ===
"""

if gold_stats['overall_accuracy'] > 0.8:
    summary_text += "✅ SUCCESS: Model learned to reason from given facts!\n"
elif gold_stats['overall_accuracy'] > 0.5:
    summary_text += "⚠️  PARTIAL: Some reasoning ability, but needs improvement\n"
else:
    summary_text += "❌ FAILURE: Model didn't learn reasoning patterns\n"

if retrieved_stats['overall_accuracy'] > 0.6:
    summary_text += "✅ GOOD: Strong performance with retrieved documents\n"
elif retrieved_stats['overall_accuracy'] > 0.3:
    summary_text += "⚠️  MODERATE: Decent performance with retrieved documents\n"
else:
    summary_text += "❌ POOR: Struggles with retrieved documents\n"

if recall_score > 0.8:
    summary_text += "✅ EXCELLENT: Retrieval finds most needed evidence\n"
elif recall_score > 0.6:
    summary_text += "⚠️  GOOD: Retrieval finds most evidence\n"
else:
    summary_text += "❌ POOR: Retrieval misses important evidence\n"

summary_text += f"""
=== FILES GENERATED ===
- evaluation_report.json: Complete numerical results
- gold_reasoning_examples.json: Detailed examples with gold reasoning chains
- retrieved_reasoning_examples.json: Detailed examples with retrieved documents
- evidence_recall_examples.json: Examples showing retrieval quality
- summary.txt: This human-readable summary

=== SAMPLE ERRORS (Gold Reasoning) ===
"""

# Add sample errors for debugging
error_count = 0
for example in gold_results:
    if not example['correct'] and error_count < 5:
        summary_text += f"""
Question: {example['question']}
Expected: {example['expected']}
Predicted: {example['predicted']}
Generated: {example['generated_text']}
Type: {example['reasoning_type']}
---"""
        error_count += 1

summary_text += f"""

=== SAMPLE ERRORS (Retrieved Reasoning) ===
"""

error_count = 0
for example in retrieved_results:
    if not example['correct'] and error_count < 5:
        summary_text += f"""
Question: {example['question']}
Expected: {example['expected']}
Predicted: {example['predicted']}
Generated: {example['generated_text']}
Type: {example['reasoning_type']}
---"""
        error_count += 1

# Save summary
with open(f"{results_dir}/summary.txt", "w") as f:
    f.write(summary_text)

print(summary_text)

print(f"\n=== DETAILED RESULTS SAVED ===")
print(f"Directory: {results_dir}")
print(f"Files:")
print(f"  - evaluation_report.json (complete results)")
print(f"  - gold_reasoning_examples.json ({len(gold_results)} examples)")
print(f"  - retrieved_reasoning_examples.json ({len(retrieved_results)} examples)")
print(f"  - evidence_recall_examples.json ({len(recall_examples)} examples)")
print(f"  - summary.txt (human-readable report)")

# Quick debug of a single example
print(f"\n=== QUICK DEBUG EXAMPLE ===")
if gold_results:
    example = gold_results
    print(f"Question: {example['question']}")
    print(f"Expected: {example['expected']}")
    print(f"Predicted: {example['predicted']}")
    print(f"Generated: {example['generated_text']}")
    print(f"Correct: {example['correct']}")
    print(f"Reasoning chain: {example['reasoning_chain']}")

def evaluate_novel_entities():
    """Test on completely unseen entities - the key test for knowledge-free reasoning."""
    print("\n[eval] Testing on novel entities...")
    
    # Load novel entity test set
    try:
        with open("corpus/qa_novel_test.jsonl", "r") as f:
            novel_qa = [json.loads(line.strip()) for line in f]
        
        print(f"[eval] Loaded {len(novel_qa)} novel entity test examples")
        
        # Test with gold reasoning (should work if truly knowledge-free)
        novel_results, novel_by_hops = detailed_accuracy_by_hops(novel_qa[:200], use_gold_reasoning=True)
        
        # Extract stats
        novel_stats = extract_stats(novel_results)
        
        print(f"\n=== NOVEL ENTITY RESULTS ===")
        print(f"Overall accuracy: {novel_stats['overall_accuracy']:.3f} ({novel_stats['correct']}/{novel_stats['total']})")
        print(f"1-hop accuracy: {novel_stats['hop1_accuracy']:.3f} ({novel_stats['hop1_correct']}/{novel_stats['hop1_total']})")
        print(f"2-hop accuracy: {novel_stats['hop2_accuracy']:.3f} ({novel_stats['hop2_correct']}/{novel_stats['hop2_total']})")
        
        # Show some examples
        print(f"\n=== SAMPLE NOVEL ENTITY EXAMPLES ===")
        for i, example in enumerate(novel_results[:3]):
            print(f"Example {i+1}:")
            print(f"  Q: {example['question']}")
            print(f"  Expected: {example['expected']}")
            print(f"  Got: {example['predicted']}")
            print(f"  Correct: {example['correct']}")
            print(f"  Chain: {example['reasoning_chain']}")
        
        return novel_stats
        
    except FileNotFoundError:
        print("[eval] Novel entity test file not found. Run create_novel_test_data.py first.")
        return None

def evaluate_corrupted_reasoning():
    """Test robustness to corrupted reasoning chains."""
    # Take valid reasoning chains and shuffle/corrupt them
    # If model relies on logic, accuracy should drop dramatically
    pass

def evaluate_novel_reasoning_patterns():
    """Test on reasoning patterns not seen during training."""
    # Create new types of multi-hop questions
    # E.g., if training had A→B→C, test A→B→C→D
    pass

def evaluate_compositional_generalization():
    """Test combining reasoning steps in novel ways."""
    # Mix reasoning patterns that were trained separately
    pass

# Test on novel entities if available
print("\n" + "="*60)
print("TESTING ON NOVEL ENTITIES")
print("="*60)
novel_stats = evaluate_novel_entities()

if novel_stats:
    # Add novel entity results to the summary
    summary_text += f"""

=== NOVEL ENTITY RESULTS ===
Novel Entity Accuracy:       {novel_stats['overall_accuracy']:.3f} ({novel_stats['correct']}/{novel_stats['total']})
  - 1-hop questions:         {novel_stats['hop1_accuracy']:.3f} ({novel_stats['hop1_correct']}/{novel_stats['hop1_total']})
  - 2-hop questions:         {novel_stats['hop2_accuracy']:.3f} ({novel_stats['hop2_correct']}/{novel_stats['hop2_total']})

=== KNOWLEDGE-FREE REASONING TEST ===
"""
    
    if novel_stats['overall_accuracy'] > 0.8:
        summary_text += "🎉 AMAZING: Model truly learned knowledge-free reasoning!\n"
    elif novel_stats['overall_accuracy'] > 0.5:
        summary_text += "⚠️  PARTIAL: Some generalization, but still entity-dependent\n"
    else:
        summary_text += "❌ MEMORIZATION: Model memorized entities, didn't learn patterns\n"