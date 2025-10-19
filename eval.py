import json
import random
import torch
import os
from datetime import datetime
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import DataLoader
import glob

# Import the working classes from training
from train_reasoner import ReasoningDataset, custom_collate_fn, setup_tokenizer
from build_retriever import load_documents, OracleRetriever
from config import CONFIG

print("[eval] Loading data and models...")

# Create results directory with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = f"results/eval_{timestamp}"
os.makedirs(results_dir, exist_ok=True)

# Setup tokenizer (same as training)
print("[eval] Setting up tokenizer...")
tokenizer = setup_tokenizer()

# Find latest checkpoint
device = "cpu" #"cuda" if torch.cuda.is_available() else "cpu"
checkpoint_files = glob.glob("models/reasoner_epoch_*.pt")
if checkpoint_files:
    latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    print(f"[eval] Loading latest checkpoint: {latest_checkpoint}")
    checkpoint = torch.load(latest_checkpoint, map_location=device)
else:
    print("[eval] No checkpoint found!")
    exit(1)

# Load model (same architecture as training)
print("[eval] Loading GPT2 model...")
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.resize_token_embeddings(len(tokenizer))
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

def comprehensive_evaluation(model, tokenizer, device, results_dir, num_samples=1000):
    """Comprehensive evaluation using the same method as training."""
    
    print(f"[eval] Loading test data...")
    try:
        # Load test data
        with open("corpus/qa_test.jsonl", "r") as f:
            test_qa_data = [json.loads(line.strip()) for line in f]
        
        test_documents = load_documents("corpus/docs_test.jsonl")
        test_retriever = OracleRetriever(test_documents)
        
        print(f"[eval] Loaded {len(test_qa_data)} test QA examples, {len(test_documents)} test documents")
        
        # Sample for evaluation (or use all if num_samples is larger)
        if num_samples < len(test_qa_data):
            test_sample = random.sample(test_qa_data, num_samples)
        else:
            test_sample = test_qa_data
            
        print(f"[eval] Evaluating on {len(test_sample)} examples")
        
        # Group by reasoning type for analysis
        by_type = {'1-hop': [], '2-hop': [], '3-hop': [], '4-hop': []}
        for item in test_sample:
            reasoning_type = item.get('reasoning_type', 'unknown')
            if reasoning_type in by_type:
                by_type[reasoning_type].append(item)
        
        print(f"[eval] Distribution: {[(k, len(v)) for k, v in by_type.items()]}")
        
        # PART 1: Evaluate with proper training format (aliased)
        print(f"\n[eval] === ALIASED EVALUATION (Training Format) ===")
        
        # Create dataset using same pipeline as training
        test_dataset = ReasoningDataset(test_sample, test_retriever, tokenizer, CONFIG["MAX_LENGTH"])
        test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=custom_collate_fn)
        
        aliased_results = []
        aliased_correct = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_dataloader):
                print(f"[eval] Processing batch {batch_idx+1}/{len(test_dataloader)}")
                
                batch_size = len(batch['qa_example'])
                
                for i in range(batch_size):
                    qa_example = batch['qa_example'][i]
                    training_example = batch['training_example'][i]
                    
                    # Get the aliased prompt (everything before target)
                    input_ids = batch['input_ids'][i].to(device)
                    labels = batch['labels'][i]
                    
                    # Find where the target starts (first non -100 label)
                    valid_label_indices = torch.where(labels != -100)[0]
                    if len(valid_label_indices) == 0:
                        continue
                        
                    prompt_length = valid_label_indices[0].item()
                    prompt_ids = input_ids[:prompt_length].unsqueeze(0)
                    
                    # Generate response
                    generated = model.generate(
                        prompt_ids,
                        max_new_tokens=20,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                    
                    # Decode generated tokens
                    generated_tokens = generated[0][prompt_length:]
                    
                    # Find EOS token and truncate
                    if tokenizer.eos_token_id in generated_tokens:
                        eos_pos = (generated_tokens == tokenizer.eos_token_id).nonzero(as_tuple=True)[0][0]
                        generated_tokens = generated_tokens[:eos_pos]
                    
                    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                    
                    # Convert aliases back to original format
                    alias_map = training_example.get('alias_map', {})
                    reverse_alias_map = {v: k for k, v in alias_map.items()}
                    
                    predicted_real = generated_text.strip()
                    for alias, original in reverse_alias_map.items():
                        predicted_real = predicted_real.replace(alias, original)
                    
                    # Extract entity ID if needed
                    import re
                    entity_match = re.search(r'[PCLSD]\d+|PJ\d+', predicted_real)
                    if entity_match:
                        predicted_real = entity_match.group(0)
                    
                    # Check correctness
                    is_correct = (predicted_real == qa_example['answer'])
                    if is_correct:
                        aliased_correct += 1
                    
                    # Store result
                    result = {
                        'question': qa_example['question'],
                        'gold_answer': qa_example['answer'],
                        'prediction': predicted_real,
                        'generated_text': generated_text.strip(),
                        'reasoning_type': qa_example.get('reasoning_type', 'unknown'),
                        'reasoning_chain': qa_example.get('facts', []),
                        'is_correct': is_correct,
                        'prompt': training_example['prompt'],
                        'data_source': 'ALIASED'
                    }
                    aliased_results.append(result)
        
        aliased_accuracy = aliased_correct / len(aliased_results) if aliased_results else 0.0
        
        # PART 2: Evaluate with original entity names (sanity check)
        print(f"\n[eval] === UNALIASED EVALUATION (Original Names) ===")
        
        unaliased_results = []
        unaliased_correct = 0
        
        with torch.no_grad():
            for i, qa_example in enumerate(test_sample[:200]):  # Limit for speed
                if i % 50 == 0:
                    print(f"[eval] Unaliased progress: {i}/{min(200, len(test_sample))}")
                
                # Get retrieved docs
                retrieved_docs, _ = test_retriever.oracle_retrieve(
                    qa_example['question'], 
                    qa_example,
                    top_k=CONFIG["TOP_K_DOCS"],
                    verbose=False
                )
                
                # Build prompt with original entity names
                original_prompt = f"<Q> {qa_example['question']}\n"
                for j, doc in enumerate(retrieved_docs[:4], 1):
                    original_prompt += f"<DOC_{j}> {doc}\n"
                original_prompt += "<REASON>"
                
                # Generate response
                prompt_tokens = tokenizer.encode(original_prompt, add_special_tokens=False)
                prompt_ids = torch.tensor(prompt_tokens).unsqueeze(0).to(device)
                
                generated = model.generate(
                    prompt_ids,
                    max_new_tokens=20,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                
                # Decode and extract answer
                generated_tokens = generated[0][len(prompt_tokens):]
                if tokenizer.eos_token_id in generated_tokens:
                    eos_pos = (generated_tokens == tokenizer.eos_token_id).nonzero(as_tuple=True)[0][0]
                    generated_tokens = generated_tokens[:eos_pos]
                
                generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                predicted_answer = generated_text.strip()
                
                # Extract entity ID if needed
                import re
                entity_match = re.search(r'[PCLSD]\d+|PJ\d+', predicted_answer)
                if entity_match:
                    predicted_answer = entity_match.group(0)
                
                # Check correctness
                is_correct = (predicted_answer == qa_example['answer'])
                if is_correct:
                    unaliased_correct += 1
                
                # Store result
                result = {
                    'question': qa_example['question'],
                    'gold_answer': qa_example['answer'],
                    'prediction': predicted_answer,
                    'generated_text': generated_text,
                    'reasoning_type': qa_example.get('reasoning_type', 'unknown'),
                    'reasoning_chain': qa_example.get('facts', []),
                    'is_correct': is_correct,
                    'prompt': original_prompt,
                    'data_source': 'UNALIASED'
                }
                unaliased_results.append(result)
        
        unaliased_accuracy = unaliased_correct / len(unaliased_results) if unaliased_results else 0.0
        
        # PART 3: Analyze results by reasoning type
        print(f"\n[eval] === ANALYSIS BY REASONING TYPE ===")
        
        def analyze_by_type(results, result_type):
            by_type_stats = {}
            for result in results:
                r_type = result['reasoning_type']
                if r_type not in by_type_stats:
                    by_type_stats[r_type] = {'correct': 0, 'total': 0}
                by_type_stats[r_type]['total'] += 1
                if result['is_correct']:
                    by_type_stats[r_type]['correct'] += 1
            
            print(f"\n{result_type} Results:")
            for r_type in sorted(by_type_stats.keys()):
                stats = by_type_stats[r_type]
                acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
                print(f"  {r_type}: {stats['correct']}/{stats['total']} = {acc:.3f}")
            
            return by_type_stats
        
        aliased_by_type = analyze_by_type(aliased_results, "ALIASED")
        unaliased_by_type = analyze_by_type(unaliased_results, "UNALIASED")
        
        # PART 4: Save comprehensive results
        print(f"\n[eval] === SAVING RESULTS ===")
        
        # Save detailed results
        detailed_results = {
            'timestamp': timestamp,
            'model_info': {
                'checkpoint': latest_checkpoint,
                'total_parameters': sum(p.numel() for p in model.parameters()),
                'device': str(device)
            },
            'evaluation_info': {
                'total_test_examples': len(test_qa_data),
                'evaluated_examples': len(test_sample),
                'aliased_examples': len(aliased_results),
                'unaliased_examples': len(unaliased_results)
            },
            'results': {
                'aliased_accuracy': aliased_accuracy,
                'unaliased_accuracy': unaliased_accuracy,
                'aliased_correct': aliased_correct,
                'unaliased_correct': unaliased_correct,
                'aliased_by_type': aliased_by_type,
                'unaliased_by_type': unaliased_by_type
            },
            'detailed_results': {
                'aliased': aliased_results,
                'unaliased': unaliased_results
            }
        }
        
        # Save to file
        results_file = os.path.join(results_dir, "comprehensive_evaluation.json")
        with open(results_file, 'w') as f:
            json.dump(detailed_results, f, indent=2)
        
        # Generate summary
        summary = f"""COMPREHENSIVE EVALUATION RESULTS
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Model: {latest_checkpoint}
Results Directory: {results_dir}

=== SUMMARY ===
Total Test Examples: {len(test_qa_data):,}
Evaluated Examples: {len(test_sample):,}

ALIASED (Training Format):
  Accuracy: {aliased_accuracy:.3f} ({aliased_correct}/{len(aliased_results)})
  
UNALIASED (Original Names):
  Accuracy: {unaliased_accuracy:.3f} ({unaliased_correct}/{len(unaliased_results)})

=== BY REASONING TYPE ===
ALIASED:
"""
        
        for r_type in sorted(aliased_by_type.keys()):
            stats = aliased_by_type[r_type]
            acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            summary += f"  {r_type}: {acc:.3f} ({stats['correct']}/{stats['total']})\n"
        
        summary += "\nUNALIASED:\n"
        for r_type in sorted(unaliased_by_type.keys()):
            stats = unaliased_by_type[r_type]
            acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            summary += f"  {r_type}: {acc:.3f} ({stats['correct']}/{stats['total']})\n"
        
        # Show sample errors
        summary += "\n=== SAMPLE ERRORS (ALIASED) ===\n"
        errors = [r for r in aliased_results if not r['is_correct']][:10]
        for i, error in enumerate(errors):
            summary += f"\nError {i+1}:\n"
            summary += f"  Q: {error['question']}\n"
            summary += f"  Expected: {error['gold_answer']}\n"
            summary += f"  Got: {error['prediction']}\n"
            summary += f"  Type: {error['reasoning_type']}\n"
        
        # Save summary
        summary_file = os.path.join(results_dir, "summary.txt")
        with open(summary_file, 'w') as f:
            f.write(summary)
        
        print(summary)
        
        return detailed_results
        
    except Exception as e:
        print(f"[eval] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

# Run comprehensive evaluation
print(f"[eval] Starting comprehensive evaluation...")
results = comprehensive_evaluation(model, tokenizer, device, results_dir, num_samples=1000)

if results:
    print(f"\n[eval] Evaluation completed successfully!")
    print(f"[eval] Results saved to: {results_dir}")
else:
    print(f"\n[eval] Evaluation failed!")