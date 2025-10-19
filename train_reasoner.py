import json
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from torch.utils.data import DataLoader, Dataset
import os
from tqdm import tqdm
from datetime import datetime
import glob
import re
import random
import matplotlib.pyplot as plt
import matplotlib.style as style

from config import CONFIG
from build_retriever import load_documents, OracleRetriever
from utils import prepare_training_example

class ReasoningDataset(Dataset):
    def __init__(self, qa_data, retriever, tokenizer, max_length=512):
        self.qa_data = qa_data
        self.retriever = retriever
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.qa_data)
    
    def __getitem__(self, idx):
        qa_example = self.qa_data[idx]
        
        # Retrieve documents using oracle retriever (silently)
        retrieved_docs, _ = self.retriever.oracle_retrieve(
            qa_example['question'], 
            qa_example,
            top_k=CONFIG["TOP_K_DOCS"],
            verbose=False
        )
        
        # Prepare training example with aliasing
        training_example = prepare_training_example(qa_example, retrieved_docs)
        
        # Get the aliased target
        target = training_example['target'].strip()
        if not target:
            # Fallback: use aliased answer directly
            alias_map = training_example.get('alias_map', {})
            target = alias_map.get(qa_example['answer'], qa_example['answer'])
        
        # Construct the full training sequence: prompt + answer
        prompt = training_example['prompt'].rstrip()
        
        # Add reasoning section and answer
        if not prompt.endswith('<REASON>'):
            prompt = prompt + '\n<REASON>'
        
        # The full sequence should be: prompt + reasoning_marker + target + EOS
        full_sequence = prompt + '\n' + target + self.tokenizer.eos_token
        
        # DEBUG: Print first few examples to see what's happening
        if idx < 3:
            print(f"\n=== DEBUG Training Example {idx} ===")
            print(f"Target: '{target}'")
            print(f"EOS token: '{self.tokenizer.eos_token}' (ID: {self.tokenizer.eos_token_id})")
            print(f"Full sequence ends with: '{full_sequence[-20:]}'")
            
            # Check tokenization
            target_with_eos = target + self.tokenizer.eos_token
            target_tokens = self.tokenizer.encode(target_with_eos, add_special_tokens=False)
            print(f"Target + EOS tokens: {target_tokens}")
            print(f"Decoded back: '{self.tokenizer.decode(target_tokens)}'")
            print()
        
        # Tokenize the full sequence
        full_tokens = self.tokenizer.encode(full_sequence, add_special_tokens=False)
        
        # Find where the target starts by tokenizing prompt separately
        prompt_tokens = self.tokenizer.encode(prompt + '\n', add_special_tokens=False)
        prompt_length = len(prompt_tokens)
        
        # CRITICAL: Filter out invalid token IDs
        vocab_size = len(self.tokenizer)
        full_tokens = [t for t in full_tokens if 0 <= t < vocab_size]
        
        # CRITICAL: Ensure EOS token is always present at the end
        if len(full_tokens) == 0 or full_tokens[-1] != self.tokenizer.eos_token_id:
            full_tokens.append(self.tokenizer.eos_token_id)
        
        # Ensure we have some target tokens (including EOS)
        if len(full_tokens) <= prompt_length:
            # Add a safe fallback target + EOS
            target_fallback = self.tokenizer.encode(" unknown", add_special_tokens=False)
            target_fallback = [t for t in target_fallback if 0 <= t < vocab_size]
            full_tokens.extend(target_fallback)
            full_tokens.append(self.tokenizer.eos_token_id)
        
        # REMOVE ALL TRUNCATION - just pad to max_length
        # If sequence is longer than max_length, that's a data problem, not a truncation problem
        if len(full_tokens) > self.max_length:
            print(f"WARNING: Sequence too long ({len(full_tokens)} > {self.max_length}), truncating properly")
            # Truncate the sequence but ensure we keep the target part
            # Try to keep as much of the target as possible
            available_target_space = self.max_length - prompt_length - 1  # -1 for EOS
            if available_target_space < 2:
                # If we can't fit any target, truncate the prompt
                max_prompt_length = self.max_length - 5  # Leave space for at least some target
                prompt_tokens = prompt_tokens[:max_prompt_length]
                prompt_length = len(prompt_tokens)
                available_target_space = self.max_length - prompt_length - 1
            
            # Create a proper sequence
            target_start = prompt_length
            target_tokens = full_tokens[target_start:]
            
            # Keep only what fits
            if len(target_tokens) > available_target_space:
                target_tokens = target_tokens[:available_target_space-1]  # -1 for EOS
            
            # Ensure EOS is at the end
            if not target_tokens or target_tokens[-1] != self.tokenizer.eos_token_id:
                target_tokens.append(self.tokenizer.eos_token_id)
            
            # Reconstruct the full sequence
            full_tokens = prompt_tokens + target_tokens
        
        # Pad to max_length
        input_ids = full_tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(full_tokens))
        input_ids = input_ids[:self.max_length]
        
        # Create labels: -100 for prompt tokens, actual tokens for target
        labels = [-100] * len(input_ids)
        
        # Find actual prompt length in the final sequence
        actual_prompt_tokens = self.tokenizer.encode(prompt + '\n', add_special_tokens=False)
        actual_prompt_tokens = [t for t in actual_prompt_tokens if 0 <= t < vocab_size]
        actual_prompt_length = min(len(actual_prompt_tokens), len(input_ids))
        
        # Set labels for target tokens only
        for i in range(actual_prompt_length, len(input_ids)):
            if input_ids[i] != self.tokenizer.pad_token_id:
                labels[i] = input_ids[i]
        
        # NEW DEBUG: Check if EOS is in labels
        if idx < 3:
            target_labels = [labels[i] for i in range(len(labels)) if labels[i] != -100]
            print(f"Target labels: {target_labels}")
            print(f"EOS in target labels: {self.tokenizer.eos_token_id in target_labels}")
            print(f"Last target label: {target_labels[-1] if target_labels else 'None'}")
            print(f"Actual prompt length: {actual_prompt_length}")
            print(f"Total sequence length: {len(full_tokens)}")
        
        # Ensure we have at least one target token
        if not any(label != -100 for label in labels):
            print(f"WARNING: No target tokens found, creating minimal target")
            # Create a minimal target at the end
            if len(input_ids) > 2:
                labels[-2] = self.tokenizer.eos_token_id
                input_ids[-2] = self.tokenizer.eos_token_id
        
        # Create attention mask
        attention_mask = [1 if token_id != self.tokenizer.pad_token_id else 0 for token_id in input_ids]
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'qa_example': qa_example,
            'training_example': training_example
        }

def custom_collate_fn(batch):
    """Custom collate function to handle the batch properly."""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    qa_examples = [item['qa_example'] for item in batch]
    training_examples = [item['training_example'] for item in batch]
    
    return {
        'input_ids': input_ids,
        'labels': labels,
        'attention_mask': attention_mask,
        'qa_example': qa_examples,
        'training_example': training_examples
    }

def setup_tokenizer():
    """Setup tokenizer with special tokens."""
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # DON'T set pad_token = eos_token!
    # Add a proper pad token instead
    special_tokens = ['<PAD>', '<Q>', '<DOC_1>', '<DOC_2>', '<DOC_3>', '<DOC_4>', '<DOC_5>', '<DOC_6>', '<REASON>']
    
    # Add entity tokens
    entity_pools = {
        'PERSON': 10, 'COMPANY': 10, 'LOCATION': 10, 'SKILL': 10,
        'PROJECT': 5, 'TECHNOLOGY': 5, 'DATE': 5
    }
    
    for entity_type, count in entity_pools.items():
        for i in range(1, count + 1):
            special_tokens.append(f'<{entity_type}{i:02d}>')
    
    # Add all tokens to tokenizer
    tokenizer.add_tokens(special_tokens)
    
    # Set pad token to our custom pad token
    tokenizer.pad_token = '<PAD>'
    
    print(f"EOS token: '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id})")
    print(f"PAD token: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")
    
    return tokenizer

def setup_model(tokenizer):
    """Setup model - either from pre-trained GPT-2 or from scratch."""
    
    if CONFIG["TRAIN_FROM_SCRATCH"]:
        print("🔥 Training from scratch (random initialization)")
        
        # Create model from config with random weights
        config = GPT2Config(
            vocab_size=len(tokenizer),
            n_positions=1024,
            n_embd=768,        # Same as your CONFIG["HIDDEN_SIZE"]
            n_layer=12,        # Standard GPT-2
            n_head=12,         # Standard GPT-2
            n_inner=3072,      # Standard GPT-2
            activation_function="gelu_new",
            resid_pdrop=0.1,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
            bos_token_id=50256,
            eos_token_id=50256,
            pad_token_id=tokenizer.pad_token_id,
        )
        model = GPT2LMHeadModel(config)
        
    else:
        print("🔄 Starting from pre-trained GPT-2")
        # Start with pre-trained GPT-2
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        
        # Resize embeddings for new tokens
        model.resize_token_embeddings(len(tokenizer))
    
    return model

def validate_batch(batch, tokenizer, batch_name=""):
    """Validate a batch for debugging."""
    print(f"Validating {batch_name} batch...")
    
    input_ids = batch['input_ids']
    labels = batch['labels']
    attention_mask = batch['attention_mask']
    
    vocab_size = len(tokenizer)
    
    # Check for invalid token IDs
    invalid_input = torch.any(input_ids >= vocab_size)
    invalid_labels = torch.any((labels >= vocab_size) & (labels != -100))
    
    if invalid_input:
        print(f"❌ Invalid input token IDs found!")
        return False
    
    if invalid_labels:
        print(f"❌ Invalid label token IDs found!")
        return False
    
    # Check if we have valid labels
    valid_label_indices = torch.where(labels[0] != -100)[0]
    
    if len(valid_label_indices) > 0:
        first_label_idx = valid_label_indices[0].item()
        last_label_idx = valid_label_indices[-1].item()
        
        prompt_part = tokenizer.decode(input_ids[0][:first_label_idx], skip_special_tokens=False)
        target_part = tokenizer.decode(input_ids[0][first_label_idx:last_label_idx+1], skip_special_tokens=False)
        
        print(f"Prompt part ends with: ...{prompt_part[-50:]}")
        print(f"Target part: '{target_part}'")
        print(f"Valid labels count: {len(valid_label_indices)}")
        
        return True
    else:
        print("❌ No valid labels found!")
        return False

def generate_test_samples(model, tokenizer, device, results_dir, step, max_new_tokens=10, num_samples=5, run_sanity_test=True):
    """Generate sample predictions using TEST data with both aliased and unaliased versions."""
    
    print(f"Loading test data for validation...")
    
    try:
        # Load test data
        with open("corpus/qa_test.jsonl", "r") as f:
            test_qa_data = [json.loads(line.strip()) for line in f]
        
        test_documents = load_documents("corpus/docs_test.jsonl")
        test_retriever = OracleRetriever(test_documents)
        
        print(f"Loaded {len(test_qa_data)} test QA examples, {len(test_documents)} test documents")
        
        # Sample a few examples for quick validation
        test_sample = random.sample(test_qa_data, min(num_samples, len(test_qa_data)))
        
        # PART 1: Generate ALIASED test predictions (the regular training format)
        print(f"\n--- Part 1: Aliased Test Samples ---")
        
        # Create test dataset with aliasing
        test_dataset = ReasoningDataset(test_sample, test_retriever, tokenizer, CONFIG["MAX_LENGTH"])
        test_dataloader = DataLoader(test_dataset, batch_size=len(test_sample), shuffle=False, collate_fn=custom_collate_fn)
        test_batch = next(iter(test_dataloader))
        
        model.eval()
        aliased_samples = []
        aliased_correct = 0
        
        with torch.no_grad():
            for i in range(len(test_batch['qa_example'])):
                qa_example = test_batch['qa_example'][i]
                training_example = test_batch['training_example'][i]
                
                # Get the aliased prompt (everything before target)
                input_ids = test_batch['input_ids'][i].to(device)
                labels = test_batch['labels'][i]
                
                # Find where the target starts (first non -100 label)
                valid_label_indices = torch.where(labels != -100)[0]
                if len(valid_label_indices) == 0:
                    continue
                    
                prompt_length = valid_label_indices[0].item()
                prompt_ids = input_ids[:prompt_length].unsqueeze(0)
                
                # Generate with proper EOS stopping and temperature
                generated = model.generate(
                    prompt_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.8,  # Add some temperature for better generation
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    # Remove early_stopping - not valid for GPT2
                )
                
                # Decode generated tokens
                generated_tokens = generated[0][prompt_length:]
                
                # DEBUG: Check what was actually generated
                if i < 3:  # Debug first few examples
                    print(f"DEBUG: Generated token IDs: {generated_tokens.tolist()}")
                    print(f"DEBUG: EOS token ID: {tokenizer.eos_token_id}")
                    print(f"DEBUG: Generated length: {len(generated_tokens)}")
                
                # Find EOS token and truncate there
                if tokenizer.eos_token_id in generated_tokens:
                    eos_pos = (generated_tokens == tokenizer.eos_token_id).nonzero(as_tuple=True)[0][0]
                    generated_tokens = generated_tokens[:eos_pos]
                
                generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=False)
                
                # Handle empty predictions
                if not generated_text.strip():
                    print(f"WARNING: Empty prediction for aliased sample {i+1}")
                    generated_text = "EMPTY_PREDICTION"
                
                # Get the expected aliased target
                expected_aliased = training_example['target']
                
                # Get alias map to convert back
                alias_map = training_example.get('alias_map', {})
                reverse_alias_map = {v: k for k, v in alias_map.items()}
                
                # Convert ALL aliases back to original format
                predicted_real = generated_text.strip()
                for alias, original in reverse_alias_map.items():
                    predicted_real = predicted_real.replace(alias, original)
                
                # Extract aliased question from training example
                aliased_prompt = training_example['prompt']
                aliased_question = aliased_prompt.split('\n')[0].replace('<Q> ', '').strip()
                
                print(f"=== Aliased Test Sample {i+1} ===")
                print(f"Model saw (aliased): {aliased_question}")
                print(f"Model generated (aliased): '{generated_text.strip()}'")
                print(f"Expected (aliased): '{expected_aliased}'")
                print(f"---")
                print(f"Original question: {qa_example['question']}")
                print(f"Predicted (real): '{predicted_real}'")
                print(f"Gold answer (real): '{qa_example['answer']}'")
                print()
                
                # Extract retrieved documents from training example
                retrieved_docs = []
                prompt_lines = training_example['prompt'].split('\n')
                for line in prompt_lines:
                    if line.startswith('<DOC_'):
                        # Remove the <DOC_X> prefix and clean up
                        doc_content = line.split('> ', 1)[-1] if '> ' in line else line
                        retrieved_docs.append(doc_content.strip())
                
                # Check if prediction is correct
                is_correct = (predicted_real == qa_example['answer'])
                if is_correct:
                    aliased_correct += 1
                
                aliased_samples.append({
                    'question': qa_example['question'],
                    'aliased_question': aliased_question,
                    'gold_answer': qa_example['answer'],
                    'expected_aliased': expected_aliased,
                    'prediction': predicted_real,
                    'prediction_aliased': generated_text.strip(),
                    'reasoning_type': qa_example.get('reasoning_type', 'unknown'),
                    'retrieved_docs': retrieved_docs,
                    'full_prompt': training_example['prompt'],
                    'is_correct': is_correct,
                    'data_source': 'TEST_ALIASED'
                })
        
        aliased_accuracy = aliased_correct / len(aliased_samples) if len(aliased_samples) > 0 else 0.0
        
        # PART 2: Generate UNALIASED test predictions (sanity test with original names)
        unaliased_samples = []
        unaliased_accuracy = 0.0
        
        if run_sanity_test:
            print(f"\n--- Part 2: Unaliased Test Samples (Original Names) ---")
            
            unaliased_samples = []
            unaliased_correct = 0
            
            with torch.no_grad():
                for i, qa_example in enumerate(test_sample):
                    # Get retrieved docs (but don't alias them)
                    retrieved_docs, _ = test_retriever.oracle_retrieve(
                        qa_example['question'], 
                        qa_example,
                        top_k=CONFIG["TOP_K_DOCS"],
                        verbose=False
                    )
                    
                    # Build prompt with ORIGINAL entity names (no aliasing)
                    original_question = qa_example['question']
                    original_prompt = f"<Q> {original_question}\n"
                    
                    for j, doc in enumerate(retrieved_docs[:3], 1):
                        original_prompt += f"<DOC_{j}> {doc}\n"
                    
                    original_prompt += "<REASON>"
                    
                    print(f"=== Unaliased Test Sample {i+1} ===")
                    print(f"Original question: {original_question}")
                    print(f"Expected answer: {qa_example['answer']}")
                    
                    # Tokenize the prompt
                    prompt_tokens = tokenizer.encode(original_prompt, add_special_tokens=False)
                    prompt_ids = torch.tensor(prompt_tokens, dtype=torch.long).unsqueeze(0).to(device)
                    
                    # Generate response
                    generated = model.generate(
                        prompt_ids,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=1.0,  # Add some temperature for better generation
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        # Remove early_stopping - not valid for GPT2
                    )
                    
                    # Decode generated tokens
                    generated_tokens = generated[0][len(prompt_tokens):]
                    
                    # DEBUG: Check what was actually generated
                    if i < 3:  # Debug first few examples
                        print(f"DEBUG: Generated token IDs: {generated_tokens.tolist()}")
                        print(f"DEBUG: EOS token ID: {tokenizer.eos_token_id}")
                        print(f"DEBUG: Generated length: {len(generated_tokens)}")
                    
                    # Find EOS token and truncate there
                    if tokenizer.eos_token_id in generated_tokens:
                        eos_pos = (generated_tokens == tokenizer.eos_token_id).nonzero(as_tuple=True)[0][0]
                        generated_tokens = generated_tokens[:eos_pos]
                    
                    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=False)
                    predicted_answer = generated_text.strip()
                    
                    # Handle empty predictions
                    if not predicted_answer:
                        print(f"WARNING: Empty prediction for unaliased sample {i+1}")
                        predicted_answer = "EMPTY_PREDICTION"
                    
                    # Check if prediction is correct
                    is_correct = (predicted_answer == qa_example['answer'])
                    if is_correct:
                        unaliased_correct += 1
                    
                    print(f"Model prediction: '{predicted_answer}'")
                    print(f"Correct: {'✅' if is_correct else '❌'}")
                    print()
                    
                    unaliased_samples.append({
                        'question': original_question,
                        'gold_answer': qa_example['answer'],
                        'prediction': predicted_answer,
                        'reasoning_type': qa_example.get('reasoning_type', 'unknown'),
                        'retrieved_docs': retrieved_docs[:3],
                        'full_prompt': original_prompt,
                        'is_correct': is_correct,
                        'data_source': 'TEST_UNALIASED'
                    })
            
            unaliased_accuracy = unaliased_correct / len(unaliased_samples) if len(unaliased_samples) > 0 else 0.0
        
        # Save BOTH test results
        sample_data = {
            'step': step,
            'aliased_test': {
                'accuracy': aliased_accuracy,
                'correct_predictions': aliased_correct,
                'total_samples': len(aliased_samples),
                'samples': aliased_samples
            },
            'unaliased_test': {
                'accuracy': unaliased_accuracy,
                'correct_predictions': len([s for s in unaliased_samples if s['is_correct']]),
                'total_samples': len(unaliased_samples),
                'samples': unaliased_samples
            }
        }
        
        test_sample_file = os.path.join(results_dir, f"test_samples_step_{step}.json")
        with open(test_sample_file, 'w') as f:
            json.dump(sample_data, f, indent=2)
        
        print(f"\n=== SUMMARY ===")
        print(f"ALIASED Test Accuracy: {aliased_accuracy:.2%} ({aliased_correct}/{len(aliased_samples)})")
        if run_sanity_test:
            print(f"UNALIASED Test Accuracy: {unaliased_accuracy:.2%} ({len([s for s in unaliased_samples if s['is_correct']])}/{len(unaliased_samples)}) - original entity names")
        
        model.train()
        return aliased_samples, aliased_accuracy
        
    except Exception as e:
        print(f"❌ Test sample generation failed: {e}")
        model.train()
        return [], 0.0

def cleanup_old_checkpoints(models_dir, max_checkpoints=2):
    """Keep only the most recent checkpoints."""
    checkpoint_files = glob.glob(os.path.join(models_dir, "reasoner_epoch_*.pt"))
    if len(checkpoint_files) > max_checkpoints:
        # Sort by modification time
        checkpoint_files.sort(key=os.path.getmtime)
        # Remove oldest files
        for old_file in checkpoint_files[:-max_checkpoints]:
            try:
                os.remove(old_file)
                print(f"Removed old checkpoint: {old_file}")
            except Exception as e:
                print(f"Failed to remove {old_file}: {e}")

def track_reasoning_skill_acquisition(model, tokenizer, device, results_dir, step, num_samples_per_hop=50):
    """Track accuracy by reasoning type over training steps."""
    
    try:
        # Load test data
        with open("corpus/qa_test.jsonl", "r") as f:
            test_qa_data = [json.loads(line.strip()) for line in f]
        
        test_documents = load_documents("corpus/docs_test.jsonl")
        test_retriever = OracleRetriever(test_documents)
        
        # Group by reasoning type
        by_type = {'1-hop': [], '2-hop': [], '3-hop': [], '4-hop': []}
        for item in test_qa_data:
            reasoning_type = item.get('reasoning_type', 'unknown')
            if reasoning_type in by_type:
                by_type[reasoning_type].append(item)
        
        # Sample equal amounts from each type for balanced evaluation
        hop_results = {}
        
        for hop_type, examples in by_type.items():
            if len(examples) == 0:
                continue
                
            # Sample from this hop type
            hop_sample = random.sample(examples, min(num_samples_per_hop, len(examples)))
            
            # Evaluate on this hop type
            hop_dataset = ReasoningDataset(hop_sample, test_retriever, tokenizer, CONFIG["MAX_LENGTH"])
            hop_dataloader = DataLoader(hop_dataset, batch_size=16, shuffle=False, collate_fn=custom_collate_fn)
            
            correct = 0
            total = 0
            
            model.eval()
            with torch.no_grad():
                for batch in hop_dataloader:
                    batch_size = len(batch['qa_example'])
                    
                    for i in range(batch_size):
                        qa_example = batch['qa_example'][i]
                        training_example = batch['training_example'][i]
                        
                        # Get prompt
                        input_ids = batch['input_ids'][i].to(device)
                        labels = batch['labels'][i]
                        
                        # Find target start
                        valid_label_indices = torch.where(labels != -100)[0]
                        if len(valid_label_indices) == 0:
                            continue
                            
                        prompt_length = valid_label_indices[0].item()
                        prompt_ids = input_ids[:prompt_length].unsqueeze(0)
                        
                        # Generate
                        generated = model.generate(
                            prompt_ids,
                            max_new_tokens=20,
                            do_sample=True,
                            temperature=0.7,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                        )
                        
                        # Decode and process
                        generated_tokens = generated[0][prompt_length:]
                        if tokenizer.eos_token_id in generated_tokens:
                            eos_pos = (generated_tokens == tokenizer.eos_token_id).nonzero(as_tuple=True)[0][0]
                            generated_tokens = generated_tokens[:eos_pos]
                        
                        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                        
                        # Convert back from aliases
                        alias_map = training_example.get('alias_map', {})
                        reverse_alias_map = {v: k for k, v in alias_map.items()}
                        
                        predicted_real = generated_text.strip()
                        for alias, original in reverse_alias_map.items():
                            predicted_real = predicted_real.replace(alias, original)
                        
                        # Extract entity if needed
                        import re
                        entity_match = re.search(r'[PCLSD]\d+|PJ\d+', predicted_real)
                        if entity_match:
                            predicted_real = entity_match.group(0)
                        
                        # Check correctness
                        total += 1
                        if predicted_real == qa_example['answer']:
                            correct += 1
            
            # Store results for this hop type
            accuracy = correct / total if total > 0 else 0.0
            hop_results[hop_type] = {
                'accuracy': accuracy,
                'correct': correct,
                'total': total
            }
            
            print(f"Step {step} - {hop_type}: {accuracy:.3f} ({correct}/{total})")
        
        # Save progression data
        progression_entry = {
            'step': step,
            'timestamp': datetime.now().isoformat(),
            'hop_accuracies': hop_results
        }
        
        # Load existing progression data
        progression_file = os.path.join(results_dir, "reasoning_progression.json")
        if os.path.exists(progression_file):
            with open(progression_file, 'r') as f:
                progression_data = json.load(f)
        else:
            progression_data = []
        
        # Add new entry
        progression_data.append(progression_entry)
        
        # Save updated progression
        with open(progression_file, 'w') as f:
            json.dump(progression_data, f, indent=2)
        
        model.train()
        return hop_results
        
    except Exception as e:
        print(f"❌ Reasoning skill tracking failed: {e}")
        model.train()
        return {}

def plot_reasoning_skill_acquisition(results_dir, step=None):
    """Plot how reasoning skills develop over training."""
    
    try:
        # Load progression data
        progression_file = os.path.join(results_dir, "reasoning_progression.json")
        if not os.path.exists(progression_file):
            print("No progression data found for plotting.")
            return
            
        with open(progression_file, 'r') as f:
            progression_data = json.load(f)
        
        if len(progression_data) == 0:
            print("No progression data available for plotting.")
            return
        
        # Extract data for plotting
        
        steps = [entry['step'] for entry in progression_data]
        hop_types = ['1-hop', '2-hop', '3-hop', '4-hop']
        
        # Set up the plot with nice styling
        plt.style.use('default')  # Clean style
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Nice color palette
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        markers = ['o', 's', '^', 'D']
        
        # Plot each reasoning type
        for i, hop_type in enumerate(hop_types):
            accuracies = []
            valid_steps = []
            
            for j, entry in enumerate(progression_data):
                if hop_type in entry['hop_accuracies']:
                    accuracies.append(entry['hop_accuracies'][hop_type]['accuracy'])
                    valid_steps.append(entry['step'])
            
            if len(accuracies) > 0:
                ax.plot(valid_steps, accuracies, 
                       marker=markers[i], linewidth=2.5, markersize=6,
                       label=f'{hop_type} Reasoning', 
                       color=colors[i], alpha=0.8)
        
        # Customize the plot
        ax.set_xlabel('Training Step', fontsize=14, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
        ax.set_title('Reasoning Skill Acquisition During Training', 
                    fontsize=16, fontweight='bold', pad=20)
        
        # Set y-axis limits and grid
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)
        
        # Legend with nice styling - moved to upper left
        legend = ax.legend(loc='upper left', fontsize=12, 
                          frameon=True, fancybox=True, shadow=True)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(0.9)
        
        # Add current step annotation if provided
        if step is not None and len(steps) > 0:
            ax.axvline(x=step, color='gray', linestyle='--', alpha=0.7, linewidth=2)
            ax.text(step, 0.95, f'Step {step}', rotation=90, 
                   verticalalignment='top', horizontalalignment='right',
                   fontsize=11, alpha=0.8, fontweight='bold')
        

        
        # Tight layout and save
        plt.tight_layout()
        
        # Always save to the same filename - just update it
        plot_path = os.path.join(results_dir, 'reasoning_skill_acquisition.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        
        print(f"📊 Reasoning progression plot updated: reasoning_skill_acquisition.png")
        
        # Close to free memory
        plt.close()
        
    except Exception as e:
        print(f"❌ Failed to generate progression plot: {e}")
        # Close any open figures
        plt.close('all')


def train_model():
    """Main training function with aliasing."""
    print("[train_reasoner] Setting up training...")
    
    # Create timestamped results folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/training_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    try:
        with open("corpus/qa_train.jsonl", "r") as f:
            qa_data = [json.loads(line.strip()) for line in f]
        documents = load_documents("corpus/docs_train.jsonl")
        print(f"Loaded {len(qa_data)} QA examples, {len(documents)} documents")
    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        return

    # Initialize components
    print("Initializing model and tokenizer...")
    retriever = OracleRetriever(documents)
    tokenizer = setup_tokenizer()
    model = setup_model(tokenizer).to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=CONFIG["LEARNING_RATE"],
        weight_decay=0.01,
        eps=1e-8,
        betas=(0.9, 0.999)
    )
    
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Vocabulary: {len(tokenizer)} tokens")
    
    # Create dataset
    dataset = ReasoningDataset(qa_data, retriever, tokenizer, CONFIG["MAX_LENGTH"])
    
    # Create dataloader
    dataloader = DataLoader(
        dataset, 
        batch_size=CONFIG["BATCH_SIZE"], 
        shuffle=True,
        collate_fn=custom_collate_fn
    )
    
    # Calculate training info
    steps_per_epoch = len(dataloader)
    total_steps = CONFIG["NUM_EPOCHS"] * steps_per_epoch
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Batch size: {CONFIG['BATCH_SIZE']}")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Training for {CONFIG['NUM_EPOCHS']} epochs = {total_steps} total steps")
    print(f"Results will be saved to: {results_dir}")
    
    # Validate first batch before training
    print("\n🔍 Validating data before training...")
    first_batch = next(iter(dataloader))
    if not validate_batch(first_batch, tokenizer, "First"):
        print("❌ Data validation failed! Stopping training.")
        return
    
    # Test forward pass with first batch
    print("🔍 Testing forward pass...")
    model.eval()
    with torch.no_grad():
        try:
            input_ids = first_batch['input_ids'].to(device)
            labels = first_batch['labels'].to(device)
            attention_mask = first_batch['attention_mask'].to(device)
            
            outputs = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
            loss = outputs.loss
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"❌ Forward pass produces invalid loss: {loss.item()}")
                return
            else:
                print(f"✅ Forward pass successful, loss: {loss.item():.4f}")
                
        except Exception as e:
            print(f"❌ Forward pass failed: {e}")
            return
    
    # Training loop
    print("\n🚀 Starting training...")
    model.train()
    step = 0
    
    # Initialize training progress tracking
    training_progress = []
    
    for epoch in range(CONFIG["NUM_EPOCHS"]):
        # Create progress bar
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{CONFIG['NUM_EPOCHS']}")
        
        for batch in pbar:
            step += 1
            
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Validate token IDs
            vocab_size = len(tokenizer)
            input_ids = torch.clamp(input_ids, 0, vocab_size - 1)
            valid_label_mask = (labels != -100)
            labels = torch.where(valid_label_mask, 
                               torch.clamp(labels, 0, vocab_size - 1), 
                               labels)
            
            # Forward pass
            outputs = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'step': f'{step}/{total_steps}'})
            
            # Log progress every 100 steps
            if step % 100 == 0:
                print(f"\nStep {step}/{total_steps}: Loss = {loss.item():.4f}")
                
                # Save training progress
                progress_entry = {
                    'step': step,
                    'epoch': epoch + 1,
                    'loss': loss.item(),
                    'timestamp': datetime.now().isoformat()
                }
                training_progress.append(progress_entry)
                
                # Save progress to file
                progress_file = os.path.join(results_dir, "training_progress.json")
                with open(progress_file, 'w') as f:
                    json.dump(training_progress, f, indent=2)
            
            # Generate sample predictions every 50 steps
            if step % 50 == 0:  # Changed from 100 to 50
                print(f"\n=== Step {step} Reasoning Skill Assessment ===")
                
                # Track reasoning skill acquisition
                hop_results = track_reasoning_skill_acquisition(
                    model, tokenizer, device, results_dir, step, 
                    num_samples_per_hop=100  # Increased from 20 to 100 for stability
                )
                
                # Print summary
                if hop_results:
                    print("Current Reasoning Abilities:")
                    for hop_type in ['1-hop', '2-hop', '3-hop', '4-hop']:
                        if hop_type in hop_results:
                            acc = hop_results[hop_type]['accuracy']
                            print(f"  {hop_type}: {acc:.3f}")
            
            # Generate progression plot every 200 steps (or at key milestones)
            if step % 200 == 0 or step % steps_per_epoch == 0:
                print(f"📊 Generating progression plot...")
                plot_reasoning_skill_acquisition(results_dir, step)
            
            # Save checkpoint every epoch
            if step % steps_per_epoch == 0:
                try:
                    checkpoint_path = f"models/reasoner_epoch_{epoch+1}.pt"
                    model_cpu = model.cpu()
                    torch.save({
                        'model_state_dict': model_cpu.state_dict(),
                        'tokenizer': tokenizer,
                        'step': step,
                        'config': CONFIG
                    }, checkpoint_path)
                    model = model.to(device)
                    print(f"Saved checkpoint: {checkpoint_path}")
                    
                    cleanup_old_checkpoints("models", max_checkpoints=2)
                except Exception as e:
                    print(f"Failed to save checkpoint: {e}")

    # Generate final progression plot
    print(f"\n📊 Generating final reasoning progression plot...")
    plot_reasoning_skill_acquisition(results_dir, step)

    # Save final model
    final_path = "models/reasoner_final.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'tokenizer': tokenizer,
        'step': step,
        'config': CONFIG
    }, final_path)
    
    print(f"\nTraining completed! Final model saved: {final_path}")
    print(f"Results saved to: {results_dir}")
    print(f"Training progress saved to: {os.path.join(results_dir, 'training_progress.json')}")
    print(f"Reasoning progression plot: {os.path.join(results_dir, 'reasoning_skill_acquisition.png')}")

if __name__ == "__main__":
    train_model()