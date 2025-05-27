import json
import torch
import torch.nn as nn
import random
import faiss
import pickle
import time
import sys
import os
from utils import build_alias, apply_alias, build_prompt
from model import TinyDec, tok, device
from config import CONFIG

print(f"[train_reasoner] Using device: {device}")

# Initialize model
model = TinyDec().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["LEARNING_RATE"])

print(f"[train_reasoner] Model initialized with {sum(p.numel() for p in model.parameters())} parameters")

# Load QA data with reasoning chains
print("[train_reasoner] Loading QA data...")
try:
    with open("corpus/qa.jsonl", "r") as f:
        qa_raw = [json.loads(line.strip()) for line in f]
    
    # Keep the full QA data with reasoning chains
    QA = []
    for item in qa_raw:
        if isinstance(item, list) and len(item) >= 3:
            # Format: [question, answer, reasoning_chain]
            question = item[0]
            answer = item[1]
            reasoning_chain = item[2] if len(item) > 2 else []
            QA.append((question, answer, reasoning_chain))
        else:
            print(f"[WARNING] Unexpected QA format: {item}")
            continue
    
    print(f"[train_reasoner] Loaded {len(QA)} QA pairs with reasoning chains")
    
    # Split QA for validation (use 10% for validation)
    val_size = min(1000, len(QA) // 10)
    val_qa = random.sample(QA, val_size)
    train_qa = [qa for qa in QA if qa not in val_qa]
    
    print(f"[train_reasoner] Train: {len(train_qa)}, Validation: {len(val_qa)}")
    
    # Show a few examples
    print("\n[DEBUG] Sample QA pairs:")
    for i, (q, a, chain) in enumerate(QA[:2]):
        print(f"  Q{i+1}: {q}")
        print(f"  A{i+1}: {a}")
        print(f"  Chain: {chain}")
        print()
        
except Exception as e:
    print(f"[ERROR] Failed to load QA data: {e}")
    sys.exit(1)

# Validate tokenizer
print("[train_reasoner] Testing tokenizer...")
test_text = "What is the capital of France?"
test_tokens = tok(test_text, return_tensors="pt")
print(f"[DEBUG] Test tokenization: '{test_text}' → {test_tokens.input_ids.shape[1]} tokens")

def evaluate_model(qa_sample, max_examples=100):
    """Quick evaluation on a sample of QA pairs."""
    model.eval()
    correct = 0
    total = 0
    
    eval_sample = random.sample(qa_sample, min(max_examples, len(qa_sample)))
    
    with torch.no_grad():
        for q, a, reasoning_chain in eval_sample:
            try:
                # Build prompt using reasoning chain
                if reasoning_chain and len(reasoning_chain) > 0:
                    all_texts = [q, a] + reasoning_chain
                    alias, inv_alias = build_alias(all_texts)
                    prompt = build_prompt(q, reasoning_chain, alias)
                    target = apply_alias(a, alias)
                else:
                    prompt = f"Question: {q}\nAnswer:"
                    target = a
                    inv_alias = {}
                
                # Tokenize prompt
                input_ids = tok(prompt, return_tensors="pt", truncation=True, max_length=512).input_ids.to(device)
                
                # Generate
                output = model.generate(
                    input_ids, 
                    max_new_tokens=10,
                    temperature=0.1,
                    do_sample=False
                )
                
                generated_text = tok.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
                
                # Apply inverse aliasing
                for pseudo, real in inv_alias.items():
                    generated_text = generated_text.replace(pseudo, real)
                
                # Extract entity (simple extraction)
                pred = generated_text.strip().split()[0] if generated_text.strip() else ""
                
                if pred == a:
                    correct += 1
                total += 1
                
            except Exception as e:
                # Skip problematic examples during evaluation
                continue
    
    model.train()
    accuracy = correct / total if total > 0 else 0.0
    return accuracy, correct, total

def save_checkpoint(step, loss, val_accuracy, is_best=False):
    """Save model checkpoint with cleanup of old checkpoints."""
    checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'val_accuracy': val_accuracy,
        'config': CONFIG
    }
    
    # Save regular checkpoint
    checkpoint_path = f"corpus/checkpoint_step_{step}.pt"
    torch.save(checkpoint, checkpoint_path)
    
    # Clean up old checkpoints (keep only last 3)
    cleanup_old_checkpoints(keep_last=3)
    
    # Save best model
    if is_best:
        best_path = "corpus/tiny_reasoner.pt"
        torch.save(checkpoint, best_path)
        print(f"[CHECKPOINT] New best model saved! Val accuracy: {val_accuracy:.3f}")
    
    print(f"[CHECKPOINT] Saved checkpoint at step {step}")
    return checkpoint_path

def cleanup_old_checkpoints(keep_last=3):
    """Remove old checkpoint files, keeping only the most recent ones."""
    import glob
    import os
    
    # Find all checkpoint files
    checkpoint_files = glob.glob("corpus/checkpoint_step_*.pt")
    
    if len(checkpoint_files) <= keep_last:
        return  # Nothing to clean up
    
    # Sort by step number (extract step from filename)
    checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    # Remove old files (keep only the last N)
    files_to_remove = checkpoint_files[:-keep_last]
    
    for file_path in files_to_remove:
        try:
            os.remove(file_path)
            step_num = file_path.split('_')[-1].split('.')[0]
            print(f"[CLEANUP] Removed old checkpoint: step_{step_num}")
        except Exception as e:
            print(f"[WARNING] Could not remove {file_path}: {e}")

# Training setup
print("[train_reasoner] Starting training...")
print("=" * 60)
model.train()
start_time = time.time()

# Track statistics
total_examples = 0
skipped_examples = 0
nan_losses = 0
best_val_accuracy = 0.0
eval_interval = 5000  # Evaluate every 5k steps
save_interval = 5000  # Save checkpoint every 5k steps

# Load existing checkpoint if available
checkpoint_path = "corpus/tiny_reasoner.pt"
start_step = 0
if os.path.exists(checkpoint_path):
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_step = checkpoint.get('step', 0)
        best_val_accuracy = checkpoint.get('val_accuracy', 0.0)
        print(f"[train_reasoner] Resumed from step {start_step}, best val accuracy: {best_val_accuracy:.3f}")
    except Exception as e:
        print(f"[WARNING] Could not load checkpoint: {e}")

for step in range(start_step, CONFIG["TRAIN_STEPS"]):
    try:
        # Sample batch
        batch = random.sample(train_qa, CONFIG["BATCH_SIZE"])
        
        total_loss = 0
        valid_examples = 0
        
        for q, a, reasoning_chain in batch:
            try:
                # USE ALIASING - This is the key fix!
                aliased_q, aliased_chain, aliased_a, alias, inv_alias = create_training_example_with_aliasing(q, reasoning_chain, a)
                
                # Build prompt with aliased entities
                if aliased_chain and len(aliased_chain) > 0:
                    prompt = f"<Q> {aliased_q}\n"
                    for i, sent in enumerate(aliased_chain):
                        prompt += f"<DOC_{i+1}> {sent}\n"
                    prompt += "<REASON>"
                    target = aliased_a
                else:
                    prompt = f"Question: {aliased_q}\nAnswer:"
                    target = aliased_a
                
                # Create full text with proper format
                full_text = prompt + " " + target + tok.eos_token
                
                # DEBUG: Print first few examples
                if step < 3 and valid_examples == 0:
                    print(f"\n[DEBUG] Step {step}, Example {valid_examples}:")
                    print(f"  Question: {q}")
                    print(f"  Answer: {a}")
                    print(f"  Target: {target}")
                    print(f"  Prompt: {prompt}")
                    print(f"  Full text: {full_text}")
                
                # Tokenize
                full_ids = tok(full_text, return_tensors="pt", truncation=True, max_length=512).input_ids.to(device)
                prompt_ids = tok(prompt, return_tensors="pt", truncation=True, max_length=512).input_ids.to(device)
                
                if full_ids.size(1) <= prompt_ids.size(1):
                    continue
                
                prompt_len = prompt_ids.size(1)
                
                # Create labels
                labels = full_ids.clone()
                labels[:, :prompt_len] = -100  # Ignore prompt tokens
                
                # DEBUG: Check if we have valid labels
                valid_label_count = (labels != -100).sum().item()
                if valid_label_count == 0:
                    print(f"[DEBUG] No valid labels! prompt_len: {prompt_len}, full_len: {full_ids.size(1)}")
                    continue
                
                # DEBUG: Print first few examples
                if step < 3 and valid_examples == 0:
                    print(f"\n[DEBUG] Step {step}, Example {valid_examples}:")
                    print(f"  Question: {q}")
                    print(f"  Answer: {a}")
                    print(f"  Target: {target}")
                    print(f"  Prompt length: {prompt_len}")
                    print(f"  Full length: {full_ids.size(1)}")
                    print(f"  Valid labels: {valid_label_count}")
                    print(f"  Labels: {labels[0, prompt_len:prompt_len+5]}")  # Show first few target labels
                
                # Forward pass
                loss, _ = model(full_ids, labels=labels)
                
                # DEBUG: Check loss value
                if step < 3 and valid_examples == 0:
                    print(f"  Raw loss: {loss}")
                    print(f"  Loss item: {loss.item()}")
                
                if torch.isnan(loss):
                    nan_losses += 1
                    if nan_losses > 10:
                        print("[ERROR] Too many NaN losses, stopping training")
                        sys.exit(1)
                    continue
                
                # Accumulate loss
                total_loss += loss
                valid_examples += 1
                total_examples += 1
                
            except Exception as e:
                print(f"[ERROR] Exception in example: {e}")
                skipped_examples += 1
                continue
        
        # Backward pass
        if valid_examples > 0:
            avg_loss = total_loss / valid_examples
            
            # DEBUG: Check average loss
            if step < 3:
                print(f"[DEBUG] Step {step}: total_loss={total_loss}, valid_examples={valid_examples}, avg_loss={avg_loss}")
            
            optimizer.zero_grad()
            avg_loss.backward()
            
            # Check gradients
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if torch.isnan(grad_norm):
                print(f"[ERROR] NaN gradients at step {step}")
                continue
            
            # DEBUG: Check gradient norm
            if step < 3:
                print(f"[DEBUG] Step {step}: grad_norm={grad_norm}")
                
            optimizer.step()
            
            # Evaluation and checkpointing
            if (step + 1) % eval_interval == 0:
                print(f"\n[EVALUATION] Step {step + 1}")
                val_accuracy, val_correct, val_total = evaluate_model(val_qa, max_examples=200)
                print(f"[EVALUATION] Validation accuracy: {val_accuracy:.3f} ({val_correct}/{val_total})")
                
                # Save checkpoint if better
                is_best = val_accuracy > best_val_accuracy
                if is_best:
                    best_val_accuracy = val_accuracy
                
                save_checkpoint(step + 1, avg_loss.item(), val_accuracy, is_best)
                
                # Early stopping check
                if val_accuracy > 0.9:
                    print(f"[EARLY STOP] High validation accuracy reached: {val_accuracy:.3f}")
                    break
            
            # Regular checkpoint saving
            elif (step + 1) % save_interval == 0:
                save_checkpoint(step + 1, avg_loss.item(), best_val_accuracy, is_best=False)
            
            # Progress reporting
            if step % 50 == 0:
                elapsed = time.time() - start_time
                steps_per_sec = (step + 1 - start_step) / elapsed if elapsed > 0 else 0
                eta_seconds = (CONFIG["TRAIN_STEPS"] - step - 1) / steps_per_sec if steps_per_sec > 0 else 0
                eta_mins = eta_seconds / 60
                
                success_rate = (total_examples / (total_examples + skipped_examples)) * 100 if (total_examples + skipped_examples) > 0 else 0
                
                print(f"Step {step:5d}/{CONFIG['TRAIN_STEPS']} | Loss: {avg_loss.item():.4f} | "
                      f"Valid: {valid_examples}/{CONFIG['BATCH_SIZE']} | "
                      f"Success: {success_rate:.1f}% | Best Val: {best_val_accuracy:.3f} | "
                      f"Speed: {steps_per_sec:.1f} steps/s | ETA: {eta_mins:.1f}m")
                sys.stdout.flush()
        else:
            print(f"[WARNING] No valid examples in batch at step {step}")
            
    except Exception as e:
        print(f"[ERROR] Exception in training step {step}: {e}")
        continue

# Final evaluation and save
print(f"\n[train_reasoner] Training completed")
print(f"[STATS] Total examples: {total_examples}, Skipped: {skipped_examples}, NaN losses: {nan_losses}")

# Final validation
final_val_accuracy, final_correct, final_total = evaluate_model(val_qa, max_examples=500)
print(f"[FINAL] Validation accuracy: {final_val_accuracy:.3f} ({final_correct}/{final_total})")

# At the end of training, always save as best model
print("[train_reasoner] Saving final model as best model...")
torch.save({
    'step': CONFIG["TRAIN_STEPS"],
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': 0.0,
    'val_accuracy': final_val_accuracy,
    'config': CONFIG
}, "corpus/tiny_reasoner.pt")

print("[train_reasoner] Training complete!")

def scramble_entity_ids(question, reasoning_chain, answer):
    """Replace entity IDs with random ones to prevent memorization."""
    import re
    
    # Find all unique entity IDs in the example
    all_text = question + " " + " ".join(reasoning_chain) + " " + answer
    entity_pattern = r'([PLC]\d+)'
    entities = list(set(re.findall(entity_pattern, all_text)))
    
    # Create random replacements
    entity_mapping = {}
    for entity in entities:
        entity_type = entity[0]
        if entity_type == 'P':
            new_id = f"P{random.randint(10000, 99999)}"
        elif entity_type == 'L':
            new_id = f"L{random.randint(10000, 99999)}"
        elif entity_type == 'C':
            new_id = f"C{random.randint(10000, 99999)}"
        entity_mapping[entity] = new_id
    
    # Apply scrambling
    scrambled_question = question
    scrambled_chain = reasoning_chain.copy()
    scrambled_answer = answer
    
    for old_id, new_id in entity_mapping.items():
        scrambled_question = scrambled_question.replace(old_id, new_id)
        scrambled_chain = [sent.replace(old_id, new_id) for sent in scrambled_chain]
        scrambled_answer = scrambled_answer.replace(old_id, new_id)
    
    return scrambled_question, scrambled_chain, scrambled_answer

def prepare_training_data_with_scrambling():
    """Prepare training data with entity scrambling and masking."""
    print("[train_reasoner] Loading and preparing training data with scrambling...")
    
    with open("corpus/qa.jsonl", "r") as f:
        qa_raw = [json.loads(line.strip()) for line in f]
    
    training_data = []
    
    for item in qa_raw:
        q, a, reasoning_chain = item[0], item[1], item[2]
        
        # 50% scrambled entity IDs
        if random.random() < 0.5:
            scrambled_q, scrambled_chain, scrambled_a = scramble_entity_ids(q, reasoning_chain, a)
            prompt = build_prompt(scrambled_q, scrambled_chain, scrambled_a)
        
        # 50% generic masking
        else:
            from train_reasoner_masked import create_masked_training_example
            masked_q, masked_chain, masked_a, _ = create_masked_training_example(q, reasoning_chain, a)
            prompt = build_prompt(masked_q, masked_chain, masked_a)
        
        training_data.append(prompt)
    
    print(f"[train_reasoner] Created {len(training_data)} training examples with scrambling")
    return training_data

def create_training_example_with_aliasing(question, reasoning_chain, answer):
    """Create training example with proper entity aliasing."""
    # Combine all text to find entities
    all_texts = [question, answer] + reasoning_chain
    
    # Build aliases
    alias, inv_alias = build_alias(all_texts)
    
    # Apply aliasing
    aliased_question = apply_alias(question, alias)
    aliased_chain = [apply_alias(sent, alias) for sent in reasoning_chain]
    aliased_answer = apply_alias(answer, alias)
    
    return aliased_question, aliased_chain, aliased_answer, alias, inv_alias

def apply_aliases(text, alias_map):
    """Replace entity IDs with aliases."""
    for entity_id, alias in alias_map.items():
        text = text.replace(entity_id, alias)
    return text