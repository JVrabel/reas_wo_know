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
            print(f"WARNING: Sequence too long ({len(full_tokens)} > {self.max_length}), skipping example")
            # Return a dummy example or handle gracefully
            full_tokens = full_tokens[:self.max_length-1] + [self.tokenizer.eos_token_id]
        
        # Pad to max_length
        input_ids = full_tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(full_tokens))
        input_ids = input_ids[:self.max_length]
        
        # Create labels: -100 for prompt tokens, actual tokens for target
        labels = [-100] * len(input_ids)
        
        # Find actual prompt length in the truncated sequence
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
    """Setup model with proper configuration."""
    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=CONFIG["MAX_LENGTH"],
        n_embd=CONFIG["HIDDEN_SIZE"],
        n_layer=CONFIG["NUM_LAYERS"],
        n_head=CONFIG["NUM_HEADS"],
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
    )
    
    model = GPT2LMHeadModel(config)
    model.resize_token_embeddings(len(tokenizer))
    
    return model

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """Create a learning rate scheduler with linear warmup and decay."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))
    
    from torch.optim.lr_scheduler import LambdaLR
    return LambdaLR(optimizer, lr_lambda)

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

def generate_sample_predictions(model, tokenizer, batch, device, max_new_tokens=10, num_samples=10):
    """Generate sample predictions using the aliased data."""
    model.eval()
    samples = []
    correct_predictions = 0
    
    with torch.no_grad():
        # Generate more samples (up to batch size or num_samples)
        actual_samples = min(num_samples, len(batch['qa_example']))
        
        for i in range(actual_samples):
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
            
            # Generate with proper EOS stopping
            generated = model.generate(
                prompt_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                early_stopping=True,
            )
            
            # DEBUG: Check what was actually generated
            generated_tokens = generated[0][prompt_length:]
            print(f"Generated token IDs: {generated_tokens.tolist()}")
            print(f"EOS token ID: {tokenizer.eos_token_id}")
            print(f"Contains EOS: {tokenizer.eos_token_id in generated_tokens}")
            
            # Decode and stop at EOS manually if needed
            generated_tokens = generated_tokens[:len(generated_tokens) - 1]
            
            # Find EOS token and truncate there
            if tokenizer.eos_token_id in generated_tokens:
                eos_pos = (generated_tokens == tokenizer.eos_token_id).nonzero(as_tuple=True)[0][0]
                generated_tokens = generated_tokens[:eos_pos]
            
            generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=False)
            
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
            
            print(f"=== Sample {i+1} ===")
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
                correct_predictions += 1
            
            samples.append({
                'question': qa_example['question'],
                'aliased_question': aliased_question,
                'gold_answer': qa_example['answer'],
                'expected_aliased': expected_aliased,
                'prediction': predicted_real,
                'prediction_aliased': generated_text.strip(),
                'reasoning_type': qa_example.get('reasoning_type', 'unknown'),
                'retrieved_docs': retrieved_docs,
                'full_prompt': training_example['prompt'],
                'is_correct': is_correct  # Add correctness flag
            })
    
    # Calculate accuracy
    accuracy = correct_predictions / actual_samples if actual_samples > 0 else 0.0
    
    model.train()
    return samples, accuracy

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

def train_model():
    """Main training function with improved stability."""
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
    
    # Optimizer with better settings
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=CONFIG["LEARNING_RATE"],
        weight_decay=CONFIG["WEIGHT_DECAY"],
        eps=1e-8,
        betas=(0.9, 0.95)  # Better betas for stability
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
    
    # Add learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=CONFIG["WARMUP_STEPS"],
        num_training_steps=total_steps
    )
    
    print(f"Total steps: {total_steps}, Warmup steps: {CONFIG['WARMUP_STEPS']}")
    
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
    
    # Training loop with gradient accumulation
    print("\n🚀 Starting training...")
    model.train()
    step = 0
    accumulated_loss = 0.0
    
    # Initialize training progress tracking
    training_progress = []
    
    for epoch in range(CONFIG["NUM_EPOCHS"]):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{CONFIG['NUM_EPOCHS']}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Validate and clamp token IDs
            vocab_size = len(tokenizer)
            input_ids = torch.clamp(input_ids, 0, vocab_size - 1)
            valid_label_mask = (labels != -100)
            labels = torch.where(valid_label_mask, 
                               torch.clamp(labels, 0, vocab_size - 1), 
                               labels)
            
            # Forward pass
            outputs = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
            loss = outputs.loss / CONFIG["GRADIENT_ACCUMULATION"]  # Scale loss
            
            # Backward pass
            loss.backward()
            accumulated_loss += loss.item()
            
            # Update weights every GRADIENT_ACCUMULATION steps
            if (batch_idx + 1) % CONFIG["GRADIENT_ACCUMULATION"] == 0:
                step += 1
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CONFIG["MAX_GRAD_NORM"])
                
                # Update weights
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                # Get current learning rate
                current_lr = scheduler.get_last_lr()[0]
                
                # Update progress bar with accumulated loss
                pbar.set_postfix({
                    'loss': f'{accumulated_loss:.4f}', 
                    'lr': f'{current_lr:.2e}',
                    'step': f'{step}/{total_steps//CONFIG["GRADIENT_ACCUMULATION"]}'
                })
                
                # Log progress every 50 steps (was 100)
                if step % 50 == 0:
                    print(f"\nStep {step}: Loss = {accumulated_loss:.4f}, LR = {current_lr:.2e}")
                    
                    # Save training progress
                    progress_entry = {
                        'step': step,
                        'epoch': epoch + 1,
                        'loss': accumulated_loss,
                        'learning_rate': current_lr,
                        'timestamp': datetime.now().isoformat()
                    }
                    training_progress.append(progress_entry)
                    
                    # Save progress to file
                    progress_file = os.path.join(results_dir, "training_progress.json")
                    with open(progress_file, 'w') as f:
                        json.dump(training_progress, f, indent=2)
                
                # Reset accumulated loss
                accumulated_loss = 0.0
                
                # Generate sample predictions every 100 steps (was 200)
                if step % 100 == 0:
                    print(f"\n=== Step {step} Sample Predictions ===")
                    try:
                        samples, accuracy = generate_sample_predictions(model, tokenizer, batch, device, num_samples=5)
                        
                        # Save samples to file with accuracy
                        sample_data = {
                            'step': step,
                            'accuracy': accuracy,
                            'correct_predictions': sum(1 for s in samples if s['is_correct']),
                            'total_samples': len(samples),
                            'samples': samples
                        }
                        
                        sample_file = os.path.join(results_dir, f"samples_step_{step}.json")
                        with open(sample_file, 'w') as f:
                            json.dump(sample_data, f, indent=2)
                        
                        # Print accuracy and a few samples
                        print(f"Accuracy: {accuracy:.2%} ({sum(1 for s in samples if s['is_correct'])}/{len(samples)})")
                        
                        for i, sample in enumerate(samples[:3]):  # Show first 3
                            status = "✅" if sample['is_correct'] else "❌"
                            print(f"Sample {i+1} {status}:")
                            print(f"  Q: {sample['question']}")
                            print(f"  Gold: {sample['gold_answer']}")
                            print(f"  Pred: {sample['prediction']}")
                            print(f"  Type: {sample['reasoning_type']}")
                            print()
                        
                    except Exception as e:
                        print(f"❌ Sample generation failed: {e}")
                        print("Continuing training without samples...")
            
            # Save checkpoint every epoch
            if step > 0 and step % (steps_per_epoch // CONFIG["GRADIENT_ACCUMULATION"]) == 0:
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

if __name__ == "__main__":
    train_model()