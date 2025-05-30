import json
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from torch.utils.data import DataLoader, Dataset
import os
from tqdm import tqdm
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
        
        # The full sequence should be: prompt + reasoning_marker + target
        full_sequence = prompt + '\n' + target
        
        # Tokenize the full sequence
        full_tokens = self.tokenizer.encode(full_sequence, add_special_tokens=False)
        
        # Find where the target starts by tokenizing prompt separately
        prompt_tokens = self.tokenizer.encode(prompt + '\n', add_special_tokens=False)
        prompt_length = len(prompt_tokens)
        
        # CRITICAL: Filter out invalid token IDs
        vocab_size = len(self.tokenizer)
        full_tokens = [t for t in full_tokens if 0 <= t < vocab_size]
        
        # Ensure we have some target tokens
        if len(full_tokens) <= prompt_length:
            print(f"WARNING: No target tokens for question: {qa_example['question'][:50]}...")
            # Add a safe fallback target
            target_fallback = self.tokenizer.encode(" unknown", add_special_tokens=False)
            target_fallback = [t for t in target_fallback if 0 <= t < vocab_size]
            full_tokens.extend(target_fallback)
        
        # Truncate if too long, but preserve some target tokens
        if len(full_tokens) > self.max_length:
            # Keep at least 10 tokens for the target
            max_prompt_length = self.max_length - 10
            if prompt_length > max_prompt_length:
                # Truncate prompt but keep the question part
                question_tokens = self.tokenizer.encode(training_example['prompt'].split('<DOC_')[0], add_special_tokens=False)
                question_tokens = [t for t in question_tokens if 0 <= t < vocab_size]
                
                # Keep question + some docs + target
                available_for_docs = max_prompt_length - len(question_tokens) - 20  # Reserve space
                if available_for_docs > 0:
                    doc_tokens = full_tokens[len(question_tokens):prompt_length][:available_for_docs]
                    reasoning_and_target = full_tokens[prompt_length:][:10]  # Keep some target
                    full_tokens = question_tokens + doc_tokens + reasoning_and_target
                else:
                    # Emergency: just question + target
                    target_tokens = full_tokens[prompt_length:][:10]
                    full_tokens = question_tokens[:max_prompt_length] + target_tokens
                
                # Recalculate prompt length
                full_text = self.tokenizer.decode(full_tokens)
                if '\n' + target in full_text:
                    target_start_text = full_text.split('\n' + target)[0] + '\n'
                    prompt_length = len(self.tokenizer.encode(target_start_text, add_special_tokens=False))
                else:
                    # Fallback: assume last 10 tokens are target
                    prompt_length = len(full_tokens) - 10
            else:
                full_tokens = full_tokens[:self.max_length]
        
        # Create input_ids and labels
        input_ids = full_tokens
        labels = [-100] * prompt_length + full_tokens[prompt_length:]  # Only learn to predict target
        
        # Ensure labels has same length as input_ids
        if len(labels) > len(input_ids):
            labels = labels[:len(input_ids)]
        elif len(labels) < len(input_ids):
            labels.extend([-100] * (len(input_ids) - len(labels)))
        
        # Pad to max_length
        attention_mask = [1] * len(input_ids)
        
        while len(input_ids) < self.max_length:
            input_ids.append(self.tokenizer.pad_token_id)
            labels.append(-100)
            attention_mask.append(0)
        
        # Convert to tensors with final safety check
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        
        # Clamp any invalid token IDs to valid range
        input_ids = torch.clamp(input_ids, 0, vocab_size - 1)
        labels = torch.where((labels >= 0) & (labels < vocab_size), labels, torch.tensor(-100))
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
            'qa_example': qa_example,
            'training_example': training_example
        }

def setup_tokenizer():
    """Setup tokenizer with special tokens."""
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # Add special tokens for aliasing - these will be the actual targets
    entity_tokens = []
    
    # Create pools of entity tokens that can be scrambled
    for entity_type in ["PERSON", "COMPANY", "LOCATION", "SKILL", "PROJECT", "TECHNOLOGY", "DATE"]:
        for i in range(1, 21):  # 20 tokens per type
            entity_tokens.append(f"<{entity_type}{i:02d}>")  # <PERSON01>, <PERSON02>, etc.
    
    # Add document markers
    doc_tokens = [f'<DOC_{i}>' for i in range(1, 11)]
    
    # Add reasoning markers
    special_markers = ["<Q>", "<REASON>", "<A>", "<PAD>"]
    
    all_special_tokens = entity_tokens + doc_tokens + special_markers
    
    # Add tokens to tokenizer
    tokenizer.add_special_tokens({
        'additional_special_tokens': all_special_tokens,
        'pad_token': '<PAD>'
    })
    
    print(f"Added {len(all_special_tokens)} special tokens")
    
    # Verify a few tokens are single tokens
    test_tokens = ["<PERSON01>", "<COMPANY01>", "<LOCATION01>"]
    for token in test_tokens:
        encoded = tokenizer.encode(token, add_special_tokens=False)
        print(f"Token '{token}' -> {encoded} (length: {len(encoded)})")
    
    return tokenizer

def debug_sample(sample, tokenizer, sample_idx):
    """Debug a single sample in detail."""
    print(f"\n{'='*80}")
    print(f"SAMPLE {sample_idx}")
    print(f"{'='*80}")
    
    qa_example = sample['qa_example']
    training_example = sample['training_example']
    
    print(f"📝 ORIGINAL DATA:")
    print(f"   Question: {qa_example['question']}")
    print(f"   Answer: {qa_example['answer']}")
    print(f"   Reasoning type: {qa_example.get('reasoning_type', 'unknown')}")
    
    print(f"\n🔄 ALIASING:")
    alias_map = training_example.get('alias_map', {})
    print(f"   Total aliases: {len(alias_map)}")
    
    # Show relevant aliases for this question
    relevant_aliases = {}
    for original, alias in alias_map.items():
        if original in qa_example['question'] or original == qa_example['answer']:
            relevant_aliases[original] = alias
    
    print(f"   Relevant aliases:")
    for orig, alias in relevant_aliases.items():
        print(f"     {orig} → {alias}")
    
    print(f"\n📄 ALIASED CONTENT:")
    print(f"   Aliased question: {training_example['prompt'].split('<DOC_')[0].strip()}")
    print(f"   Aliased target: {training_example['target']}")
    
    print(f"\n📚 RETRIEVED DOCUMENTS:")
    prompt_lines = training_example['prompt'].split('\n')
    doc_lines = [line for line in prompt_lines if line.startswith('<DOC_')]
    for i, doc_line in enumerate(doc_lines[:3]):  # Show first 3 docs
        print(f"   Doc {i+1}: {doc_line[:100]}...")
    
    print(f"\n🔢 TOKENIZATION:")
    input_ids = sample['input_ids']
    labels = sample['labels']
    attention_mask = sample['attention_mask']
    
    print(f"   Input shape: {input_ids.shape}")
    print(f"   Labels shape: {labels.shape}")
    print(f"   Attention mask shape: {attention_mask.shape}")
    
    # Find valid labels (target tokens)
    valid_label_indices = torch.where(labels != -100)[0]
    print(f"   Valid labels count: {len(valid_label_indices)}")
    
    if len(valid_label_indices) > 0:
        first_label_idx = valid_label_indices[0].item()
        last_label_idx = valid_label_indices[-1].item()
        
        print(f"   Target starts at position: {first_label_idx}")
        print(f"   Target ends at position: {last_label_idx}")
        
        # Decode different parts
        full_decoded = tokenizer.decode(input_ids, skip_special_tokens=False)
        prompt_part = tokenizer.decode(input_ids[:first_label_idx], skip_special_tokens=False)
        target_part = tokenizer.decode(input_ids[first_label_idx:last_label_idx+1], skip_special_tokens=False)
        
        print(f"\n📖 DECODED CONTENT:")
        print(f"   Full sequence length: {len(full_decoded)} chars")
        print(f"   Prompt part (last 100 chars): ...{prompt_part[-100:]}")
        print(f"   Target part: '{target_part}'")
        
        # Verify target correctness
        expected_target = training_example['target'].strip()
        if expected_target in target_part:
            print(f"   ✅ Target verification: PASS")
        else:
            print(f"   ❌ Target verification: FAIL")
            print(f"      Expected: '{expected_target}'")
            print(f"      Found: '{target_part}'")
    else:
        print(f"   ❌ NO VALID LABELS FOUND!")
        print(f"   First 10 labels: {labels[:10].tolist()}")
        print(f"   Last 10 labels: {labels[-10:].tolist()}")
    
    # Check for padding
    non_pad_tokens = torch.sum(attention_mask).item()
    print(f"   Non-padding tokens: {non_pad_tokens}/{len(input_ids)}")
    
    # Vocabulary validation
    vocab_size = len(tokenizer)
    invalid_input_ids = torch.where(input_ids >= vocab_size)[0]
    invalid_label_ids = torch.where((labels >= vocab_size) & (labels != -100))[0]
    
    if len(invalid_input_ids) > 0:
        print(f"   ❌ Invalid input token IDs: {len(invalid_input_ids)}")
    if len(invalid_label_ids) > 0:
        print(f"   ❌ Invalid label token IDs: {len(invalid_label_ids)}")
    
    if len(invalid_input_ids) == 0 and len(invalid_label_ids) == 0:
        print(f"   ✅ All token IDs valid")

def debug_batch(batch, tokenizer, batch_idx):
    """Debug an entire batch."""
    print(f"\n{'#'*100}")
    print(f"BATCH {batch_idx}")
    print(f"{'#'*100}")
    
    batch_size = len(batch['input_ids'])
    print(f"Batch size: {batch_size}")
    
    # Overall batch statistics
    input_ids = batch['input_ids']
    labels = batch['labels']
    attention_mask = batch['attention_mask']
    
    print(f"Batch shapes: {input_ids.shape}")
    
    # Count valid labels across batch
    total_valid_labels = torch.sum(labels != -100).item()
    total_tokens = labels.numel()
    print(f"Valid labels: {total_valid_labels}/{total_tokens} ({100*total_valid_labels/total_tokens:.1f}%)")
    
    # Check for any invalid token IDs
    vocab_size = len(tokenizer)
    invalid_inputs = torch.sum(input_ids >= vocab_size).item()
    invalid_labels = torch.sum((labels >= vocab_size) & (labels != -100)).item()
    
    if invalid_inputs > 0 or invalid_labels > 0:
        print(f"❌ Invalid tokens found: {invalid_inputs} inputs, {invalid_labels} labels")
    else:
        print(f"✅ All tokens valid")
    
    # Debug each sample in the batch
    for i in range(batch_size):
        sample = {
            'input_ids': input_ids[i],
            'labels': labels[i],
            'attention_mask': attention_mask[i],
            'qa_example': batch['qa_example'][i],
            'training_example': batch['training_example'][i]
        }
        debug_sample(sample, tokenizer, f"{batch_idx}.{i}")

def create_batch_alias_map(batch_qa_examples):
    """Create a scrambled alias map for this batch."""
    import re
    import random
    
    # Collect all entities that appear in this batch
    batch_entities = set()
    
    for qa in batch_qa_examples:
        # Extract entities from question and answer using regex
        question_entities = re.findall(r'[PCLSTD]\d+', qa['question'])
        answer_entities = re.findall(r'[PCLSTD]\d+', qa['answer'])
        
        batch_entities.update(question_entities)
        batch_entities.update(answer_entities)
        
        # Also add the answer directly
        if qa['answer'].startswith(('P', 'C', 'L', 'S', 'T', 'D')):
            batch_entities.add(qa['answer'])
    
    print(f"Found {len(batch_entities)} entities in batch: {list(batch_entities)[:10]}...")
    
    # Create random mapping for this batch
    alias_map = {}
    used_aliases = set()
    
    for entity in batch_entities:
        # Determine entity type based on prefix
        if entity.startswith('P'):
            entity_type = "PERSON"
        elif entity.startswith('C'):
            entity_type = "COMPANY"
        elif entity.startswith('L'):
            entity_type = "LOCATION"
        elif entity.startswith('S'):
            entity_type = "SKILL"
        elif entity.startswith('T'):
            entity_type = "TECHNOLOGY"
        elif entity.startswith('D'):
            entity_type = "DATE"
        else:
            entity_type = "PROJECT"  # fallback
        
        # Find an unused alias of the right type
        for i in range(1, 21):
            alias = f"<{entity_type}{i:02d}>"  # <PERSON01>, <COMPANY05>, etc.
            if alias not in used_aliases:
                alias_map[entity] = alias
                used_aliases.add(alias)
                break
    
    return alias_map

def apply_batch_aliasing(batch_training_examples, alias_map):
    """Apply aliasing to a batch of training examples."""
    aliased_batch = []
    
    for training_example in batch_training_examples:
        qa_example = training_example['qa_example']
        
        # Apply aliasing to prompt and target
        aliased_prompt = training_example['prompt']
        aliased_target = qa_example['answer']  # Start with original answer
        
        # Apply all aliases
        for original, alias in alias_map.items():
            aliased_prompt = aliased_prompt.replace(original, alias)
            if aliased_target == original:
                aliased_target = alias
        
        aliased_example = {
            'prompt': aliased_prompt,
            'target': aliased_target,  # This should now be like <COMPANY14>
            'alias_map': alias_map,
            'qa_example': qa_example
        }
        aliased_batch.append(aliased_example)
    
    return aliased_batch

def tokenize_example(aliased_example, tokenizer):
    """Tokenize a single aliased example."""
    prompt = aliased_example['prompt'].rstrip()
    target = aliased_example['target'].strip()
    
    # Verify target is a single special token
    if not (target.startswith('<') and target.endswith('>')):
        print(f"ERROR: Target '{target}' is not a special token!")
        return None
    
    # Tokenize prompt and target separately
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    target_tokens = tokenizer.encode(target, add_special_tokens=False)
    
    # Verify target is exactly one token
    if len(target_tokens) != 1:
        print(f"ERROR: Target '{target}' should be 1 token but got {len(target_tokens)}: {target_tokens}")
        return None
    
    # Combine with space
    space_token = tokenizer.encode(" ", add_special_tokens=False)[0]
    input_ids = prompt_tokens + [space_token] + target_tokens
    labels = [-100] * (len(prompt_tokens) + 1) + target_tokens
    
    # Truncate if needed
    if len(input_ids) > 512:
        # Keep last part of prompt + target
        keep_prompt = 512 - 2  # Reserve space for space + target
        input_ids = prompt_tokens[-keep_prompt:] + [space_token] + target_tokens
        labels = [-100] * (keep_prompt + 1) + target_tokens
    
    # Pad
    attention_mask = [1] * len(input_ids)
    while len(input_ids) < 512:
        input_ids.append(tokenizer.pad_token_id)
        labels.append(-100)
        attention_mask.append(0)
    
    return {
        'input_ids': torch.tensor(input_ids, dtype=torch.long),
        'labels': torch.tensor(labels, dtype=torch.long),
        'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
        'qa_example': aliased_example['qa_example'],
        'training_example': aliased_example
    }

def collate_fn(batch_training_examples):
    """Custom collate function that applies batch-level aliasing."""
    # Extract QA examples for alias map creation
    qa_examples = [ex['qa_example'] for ex in batch_training_examples]
    
    # Create batch-specific alias map
    alias_map = create_batch_alias_map(qa_examples)
    
    # Apply aliasing to the batch
    aliased_batch = apply_batch_aliasing(batch_training_examples, alias_map)
    
    # Now tokenize each example
    tokenized_batch = []
    for aliased_example in aliased_batch:
        tokenized_example = tokenize_example(aliased_example, tokenizer)
        tokenized_batch.append(tokenized_example)
    
    # Stack tensors
    return {
        'input_ids': torch.stack([item['input_ids'] for item in tokenized_batch]),
        'labels': torch.stack([item['labels'] for item in tokenized_batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in tokenized_batch]),
        'qa_example': [item['qa_example'] for item in tokenized_batch],
        'training_example': [item['training_example'] for item in tokenized_batch]
    }

def main():
    print("[train_debug] Setting up debug session...")
    
    # Load data
    print("Loading data...")
    with open('corpus/qa_train.jsonl', 'r') as f:
        qa_data = [json.loads(line) for line in f]
    
    documents = load_documents('corpus/docs_train.jsonl')
    print(f"Loaded {len(qa_data)} QA examples, {len(documents)} documents")
    
    # Setup tokenizer
    print("Setting up tokenizer...")
    tokenizer = setup_tokenizer()
    print(f"Vocabulary: {len(tokenizer)} tokens")
    
    # Setup retriever
    print("Setting up retriever...")
    retriever = OracleRetriever(documents)
    
    # Create dataset with small subset
    debug_data = qa_data[:10]  # Just first 10 examples
    dataset = ReasoningDataset(debug_data, retriever, tokenizer, max_length=CONFIG["MAX_LENGTH"])
    
    # Create dataloader with batch size 2
    dataloader = DataLoader(
        dataset, 
        batch_size=2, 
        shuffle=False,  # Don't shuffle for debugging
        collate_fn=lambda x: {
            'input_ids': torch.stack([item['input_ids'] for item in x]),
            'labels': torch.stack([item['labels'] for item in x]),
            'attention_mask': torch.stack([item['attention_mask'] for item in x]),
            'qa_example': [item['qa_example'] for item in x],
            'training_example': [item['training_example'] for item in x]
        }
    )
    
    print(f"\nDebugging {len(debug_data)} samples in batches of 2...")
    print(f"Will process {len(dataloader)} batches")
    
    # Debug first few batches
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= 3:  # Only debug first 3 batches
            break
            
        debug_batch(batch, tokenizer, batch_idx)
        
        # Ask user if they want to continue
        if batch_idx < 2:
            response = input(f"\nPress Enter to continue to next batch, or 'q' to quit: ")
            if response.lower() == 'q':
                break
    
    print(f"\n🎉 Debug session complete!")

if __name__ == "__main__":
    main()