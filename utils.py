import secrets
import re
import random

# Fixed regex pattern for entity extraction
ID_RE = re.compile(r"[PCLSTD]\d+|PJ\d+")

def extract_entities(texts):
    """Extract all entity IDs from a list of texts."""
    entities = set()
    
    for text in texts:
        if isinstance(text, str):
            # Find all entity IDs matching the pattern
            matches = ID_RE.findall(text)
            entities.update(matches)
    
    return sorted(list(entities))

def build_alias(texts):
    """Build randomized but consistent aliases within each example."""
    entities = extract_entities(texts)
    alias = {}
    
    # Group entities by type
    by_type = {"PERSON": [], "COMPANY": [], "LOCATION": [], "PROJECT": [], 
               "SKILL": [], "TECHNOLOGY": [], "DATE": []}
    
    for entity in entities:
        if entity.startswith('P') and not entity.startswith('PJ'):
            by_type["PERSON"].append(entity)
        elif entity.startswith('C'):
            by_type["COMPANY"].append(entity)
        elif entity.startswith('L'):
            by_type["LOCATION"].append(entity)
        elif entity.startswith('PJ'):
            by_type["PROJECT"].append(entity)
        elif entity.startswith('S'):
            by_type["SKILL"].append(entity)
        elif entity.startswith('T'):
            by_type["TECHNOLOGY"].append(entity)
        elif entity.startswith('D'):
            by_type["DATE"].append(entity)
    
    # For each type, create randomized but consistent mapping
    for entity_type, entity_list in by_type.items():
        if not entity_list:
            continue
            
        # Generate random alias numbers from small range (1-20)
        num_entities = len(entity_list)
        alias_numbers = random.sample(range(1, 21), min(num_entities, 20))
        
        # If we need more than 20, extend the range
        if num_entities > 20:
            extra_numbers = random.sample(range(21, 101), num_entities - 20)
            alias_numbers.extend(extra_numbers)
        
        # Assign consistently within this example
        for i, entity in enumerate(entity_list):
            alias_token = f"<{entity_type}{alias_numbers[i]:02d}>"
            alias[entity] = alias_token
    
    return alias  # ← Only return alias map, no inverse

def apply_alias(text, alias_map):
    """Apply alias mapping to text, replacing entities with aliases."""
    if not isinstance(text, str):
        return text
    
    result = text
    # Sort by length (longest first) to avoid partial replacements
    for entity in sorted(alias_map.keys(), key=len, reverse=True):
        if entity in result:
            result = result.replace(entity, alias_map[entity])
    
    return result

def build_prompt(question, retrieved_docs, alias_map):
    """Build training prompt with aliased question and documents."""
    # Apply aliasing to question and documents
    aliased_question = apply_alias(question, alias_map)
    aliased_docs = [apply_alias(doc, alias_map) for doc in retrieved_docs]
    
    # Build prompt
    prompt_parts = [f"<Q> {aliased_question}"]
    
    for i, doc in enumerate(aliased_docs, 1):
        prompt_parts.append(f"<DOC_{i}> {doc}")
    
    prompt_parts.append("<REASON>")
    
    return "\n".join(prompt_parts)

def prepare_training_example(qa_example, retrieved_docs):
    """Prepare a complete training example with aliasing."""
    # Extract all text for consistent aliasing
    all_texts = [qa_example['question'], qa_example['answer']] + retrieved_docs
    all_texts.extend(qa_example.get('facts', []))
    
    # Build alias mapping for this example (local scope only)
    alias_map = build_alias(all_texts)
    
    # Build prompt
    prompt = build_prompt(qa_example['question'], retrieved_docs, alias_map)
    
    # Apply aliasing to target answer
    target = apply_alias(qa_example['answer'], alias_map)
    
    return {
        'prompt': prompt,
        'target': target,
        'alias_map': alias_map,  # Only for debugging this example
        'original_question': qa_example['question'],
        'original_answer': qa_example['answer']
    }