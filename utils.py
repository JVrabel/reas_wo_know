import secrets
import re

# Fixed regex pattern - removed escaped backslashes
ID_RE = re.compile(r"[PLSCD]\d+")

def extract_entities(texts):
    """Extract all entity IDs from a list of texts."""
    entities = set()
    
    for text in texts:
        if isinstance(text, str):
            # Find all entity IDs matching the pattern
            matches = ID_RE.findall(text)
            entities.update(matches)
    
    return sorted(list(entities))

def build_alias(all_texts):
    """Build entity aliases for the given texts with random assignment."""
    # Find all entities
    entities = extract_entities(all_texts)
    
    # Group entities by type
    entity_groups = {}
    for entity in entities:
        entity_type = entity[0]  # P, L, S, C, D
        if entity_type not in entity_groups:
            entity_groups[entity_type] = []
        entity_groups[entity_type].append(entity)
    
    # Create random aliases for each type
    alias = {}
    inv_alias = {}
    
    for entity_type, entity_list in entity_groups.items():
        # Shuffle to randomize assignment
        import random
        shuffled_entities = entity_list.copy()
        random.shuffle(shuffled_entities)
        
        for i, entity in enumerate(shuffled_entities):
            if entity_type == 'P':
                alias_name = f"<PERSON{i+1}>"
            elif entity_type == 'L':
                alias_name = f"<LOCATION{i+1}>"
            elif entity_type == 'C':
                alias_name = f"<COMPANY{i+1}>"
            elif entity_type == 'S':
                alias_name = f"<SKILL{i+1}>"
            elif entity_type == 'D':
                alias_name = f"<DATE{i+1}>"
            else:
                alias_name = f"<ENTITY{i+1}>"
            
            alias[entity] = alias_name
            inv_alias[alias_name] = entity
    
    return alias, inv_alias

def apply_alias(text, alias):
    """Replace all real entity IDs with their aliases."""
    for real, pseudo in alias.items():
        text = text.replace(real, pseudo)
    return text

def build_prompt(q_raw, chunks, alias):
    """Build the full prompt with question and retrieved documents."""
    q = apply_alias(q_raw, alias)
    ctx = "\n".join(f"<DOC_{i+1}> " + apply_alias(c, alias) for i, c in enumerate(chunks))
    return f"<Q> {q}\n{ctx}\n<REASON>"