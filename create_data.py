import random
import json
import os
from collections import defaultdict

def generate_entities():
    """Generate entity IDs for train and test sets"""
    train_entities = {
        'persons': [f'P{i:04d}' for i in range(2100)],
        'companies': [f'C{i:04d}' for i in range(350)],
        'locations': [f'L{i:04d}' for i in range(350)],
        'projects': [f'PJ{i:03d}' for i in range(500)],
        'skills': [f'S{i:03d}' for i in range(105)],
        'technologies': [f'T{i:03d}' for i in range(140)],
        'dates': [f'D{i:03d}' for i in range(260)]
    }
    
    test_entities = {
        'persons': [f'P{i:04d}' for i in range(2100, 3000)],
        'companies': [f'C{i:04d}' for i in range(350, 500)],
        'locations': [f'L{i:04d}' for i in range(350, 500)],
        'projects': [f'PJ{i:03d}' for i in range(500, 800)],
        'skills': [f'S{i:03d}' for i in range(105, 150)],
        'technologies': [f'T{i:03d}' for i in range(140, 200)],
        'dates': [f'D{i:03d}' for i in range(260, 370)]
    }
    
    return train_entities, test_entities

def validate_no_shortcuts(chain, all_facts):
    """Ensure a reasoning chain has no shorter paths to the answer"""
    question_type = chain['reasoning_type']
    
    if question_type == '3-hop':
        return validate_3hop_no_shortcuts(chain, all_facts)
    elif question_type == '2-hop':
        return validate_2hop_no_shortcuts(chain, all_facts)
    
    return True  # 1-hop questions can't have shortcuts

def validate_3hop_no_shortcuts(chain, all_facts):
    """Validate 3-hop question has no 1-hop or 2-hop shortcuts"""
    
    if "What skill is needed for" in chain['question'] and "'s company's project" in chain['question']:
        # Extract person from "What skill is needed for P001's company's project?"
        person = chain['question'].split("What skill is needed for ")[1].split("'s company's project?")[0]
        answer = chain['answer']
        
        # Find the project from the chain facts
        project = None
        for fact in chain['facts']:
            if "requires skill" in fact:
                project = fact.split(" requires skill")[0]
                break
        
        # Check for 1-hop shortcut: person directly has the skill
        for fact in all_facts:
            if (f"{person} is skilled in {answer}" in fact or 
                f"{person} has skill {answer}" in fact):
                return False  # Found 1-hop shortcut!
        
        # Check for 2-hop shortcut: person works on project directly
        if project:
            for fact in all_facts:
                if (f"{person} works on {project}" in fact or 
                    f"{person} contributes to {project}" in fact):
                    return False  # Found 2-hop shortcut!
    
    elif "Where is" in chain['question'] and "'s company's project located" in chain['question']:
        person = chain['question'].split("Where is ")[1].split("'s company's project located?")[0]
        answer = chain['answer']
        
        # Find the project from the chain facts
        project = None
        for fact in chain['facts']:
            if "is located in" in fact:
                project = fact.split(" is located in")[0]
                break
        
        # Check for 1-hop shortcut: person lives in that location
        for fact in all_facts:
            if (f"{person} lives in {answer}" in fact or 
                f"{person} resides in {answer}" in fact or
                f"{answer} is home to {person}" in fact):
                return False
        
        # Check for 2-hop shortcut: person works on project directly
        if project:
            for fact in all_facts:
                if (f"{person} works on {project}" in fact or
                    f"{person} contributes to {project}" in fact):
                    return False
    
    return True

def validate_2hop_no_shortcuts(chain, all_facts):
    """Validate 2-hop question has no 1-hop shortcuts"""
    
    if "Where is" in chain['question'] and "'s company located" in chain['question']:
        person = chain['question'].split("Where is ")[1].split("'s company located?")[0]
        answer = chain['answer']
        
        # Check for 1-hop shortcut: person lives in company's location
        for fact in all_facts:
            if (f"{person} lives in {answer}" in fact or 
                f"{person} resides in {answer}" in fact or
                f"{answer} is home to {person}" in fact):
                return False
    
    elif "Where is" in chain['question'] and "'s project located" in chain['question']:
        person = chain['question'].split("Where is ")[1].split("'s project located?")[0]
        answer = chain['answer']
        
        # Check for 1-hop shortcut: person lives in project's location
        for fact in all_facts:
            if (f"{person} lives in {answer}" in fact or 
                f"{person} resides in {answer}" in fact or
                f"{answer} is home to {person}" in fact):
                return False
    
    elif "What skill is needed for" in chain['question'] and "'s project" in chain['question']:
        company = chain['question'].split("What skill is needed for ")[1].split("'s project?")[0]
        answer = chain['answer']
        
        # This is already a 2-hop question, no obvious 1-hop shortcuts to check
        pass
    
    return True

def generate_reasoning_chains(entities, num_chains=500):
    """Generate reasoning chains with shortcut validation"""
    chains = []
    all_generated_facts = []  # Track all facts we generate
    max_attempts = num_chains * 3  # Allow multiple attempts to find valid chains
    attempts = 0
    
    while len(chains) < num_chains and attempts < max_attempts:
        attempts += 1
        
        # Determine chain type
        if len(chains) < num_chains // 3:
            chain_type = '1-hop'
        elif len(chains) < 2 * num_chains // 3:
            chain_type = '2-hop'
        else:
            chain_type = '3-hop'
        
        # Generate potential chain
        potential_chain = None
        
        if chain_type == '1-hop':
            question_type = random.choice(['where_live', 'where_work', 'what_skill'])
            
            if question_type == 'where_live':
                person = random.choice(entities['persons'])
                location = random.choice(entities['locations'])
                
                potential_chain = {
                    'question': f"Where does {person} live?",
                    'facts': [f"{person} lives in {location}"],
                    'answer': location,
                    'reasoning_type': '1-hop'
                }
                
            elif question_type == 'where_work':
                person = random.choice(entities['persons'])
                company = random.choice(entities['companies'])
                
                potential_chain = {
                    'question': f"Where does {person} work?",
                    'facts': [f"{person} works at {company}"],
                    'answer': company,
                    'reasoning_type': '1-hop'
                }
                
            elif question_type == 'what_skill':
                person = random.choice(entities['persons'])
                skill = random.choice(entities['skills'])
                
                potential_chain = {
                    'question': f"What skill does {person} have?",
                    'facts': [f"{person} is skilled in {skill}"],
                    'answer': skill,
                    'reasoning_type': '1-hop'
                }
        
        elif chain_type == '2-hop':
            question_type = random.choice(['company_location', 'project_location', 'company_skill'])
            
            if question_type == 'company_location':
                person = random.choice(entities['persons'])
                company = random.choice(entities['companies'])
                location = random.choice(entities['locations'])
                
                potential_chain = {
                    'question': f"Where is {person}'s company located?",
                    'facts': [
                        f"{person} works at {company}",
                        f"{company} is headquartered in {location}"
                    ],
                    'answer': location,
                    'reasoning_type': '2-hop'
                }
                
            elif question_type == 'project_location':
                person = random.choice(entities['persons'])
                project = random.choice(entities['projects'])
                location = random.choice(entities['locations'])
                
                potential_chain = {
                    'question': f"Where is {person}'s project located?",
                    'facts': [
                        f"{person} works on {project}",
                        f"{project} is located in {location}"
                    ],
                    'answer': location,
                    'reasoning_type': '2-hop'
                }
                
            elif question_type == 'company_skill':
                company = random.choice(entities['companies'])
                project = random.choice(entities['projects'])
                skill = random.choice(entities['skills'])
                
                potential_chain = {
                    'question': f"What skill is needed for {company}'s project?",
                    'facts': [
                        f"{company} manages {project}",
                        f"{project} requires skill {skill}"
                    ],
                    'answer': skill,
                    'reasoning_type': '2-hop'
                }
        
        elif chain_type == '3-hop':
            person = random.choice(entities['persons'])
            company = random.choice(entities['companies'])
            project = random.choice(entities['projects'])
            
            question_type = random.choice(['project_skill', 'project_location'])
            
            if question_type == 'project_skill':
                skill = random.choice(entities['skills'])
                
                potential_chain = {
                    'question': f"What skill is needed for {person}'s company's project?",
                    'facts': [
                        f"{person} works at {company}",
                        f"{company} manages {project}",
                        f"{project} requires skill {skill}"
                    ],
                    'answer': skill,
                    'reasoning_type': '3-hop'
                }
                
            elif question_type == 'project_location':
                location = random.choice(entities['locations'])
                
                potential_chain = {
                    'question': f"Where is {person}'s company's project located?",
                    'facts': [
                        f"{person} works at {company}",
                        f"{company} manages {project}",
                        f"{project} is located in {location}"
                    ],
                    'answer': location,
                    'reasoning_type': '3-hop'
                }
        
        # Validate the potential chain
        if potential_chain and validate_no_shortcuts(potential_chain, all_generated_facts):
            chains.append(potential_chain)
            # Add this chain's facts to our tracking
            all_generated_facts.extend(potential_chain['facts'])
    
    print(f"Generated {len(chains)} valid chains out of {attempts} attempts")
    return chains

def generate_filler_facts(entities, existing_facts, num_facts=1000):
    """Generate filler facts that don't contradict existing facts"""
    
    # Parse existing facts to track used entities and their relationships
    used_entities = defaultdict(dict)  # entity -> {relation_type: value}
    
    for fact in existing_facts:
        if " lives in " in fact:
            person, location = fact.split(" lives in ")
            used_entities[person]['lives_in'] = location
        elif " works at " in fact:
            person, company = fact.split(" works at ")
            used_entities[person]['works_at'] = company
        elif " is headquartered in " in fact:
            company, location = fact.split(" is headquartered in ")
            used_entities[company]['headquartered_in'] = location
        elif " manages " in fact:
            company, project = fact.split(" manages ")
            used_entities[company]['manages'] = project
        elif " works on " in fact:
            person, project = fact.split(" works on ")
            used_entities[person]['works_on'] = project
        elif " is located in " in fact:
            project, location = fact.split(" is located in ")
            used_entities[project]['located_in'] = location
        elif " requires skill " in fact:
            project, skill = fact.split(" requires skill ")
            used_entities[project]['requires_skill'] = skill
        elif " is skilled in " in fact:
            person, skill = fact.split(" is skilled in ")
            used_entities[person]['skilled_in'] = skill
    
    filler_facts = []
    
    # Generate safe filler facts using unused entities
    fact_templates = [
        # Person facts (only for unused persons)
        lambda: generate_person_fact(entities, used_entities),
        # Company facts (only for unused companies)  
        lambda: generate_company_fact(entities, used_entities),
        # Project facts (only for unused projects)
        lambda: generate_project_fact(entities, used_entities),
        # Technology dependency facts (safe - no conflicts)
        lambda: f"{random.choice(entities['technologies'])} depends on {random.choice(entities['technologies'])}",
        # Date facts (safe - no conflicts)
        lambda: f"{random.choice(entities['projects'])} started on {random.choice(entities['dates'])}",
    ]
    
    attempts = 0
    while len(filler_facts) < num_facts and attempts < num_facts * 3:
        attempts += 1
        try:
            template = random.choice(fact_templates)
            fact = template()
            if fact and fact not in filler_facts:
                filler_facts.append(fact)
        except:
            continue  # Skip if template fails
    
    return filler_facts

def generate_person_fact(entities, used_entities):
    """Generate a fact about an unused person"""
    # Find persons not used in reasoning chains
    unused_persons = [p for p in entities['persons'] if p not in used_entities]
    if not unused_persons:
        return None
    
    person = random.choice(unused_persons)
    fact_type = random.choice(['lives_in', 'works_at', 'skilled_in'])
    
    if fact_type == 'lives_in':
        location = random.choice(entities['locations'])
        return f"{person} lives in {location}"
    elif fact_type == 'works_at':
        company = random.choice(entities['companies'])
        return f"{person} works at {company}"
    elif fact_type == 'skilled_in':
        skill = random.choice(entities['skills'])
        return f"{person} is skilled in {skill}"

def generate_company_fact(entities, used_entities):
    """Generate a fact about an unused company"""
    unused_companies = [c for c in entities['companies'] if c not in used_entities]
    if not unused_companies:
        return None
    
    company = random.choice(unused_companies)
    fact_type = random.choice(['headquartered_in', 'manages'])
    
    if fact_type == 'headquartered_in':
        location = random.choice(entities['locations'])
        return f"{company} is headquartered in {location}"
    elif fact_type == 'manages':
        project = random.choice(entities['projects'])
        return f"{company} manages {project}"

def generate_project_fact(entities, used_entities):
    """Generate a fact about an unused project"""
    unused_projects = [p for p in entities['projects'] if p not in used_entities]
    if not unused_projects:
        return None
    
    project = random.choice(unused_projects)
    fact_type = random.choice(['located_in', 'requires_skill'])
    
    if fact_type == 'located_in':
        location = random.choice(entities['locations'])
        return f"{project} is located in {location}"
    elif fact_type == 'requires_skill':
        skill = random.choice(entities['skills'])
        return f"{project} requires skill {skill}"

def create_documents_from_chains(chains, filler_facts, docs_per_split=1000, max_facts_per_doc=10):
    """Create small focused documents with max 10 facts each"""
    documents = []
    
    # Extract all required facts from chains
    required_facts = []
    for chain in chains:
        required_facts.extend(chain['facts'])
    
    # Combine required facts with filler facts
    all_facts = required_facts + filler_facts
    random.shuffle(all_facts)
    
    # Create documents with at most max_facts_per_doc facts each
    current_doc_facts = []
    
    for fact in all_facts:
        current_doc_facts.append(fact)
        
        # If we reach max facts per doc, create document
        if len(current_doc_facts) >= max_facts_per_doc:
            document = ". ".join(current_doc_facts) + "."
            documents.append(document)
            current_doc_facts = []
    
    # Add remaining facts as final document
    if current_doc_facts:
        document = ". ".join(current_doc_facts) + "."
        documents.append(document)
    
    return documents

def save_data(train_chains, test_chains, train_docs, test_docs):
    """Save all generated data"""
    os.makedirs('corpus', exist_ok=True)
    
    # Save QA pairs
    with open('corpus/qa_train.jsonl', 'w') as f:
        for chain in train_chains:
            f.write(json.dumps(chain) + '\n')
    
    with open('corpus/qa_test.jsonl', 'w') as f:
        for chain in test_chains:
            f.write(json.dumps(chain) + '\n')
    
    # Save documents
    with open('corpus/docs_train.jsonl', 'w') as f:
        for doc in train_docs:
            f.write(json.dumps(doc) + '\n')
    
    with open('corpus/docs_test.jsonl', 'w') as f:
        for doc in test_docs:
            f.write(json.dumps(doc) + '\n')

def main():
    print("Generating entities...")
    train_entities, test_entities = generate_entities()
    
    print("Generating reasoning chains...")
    train_chains = generate_reasoning_chains(train_entities, num_chains=40000)
    test_chains = generate_reasoning_chains(test_entities, num_chains=20000)
    
    # Extract facts from chains for consistency checking
    train_chain_facts = []
    for chain in train_chains:
        train_chain_facts.extend(chain['facts'])
    
    test_chain_facts = []
    for chain in test_chains:
        test_chain_facts.extend(chain['facts'])
    
    print("Generating filler facts...")
    train_filler = generate_filler_facts(train_entities, train_chain_facts, num_facts=100000)
    test_filler = generate_filler_facts(test_entities, test_chain_facts, num_facts=50000)
    
    print("Creating documents...")
    train_docs = create_documents_from_chains(train_chains, train_filler, max_facts_per_doc=10)
    test_docs = create_documents_from_chains(test_chains, test_filler, max_facts_per_doc=10)
    
    print("Saving data...")
    save_data(train_chains, test_chains, train_docs, test_docs)
    
    print(f"Generated {len(train_chains)} training QA pairs")
    print(f"Generated {len(test_chains)} test QA pairs") 
    print(f"Generated {len(train_docs)} training documents")
    print(f"Generated {len(test_docs)} test documents")
    print("Saved to corpus/ directory")

if __name__ == "__main__":
    main()