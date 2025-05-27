import json
import re
import random
from collections import defaultdict
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class OracleRetriever:
    def __init__(self, documents):
        self.documents = documents
        
        # Build entity index for finding documents
        self.entity_index = self.build_entity_index()
    
    def build_entity_index(self):
        """Build index mapping entities to document IDs"""
        entity_index = defaultdict(set)
        
        for doc_id, doc in enumerate(self.documents):
            entities = self.extract_entities_from_text(doc)
            for entity in entities:
                entity_index[entity].add(doc_id)
        
        return entity_index
    
    def extract_entities_from_text(self, text):
        """Extract entity IDs from text"""
        entities = set()
        patterns = [
            r'P\d{4}', r'C\d{4}', r'L\d{4}', r'PJ\d{3}', 
            r'S\d{3}', r'T\d{3}', r'D\d{3}'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            entities.update(matches)
        
        return entities
    
    def oracle_retrieve(self, question, qa_data, top_k=6):
        """Oracle retrieval that guarantees required facts + realistic extras"""
        
        # Get required document IDs containing the facts
        required_facts = qa_data['facts']
        required_doc_ids = []
        
        print(f"Looking for required facts: {required_facts}")
        
        for fact in required_facts:
            found = False
            for doc_id, doc in enumerate(self.documents):
                if fact in doc:
                    if doc_id not in required_doc_ids:
                        required_doc_ids.append(doc_id)
                        print(f"✅ Found '{fact}' in doc {doc_id}")
                    found = True
                    break
            
            if not found:
                print(f"❌ Could not find fact: '{fact}'")
        
        # Add realistic distractors: documents containing question entities
        question_entities = self.extract_entities_from_text(question)
        distractor_docs = set()
        
        for entity in question_entities:
            if entity in self.entity_index:
                distractor_docs.update(self.entity_index[entity])
        
        # Remove required docs from distractors
        distractor_docs = [d for d in distractor_docs if d not in required_doc_ids]
        
        # Calculate how many distractors we can add
        remaining_slots = top_k - len(required_doc_ids)
        
        if remaining_slots > 0:
            # Add entity-related distractors first
            entity_distractors = list(distractor_docs)[:min(2, remaining_slots)]
            remaining_slots -= len(entity_distractors)
            
            # Add random distractors if we still have slots
            random_distractors = []
            if remaining_slots > 0:
                all_doc_ids = list(range(len(self.documents)))
                available_random = [d for d in all_doc_ids 
                                  if d not in required_doc_ids and d not in distractor_docs]
                random_distractors = random.sample(
                    available_random,
                    min(remaining_slots, len(available_random))
                )
            
            # Combine all documents
            final_doc_ids = required_doc_ids + entity_distractors + random_distractors
        else:
            # If we have too many required docs, just use required ones
            final_doc_ids = required_doc_ids[:top_k]
        
        # Shuffle and ensure we don't exceed top_k
        random.shuffle(final_doc_ids)
        final_doc_ids = final_doc_ids[:top_k]
        
        retrieved_docs = [self.documents[doc_id] for doc_id in final_doc_ids]
        
        print(f"Oracle retrieved {len(final_doc_ids)} documents:")
        print(f"  - {len(required_doc_ids)} required docs")
        if remaining_slots > 0:
            print(f"  - {len(entity_distractors)} entity distractors") 
            print(f"  - {len(random_distractors)} random distractors")
        
        return retrieved_docs, final_doc_ids

class HybridRetriever:
    def __init__(self, documents, embedding_model='all-MiniLM-L6-v2'):
        self.documents = documents
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Build entity index for fast keyword lookup
        self.entity_index = self.build_entity_index()
        
        # Pre-compute embeddings for all documents
        print("Computing document embeddings...")
        self.doc_embeddings = self.embedding_model.encode(documents)
        print(f"Computed embeddings for {len(documents)} documents")
    
    def build_entity_index(self):
        """Build index mapping entities to document IDs"""
        entity_index = defaultdict(set)
        
        for doc_id, doc in enumerate(self.documents):
            entities = self.extract_entities_from_text(doc)
            for entity in entities:
                entity_index[entity].add(doc_id)
        
        return entity_index
    
    def extract_entities_from_text(self, text):
        """Extract entity IDs from text"""
        entities = set()
        patterns = [
            r'P\d{4}', r'C\d{4}', r'L\d{4}', r'PJ\d{3}', 
            r'S\d{3}', r'T\d{3}', r'D\d{3}'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            entities.update(matches)
        
        return entities

def load_documents(filepath):
    """Load documents from JSONL file"""
    documents = []
    with open(filepath, 'r') as f:
        for line in f:
            doc = json.loads(line.strip())
            documents.append(doc)
    return documents

def test_oracle_retriever():
    """Test the oracle retriever"""
    
    # Load documents
    print("Loading documents...")
    train_docs = load_documents('corpus/docs_train.jsonl')
    
    # Initialize oracle retriever
    print("Initializing oracle retriever...")
    retriever = OracleRetriever(train_docs)
    
    # Load questions
    print("Loading test questions...")
    with open('corpus/qa_train.jsonl', 'r') as f:
        questions = [json.loads(line) for line in f]
    
    # Test with multi-hop questions
    test_questions = []
    for q in questions:
        if q['reasoning_type'] == '2-hop' and len(test_questions) == 0:
            test_questions.append(q)
        elif q['reasoning_type'] == '3-hop' and len(test_questions) == 1:
            test_questions.append(q)
            break
    
    for i, question_data in enumerate(test_questions):
        question = question_data['question']
        expected_facts = question_data['facts']
        reasoning_type = question_data['reasoning_type']
        
        print(f"\n{'='*60}")
        print(f"TEST {i+1}: {reasoning_type} Question")
        print(f"{'='*60}")
        print(f"Question: {question}")
        print(f"Expected facts: {expected_facts}")
        
        # Oracle retrieval with 6 documents
        retrieved_docs, doc_ids = retriever.oracle_retrieve(question, question_data, top_k=6)
        
        print(f"\nRetrieved documents:")
        for j, doc in enumerate(retrieved_docs):
            print(f"  {j+1}. Doc {doc_ids[j]}: {doc}")
        
        # Check fact coverage
        all_retrieved_text = " ".join(retrieved_docs)
        facts_found = 0
        
        print(f"\n--- Fact Coverage ---")
        for fact in expected_facts:
            if fact in all_retrieved_text:
                print(f"✅ FOUND: {fact}")
                facts_found += 1
            else:
                print(f"❌ MISSING: {fact}")
        
        print(f"\nCoverage: {facts_found}/{len(expected_facts)} facts found")
        
        if facts_found == len(expected_facts):
            print("🎯 SUCCESS: All facts retrieved!")
        else:
            print("⚠️  FAILURE: Missing facts")

if __name__ == "__main__":
    test_oracle_retriever()