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
    
    def oracle_retrieve(self, question, qa_data, top_k=6, verbose=False):
        """Oracle retrieval with optional verbose output."""
        required_facts = qa_data['facts']
        
        if verbose:
            print(f"Looking for required facts: {required_facts}")
        
        # Find documents containing required facts
        required_docs = []
        for fact in required_facts:
            for doc_id, doc in enumerate(self.documents):
                # Handle both string documents and dict documents
                doc_content = doc if isinstance(doc, str) else doc.get('content', str(doc))
                
                if fact in doc_content:
                    required_docs.append(doc_id)
                    if verbose:
                        print(f"✅ Found '{fact}' in doc {doc_id}")
                    break
        
        # Get entity distractors (documents mentioning same entities)
        entities_in_question = self.extract_entities_from_text(question)
        entity_distractors = []
        for doc_id, doc in enumerate(self.documents):
            if doc_id not in required_docs:
                doc_content = doc if isinstance(doc, str) else doc.get('content', str(doc))
                if any(entity in doc_content for entity in entities_in_question):
                    entity_distractors.append(doc_id)
                    if len(entity_distractors) >= 2:
                        break
        
        # Get random distractors
        all_used = set(required_docs + entity_distractors)
        available_docs = [i for i in range(len(self.documents)) if i not in all_used]
        random_distractors = random.sample(available_docs, min(3, len(available_docs)))
        
        # Combine all retrieved documents
        retrieved_doc_ids = required_docs + entity_distractors + random_distractors
        retrieved_doc_ids = retrieved_doc_ids[:top_k]  # Limit to top_k
        
        # Get actual document content
        retrieved_docs = []
        for doc_id in retrieved_doc_ids:
            doc = self.documents[doc_id]
            doc_content = doc if isinstance(doc, str) else doc.get('content', str(doc))
            retrieved_docs.append(doc_content)
        
        if verbose:
            print(f"Oracle retrieved {len(retrieved_doc_ids)} documents:")
            print(f"  - {len(required_docs)} required docs")
            print(f"  - {len(entity_distractors)} entity distractors") 
            print(f"  - {len(random_distractors)} random distractors")
        
        return retrieved_docs, retrieved_doc_ids

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
        elif q['reasoning_type'] == '4-hop' and len(test_questions) == 2:
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