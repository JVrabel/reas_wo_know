import json
import random
from build_retriever import load_documents, OracleRetriever

def test_retrieval_comprehensive(top_k=6):
    """Comprehensive test of oracle retriever across all reasoning types"""
    
    # Load documents and questions
    print("Loading documents...")
    train_docs = load_documents('corpus/docs_train.jsonl')
    
    print("Loading questions...")
    with open('corpus/qa_train.jsonl', 'r') as f:
        all_questions = [json.loads(line) for line in f]
    
    # Initialize oracle retriever
    print("Initializing oracle retriever...")
    retriever = OracleRetriever(train_docs)
    
    print(f"Testing with top_k={top_k} documents per question")
    
    # Group questions by reasoning type
    questions_by_type = {
        '1-hop': [],
        '2-hop': [],
        '3-hop': [],
        '4-hop': []
    }
    
    for q in all_questions:
        reasoning_type = q['reasoning_type']
        if reasoning_type in questions_by_type:
            questions_by_type[reasoning_type].append(q)
    
    print(f"Found questions: {len(questions_by_type['1-hop'])} 1-hop, "
          f"{len(questions_by_type['2-hop'])} 2-hop, "
          f"{len(questions_by_type['3-hop'])} 3-hop, "
          f"{len(questions_by_type['4-hop'])} 4-hop")
    
    # Test each reasoning type
    results = {}
    
    for reasoning_type in ['1-hop', '2-hop', '3-hop', '4-hop']:
        print(f"\n{'='*80}")
        print(f"TESTING {reasoning_type.upper()} QUESTIONS (top_k={top_k})")
        print(f"{'='*80}")
        
        # Sample 5 questions of this type
        questions = questions_by_type[reasoning_type]
        test_questions = random.sample(questions, min(5, len(questions)))
        
        type_results = {
            'total_questions': len(test_questions),
            'perfect_retrieval': 0,
            'partial_retrieval': 0,
            'failed_retrieval': 0,
            'total_facts_expected': 0,
            'total_facts_found': 0
        }
        
        for i, question_data in enumerate(test_questions):
            question = question_data['question']
            expected_facts = question_data['facts']
            
            print(f"\n--- {reasoning_type} Test {i+1}/5 ---")
            print(f"Question: {question}")
            print(f"Expected facts ({len(expected_facts)}): {expected_facts}")
            
            # Oracle retrieval with configurable top_k
            retrieved_docs, doc_ids = retriever.oracle_retrieve(
                question, question_data, top_k=top_k
            )
            
            print(f"\nRetrieved {len(retrieved_docs)} documents:")
            for j, doc in enumerate(retrieved_docs):
                print(f"  {j+1}. Doc {doc_ids[j]}: {doc}")
            
            # Check fact coverage
            all_retrieved_text = " ".join(retrieved_docs)
            facts_found = []
            facts_missing = []
            
            for fact in expected_facts:
                if fact in all_retrieved_text:
                    facts_found.append(fact)
                else:
                    facts_missing.append(fact)
            
            print(f"\n--- Fact Coverage ---")
            for fact in facts_found:
                print(f"✅ FOUND: {fact}")
            for fact in facts_missing:
                print(f"❌ MISSING: {fact}")
            
            coverage = len(facts_found) / len(expected_facts)
            print(f"\nCoverage: {len(facts_found)}/{len(expected_facts)} ({coverage:.1%})")
            
            # Update statistics
            type_results['total_facts_expected'] += len(expected_facts)
            type_results['total_facts_found'] += len(facts_found)
            
            if len(facts_found) == len(expected_facts):
                print("🎯 PERFECT: All facts retrieved!")
                type_results['perfect_retrieval'] += 1
            elif len(facts_found) > 0:
                print("⚠️  PARTIAL: Some facts missing")
                type_results['partial_retrieval'] += 1
            else:
                print("❌ FAILED: No facts found")
                type_results['failed_retrieval'] += 1
        
        results[reasoning_type] = type_results
    
    # Print summary
    print(f"\n{'='*80}")
    print("RETRIEVAL PERFORMANCE SUMMARY")
    print(f"{'='*80}")
    
    for reasoning_type in ['1-hop', '2-hop', '3-hop', '4-hop']:
        stats = results[reasoning_type]
        total_q = stats['total_questions']
        perfect = stats['perfect_retrieval']
        partial = stats['partial_retrieval']
        failed = stats['failed_retrieval']
        fact_recall = stats['total_facts_found'] / stats['total_facts_expected']
        
        print(f"\n{reasoning_type.upper()} Questions ({total_q} tested):")
        print(f"  🎯 Perfect retrieval: {perfect}/{total_q} ({perfect/total_q:.1%})")
        print(f"  ⚠️  Partial retrieval: {partial}/{total_q} ({partial/total_q:.1%})")
        print(f"  ❌ Failed retrieval:  {failed}/{total_q} ({failed/total_q:.1%})")
        print(f"  📊 Fact recall:       {stats['total_facts_found']}/{stats['total_facts_expected']} ({fact_recall:.1%})")
    
    # Overall statistics
    total_questions = sum(r['total_questions'] for r in results.values())
    total_perfect = sum(r['perfect_retrieval'] for r in results.values())
    total_facts_expected = sum(r['total_facts_expected'] for r in results.values())
    total_facts_found = sum(r['total_facts_found'] for r in results.values())
    
    print(f"\nOVERALL PERFORMANCE:")
    print(f"  Questions with perfect retrieval: {total_perfect}/{total_questions} ({total_perfect/total_questions:.1%})")
    print(f"  Overall fact recall: {total_facts_found}/{total_facts_expected} ({total_facts_found/total_facts_expected:.1%})")
    
    if total_facts_found == total_facts_expected:
        print("\n🎉 SUCCESS: Oracle retriever working perfectly!")
    else:
        print(f"\n⚠️  WARNING: {total_facts_expected - total_facts_found} facts still missing")

if __name__ == "__main__":
    import sys
    
    # Allow command line argument for top_k
    if len(sys.argv) > 1:
        top_k = int(sys.argv[1])
    else:
        top_k = 5
    
    test_retrieval_comprehensive(top_k=top_k) 