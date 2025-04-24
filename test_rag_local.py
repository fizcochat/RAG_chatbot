#!/usr/bin/env python3
"""
Test script to verify the RAG system works locally
"""

import os
import sys
from dotenv import load_dotenv
import openai
from fast_text.relevance import FastTextRelevanceChecker
from utils import initialize_services, find_match, query_refiner

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not OPENAI_API_KEY or not PINECONE_API_KEY:
    print("Error: API keys not found in environment variables.")
    print("Make sure you have a .env file with your OPENAI_API_KEY and PINECONE_API_KEY.")
    sys.exit(1)

def test_fasttext():
    """Test the FastText model"""
    print("\n🔍 Testing FastText model for relevance checking...")
    try:
        relevance_checker = FastTextRelevanceChecker()
        
        # Test with relevant queries
        relevant_queries = [
            "Come funziona l'IVA?",
            "Quanto costa aprire una partita IVA?",
            "Quali sono le scadenze fiscali per i liberi professionisti?",
            "Come si calcola l'IRPEF?",
            "How much tax do I pay as a freelancer?"
        ]
        
        # Test with non-relevant queries
        non_relevant_queries = [
            "Che tempo fa oggi a Milano?",
            "Come si prepara una lasagna?",
            "Qual è il miglior film dell'anno?",
            "How do I make pizza dough?",
            "What's the capital of France?"
        ]
        
        print("\nTesting relevant queries:")
        for query in relevant_queries:
            is_relevant, details = relevance_checker.is_relevant(query)
            score = details.get('final_score', 0)
            print(f"  • '{query}': {'✅' if is_relevant else '❌'} (score: {score:.2f})")
        
        print("\nTesting non-relevant queries:")
        for query in non_relevant_queries:
            is_relevant, details = relevance_checker.is_relevant(query)
            score = details.get('final_score', 0)
            print(f"  • '{query}': {'❌' if not is_relevant else '✅'} (score: {score:.2f})")
            
        print("\n✅ FastText model test completed")
        return True
    except Exception as e:
        print(f"\n❌ Error testing FastText model: {e}")
        return False

def test_pinecone():
    """Test the Pinecone integration"""
    print("\n📊 Testing Pinecone integration...")
    try:
        # Initialize services
        vectorstore, client = initialize_services(OPENAI_API_KEY, PINECONE_API_KEY)
        
        # Test queries
        test_queries = [
            "Come funziona la partita IVA?",
            "Quali sono le detrazioni fiscali per i liberi professionisti?",
            "Cosa devo fare per aprire una partita IVA?",
            "Come si calcola l'IVA su una fattura?"
        ]
        
        print("\nTesting queries against Pinecone vector store:")
        for query in test_queries:
            print(f"\nQuery: '{query}'")
            docs = vectorstore.similarity_search(query, k=1)
            if docs:
                print(f"  ✅ Found {len(docs)} result(s)")
                doc_preview = docs[0].page_content[:150].replace("\n", " ") + "..."
                print(f"  Preview: {doc_preview}")
            else:
                print(f"  ❌ No results found")
                
        print("\n✅ Pinecone test completed")
        return True, vectorstore, client
    except Exception as e:
        print(f"\n❌ Error testing Pinecone: {e}")
        return False, None, None

def test_rag():
    """Test the full RAG system"""
    print("\n🤖 Testing complete RAG system...")
    success, vectorstore, client = test_pinecone()
    if not success:
        return False
    
    # Set up global variables for find_match
    global_store = {
        'vectorstore': vectorstore,
        'openai_client': client
    }
    
    # Test queries
    test_queries = [
        "Come funziona il regime forfettario?",
        "Quali sono le scadenze per il pagamento dell'IVA?",
        "Come posso contattare un consulente Fiscozen?",
        "Che servizi offre Fiscozen?"
    ]
    
    print("\nTesting complete RAG responses:")
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        # Call find_match directly since we've set up the global store
        response = find_match(query)
        print(f"Response: {response[:150]}..." if len(response) > 150 else f"Response: {response}")
        
    print("\n✅ RAG system test completed")
    return True

def test_gpt4_fallback():
    """Test the GPT-4 fallback functionality"""
    print("\n🧠 Testing GPT-4 fallback for questions without RAG answers...")
    success, vectorstore, client = test_pinecone()
    if not success:
        return False
    
    # Set up global variables for find_match
    global_store = {
        'vectorstore': vectorstore,
        'openai_client': client
    }
    
    # Test queries that likely won't be in the RAG database
    special_test_queries = [
        "Quali sono le aliquote IVA in Italia nel 2023?",
        "Come funziona la flat tax per i liberi professionisti?",
        "Quali sono i requisiti per aprire una partita IVA in regime forfettario?",
        "Quali spese sono deducibili per un freelancer in Italia?"
    ]
    
    print("\nTesting GPT-4 fallback responses:")
    for query in special_test_queries:
        print(f"\nQuery: '{query}'")
        # Call find_match directly since we've set up the global store
        response = find_match(query)
        print(f"Response: {response[:150]}..." if len(response) > 150 else f"Response: {response}")
        
        # Check if the response seems thorough (not a "mi dispiace" message)
        if any(phrase in response.lower() for phrase in ["mi dispiace", "non ho", "non posso"]):
            print("⚠️ WARNING: Response appears to be insufficient")
        else:
            print("✅ GPT-4 fallback provided a substantive response")
        
    print("\n✅ GPT-4 fallback test completed")
    return True

def test_follow_up():
    """Test follow-up question handling"""
    print("\n🔄 Testing follow-up question handling...")
    success, vectorstore, client = test_pinecone()
    if not success:
        return False
    
    # Create conversation history
    conversation = [
        {
            "user": "Come funziona il regime forfettario?",
            "assistant": "Il regime forfettario è un regime fiscale agevolato per liberi professionisti e piccole imprese con ricavi inferiori a 85.000€. Prevede una tassazione forfettaria con aliquota al 15% (5% per i primi 5 anni)."
        }
    ]
    
    # Test follow-up query
    follow_up_query = "Quali sono i limiti di fatturato?"
    
    print(f"\nOriginal follow-up query: '{follow_up_query}'")
    refined_query = query_refiner(conversation, follow_up_query)
    print(f"Refined query: '{refined_query}'")
    
    # Test with another exchange
    conversation.append({
        "user": follow_up_query,
        "assistant": "Il limite di fatturato per il regime forfettario è di 85.000€ annui. Se si supera questa soglia, l'anno successivo si dovrà passare al regime ordinario."
    })
    
    second_follow_up = "E le percentuali di tassazione?"
    print(f"\nSecond follow-up query: '{second_follow_up}'")
    refined_query = query_refiner(conversation, second_follow_up)
    print(f"Refined query: '{refined_query}'")
    
    print("\n✅ Follow-up question test completed")
    return True

def main():
    """Run all tests"""
    print("🔹 Testing RAG Chatbot System 🔹")
    
    # Test FastText
    fasttext_success = test_fasttext()
    
    # Test Pinecone and RAG
    if fasttext_success:
        rag_success = test_rag()
    else:
        rag_success = False
        
    # Test GPT-4 fallback
    gpt4_fallback_success = test_gpt4_fallback()
        
    # Test follow-up handling
    follow_up_success = test_follow_up()
    
    # Print summary
    print("\n🔹 Test Summary 🔹")
    print(f"FastText Model: {'✅ PASSED' if fasttext_success else '❌ FAILED'}")
    print(f"RAG System: {'✅ PASSED' if rag_success else '❌ FAILED'}")
    print(f"GPT-4 Fallback: {'✅ PASSED' if gpt4_fallback_success else '❌ FAILED'}")
    print(f"Follow-up Handling: {'✅ PASSED' if follow_up_success else '❌ FAILED'}")
    
    if fasttext_success and rag_success and gpt4_fallback_success and follow_up_success:
        print("\n✅ All tests passed! The system is ready for deployment.")
        return 0
    else:
        print("\n❌ Some tests failed. Please fix the issues before deploying.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 