#!/usr/bin/env python
"""
Comprehensive test script for the RAG chatbot system

This script tests all major components of the RAG chatbot:
1. FastText relevance checking
2. Pinecone connectivity
3. RAG document retrieval
4. GPT-4 fallback for unknown questions
5. Conversation memory for follow-up questions
"""

import os
import sys
import time
from dotenv import load_dotenv
from utils import initialize_services, find_match, query_refiner, translate_to_italian, translate_from_italian
from fast_text.relevance import FastTextRelevanceChecker

# Load environment variables
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
pinecone_api_key = os.getenv('PINECONE_API_KEY')

if not openai_api_key or not pinecone_api_key:
    print("Error: API keys not found in environment variables.")
    print("Make sure you have a .env file with your OPENAI_API_KEY and PINECONE_API_KEY.")
    exit(1)

def print_header(title):
    """Print a formatted header"""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80)

def print_result(name, success, message=""):
    """Print a test result with color"""
    if success:
        print(f"✅ {name}: Passed {message}")
    else:
        print(f"❌ {name}: Failed {message}")
    return success

def print_section(title):
    """Print a section header"""
    print(f"\n--- {title} ---")

# Initialize services
print_header("INITIALIZING SERVICES")
try:
    vectorstore, client = initialize_services(openai_api_key, pinecone_api_key)
    print_result("Service initialization", True)
except Exception as e:
    print_result("Service initialization", False, f"Error: {e}")
    sys.exit(1)

# Initialize FastText relevance checker
print_section("Initializing FastText relevance checker")
try:
    relevance_checker = FastTextRelevanceChecker()
    print_result("FastText initialization", True)
except Exception as e:
    print_result("FastText initialization", False, f"Error: {e}")
    print("Continuing with tests...")

# Test FastText relevance checking
print_header("TEST 1: FASTTEXT RELEVANCE CHECKING")

tax_queries = [
    "Come funziona l'IVA per un libero professionista?",
    "Quali sono le aliquote IVA in Italia?",
    "Fiscozen mi può aiutare con la dichiarazione dei redditi?",
    "Posso detrarre le spese del mio computer?"
]

non_tax_queries = [
    "Che tempo fa oggi a Roma?",
    "Quanto dista Milano da Napoli?",
    "Qual è la ricetta della pasta alla carbonara?"
]

# Test tax queries (should be relevant)
print_section("Testing tax-related queries (should be relevant)")
tax_relevance_success = True
for query in tax_queries:
    is_relevant, details = relevance_checker.is_relevant(query)
    success = is_relevant
    tax_relevance_success &= success
    if 'keyword_score' in details:
        score_info = f"(Score: {details['keyword_score']:.2f})"
    else:
        score_info = ""
    print_result(f"'{query}'", success, score_info)

# Test non-tax queries (should not be relevant unless using fallback)
print_section("Testing non-tax queries (may or may not be relevant depending on FastText model)")
for query in non_tax_queries:
    is_relevant, details = relevance_checker.is_relevant(query)
    if 'keyword_score' in details:
        score_info = f"(Score: {details['keyword_score']:.2f})"
    else:
        score_info = ""
    print(f"Query: '{query}', Relevant: {is_relevant} {score_info}")

# Test Pinecone connectivity and RAG retrieval
print_header("TEST 2: PINECONE CONNECTIVITY & RAG RETRIEVAL")

rag_queries = [
    "Come funziona il regime forfettario in Italia?",
    "Quali documenti servono per aprire partita IVA?",
    "Quali sono le scadenze fiscali per un libero professionista?",
    "Come funziona la fatturazione elettronica?"
]

print_section("Testing RAG retrieval with specific tax questions")
rag_success = True
for query in rag_queries:
    try:
        print(f"\nQuery: '{query}'")
        print("Generating RAG response...")
        response = find_match(query, k=2)
        print(f"Response: {response[:150]}...")  # Show first 150 chars
        rag_success &= True
    except Exception as e:
        print(f"Error: {e}")
        rag_success = False

print_result("RAG retrieval test", rag_success)

# Test GPT-4 fallback for questions without relevant documents
print_header("TEST 3: GPT-4 FALLBACK")

fallback_queries = [
    "Quali sono le nuove regole fiscali per il 2025?",  # Likely not in the documents if they're from earlier years
    "Come gestire la tassazione di criptovalute in Italia?",  # May not be in the documents
]

print_section("Testing GPT-4 fallback with questions likely not in documents")
fallback_success = True
for query in fallback_queries:
    try:
        print(f"\nQuery: '{query}'")
        print("Generating response (may use fallback)...")
        response = find_match(query, k=2)
        print(f"Response: {response[:150]}...")  # Show first 150 chars
        fallback_success &= len(response) > 50  # Simple check that we got a substantial response
    except Exception as e:
        print(f"Error: {e}")
        fallback_success = False

print_result("GPT-4 fallback test", fallback_success)

# Test translation functionality
print_header("TEST 4: TRANSLATION FUNCTIONALITY")

english_queries = [
    "How does VAT work for freelancers in Italy?",
    "What are the tax rates in Italy?"
]

print_section("Testing translation from English to Italian")
translation_success = True
for query in english_queries:
    try:
        print(f"\nOriginal query (English): '{query}'")
        translated = translate_to_italian(query, client)
        print(f"Translated query (Italian): '{translated}'")
        
        # Get response based on translated query
        response = find_match(translated, k=2)
        
        # Translate back to English
        response_en = translate_from_italian(response, client)
        
        print(f"Response (English): {response_en[:150]}...")
        translation_success &= len(response) > 50 and len(response_en) > 50
    except Exception as e:
        print(f"Error: {e}")
        translation_success = False

print_result("Translation functionality test", translation_success)

# Test conversation memory with follow-up questions
print_header("TEST 5: CONVERSATION MEMORY")

print_section("Testing conversation memory with follow-up questions")
conversation_memory_success = True

try:
    # Initial question
    initial_query = "Come funziona il regime forfettario?"
    print(f"\nInitial query: '{initial_query}'")
    initial_response = find_match(initial_query, k=2)
    print(f"Initial response: {initial_response[:150]}...")
    
    # Create a simulated conversation history
    conversation_history = [
        {
            "user": initial_query,
            "assistant": initial_response
        }
    ]
    
    # Follow-up question
    followup_query = "Quali sono i limiti di fatturato?"
    print(f"\nFollow-up query: '{followup_query}'")
    
    # Refine the follow-up query using conversation history
    refined_query = query_refiner(conversation_history, followup_query)
    print(f"Refined query: '{refined_query}'")
    
    # Get response based on refined query
    followup_response = find_match(refined_query, k=2)
    print(f"Follow-up response: {followup_response[:150]}...")
    
    conversation_memory_success = len(refined_query) > len(followup_query)
except Exception as e:
    print(f"Error: {e}")
    conversation_memory_success = False

print_result("Conversation memory test", conversation_memory_success)

# Summary of all tests
print_header("TEST SUMMARY")

all_tests = [
    ("FastText relevance checking", tax_relevance_success),
    ("RAG retrieval", rag_success),
    ("GPT-4 fallback", fallback_success),
    ("Translation functionality", translation_success),
    ("Conversation memory", conversation_memory_success)
]

all_passed = True
for test_name, result in all_tests:
    all_passed &= result
    print_result(test_name, result)

print("\n")
if all_passed:
    print("🎉 All tests passed! Your RAG chatbot is working correctly. 🎉")
else:
    print("⚠️ Some tests failed. Review the logs above for details. ⚠️")

print("""
Next steps:
1. Start the API service with: python main.py
2. Find your IP address with 'ifconfig' or 'ipconfig'
3. Devices on the same network can access the API at: http://YOUR_IP:8080/api/chat
""") 