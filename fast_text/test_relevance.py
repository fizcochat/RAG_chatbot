"""
Test script for the FastText relevance checker.
"""

import os
import sys
import logging
import pytest

# Add parent directory to path for imports
project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from fast_text import FastTextRelevanceChecker

# Configure logging
logging.basicConfig(level=logging.INFO)

# Test queries and their expected relevance
TEST_QUERIES = [
    # Tax-related queries (should be relevant)
    ("Come funziona l'IVA per un libero professionista?", True),
    ("Quali sono le aliquote IVA in Italia?", True),
    ("Ho bisogno di aiuto con la dichiarazione dei redditi", True),
    ("Come posso ridurre il carico fiscale della mia attività?", True),
    ("Devo pagare l'IVA sulle fatture estere?", True),
    ("Quali spese posso detrarre come freelancer?", True),
    ("Come funziona il regime forfettario?", True),
    ("Vorrei sapere di più sui servizi di Fiscozen", True),
    
    # Non-tax queries (should not be relevant)
    ("Che tempo farà domani a Roma?", False),
    ("Come si prepara una buona pasta alla carbonara?", False),
    ("Quali sono i migliori ristoranti in città?", False),
    ("Mi puoi consigliare un buon film da vedere?", False),
    ("Come posso migliorare il mio inglese?", False),
    ("Dove posso trovare un idraulico?", False),
    ("Quali sono gli orari del supermercato?", False),
    ("Come si allena un cane?", False)
]

@pytest.fixture
def relevance_checker():
    """Fixture to create and return a FastTextRelevanceChecker instance."""
    model_path = "fast_text/models/tax_classifier.bin"
    return FastTextRelevanceChecker(model_path)

def test_relevance_checker_initialization(relevance_checker):
    """Test that the relevance checker initializes correctly."""
    assert relevance_checker is not None
    assert isinstance(relevance_checker, FastTextRelevanceChecker)

@pytest.mark.parametrize("query,expected_relevant", TEST_QUERIES)
def test_query_relevance(relevance_checker, query, expected_relevant):
    """Test relevance checking for various queries."""
    try:
        # Get relevance check result and details
        is_relevant, details = relevance_checker.is_relevant(query)
        
        # Basic assertions
        assert isinstance(is_relevant, bool)
        assert isinstance(details, dict)
        assert 'preprocessed_text' in details
        assert 'keyword_score' in details
        
        # Check if the relevance matches expected
        assert is_relevant == expected_relevant, \
            f"Query '{query}' was classified as {'relevant' if is_relevant else 'not relevant'} " \
            f"but expected {'relevant' if expected_relevant else 'not relevant'}"
        
        # Additional assertions for details
        assert 0 <= details['keyword_score'] <= 1, "Keyword score should be between 0 and 1"
        assert details['preprocessed_text'], "Preprocessed text should not be empty"
        
    except Exception as e:
        pytest.fail(f"Error processing query '{query}': {str(e)}")

def test_keyword_scoring(relevance_checker):
    """Test the keyword scoring functionality."""
    # Test with a known tax-related query
    query = "Come funziona l'IVA?"
    _, details = relevance_checker.is_relevant(query)
    
    assert details['keyword_score'] > 0.5, "Tax-related query should have high keyword score"
    assert 'iva' in details['preprocessed_text'], "Preprocessed text should contain 'iva'"

def test_invalid_input(relevance_checker):
    """Test handling of invalid input."""
    # Test with empty string
    is_relevant, details = relevance_checker.is_relevant("")
    assert not is_relevant, "Empty string should not be relevant"
    
    # Test with None (should raise TypeError)
    with pytest.raises(Exception):
        relevance_checker.is_relevant(None)

def test_context_handling(relevance_checker):
    """Test handling of conversation context for follow-up questions."""
    # First query establishes context
    query1 = "Come funziona l'IVA per un libero professionista?"
    is_relevant1, _ = relevance_checker.is_relevant(query1)
    assert is_relevant1, "Initial tax query should be relevant"
    
    # Follow-up question
    query2 = "Quali sono le aliquote?"
    is_relevant2, details2 = relevance_checker.is_relevant(query2)
    assert is_relevant2, "Follow-up question should be relevant due to context"
    assert details2.get('context_relevance', False), "Context relevance should be True for follow-up"

def main():
    print("\n🔹 Testing FastText Relevance Checker 🔹\n")
    
    # Initialize the checker
    model_path = "fast_text/models/tax_classifier.bin"
    print(f"Model path: {model_path}")
    checker = FastTextRelevanceChecker(model_path)
    
    # Test all queries
    print(f"Testing {len(TEST_QUERIES)} queries...\n")
    
    tax_correct = 0
    non_tax_correct = 0
    tax_total = sum(1 for _, is_tax in TEST_QUERIES if is_tax)
    non_tax_total = sum(1 for _, is_tax in TEST_QUERIES if not is_tax)
    
    for query, expected_relevant in TEST_QUERIES:
        print(f"\n📝 Query: {query}")
        try:
            # Get relevance check result and details
            is_relevant, details = checker.is_relevant(query)
            
            # Print details
            print(f"   Preprocessed: {details['preprocessed_text']}")
            print(f"   Keyword score: {details['keyword_score']:.3f}")
            
            if 'model_predictions' in details:
                print("   Model predictions:")
                for label, prob in details['model_predictions'].items():
                    print(f"   - {label}: {prob:.3f}")
            
            if details['keywords_found']:
                print(f"   Keywords found: {details['keywords_found']}")
            if details['phrases_found']:
                print(f"   Phrases found: {details['phrases_found']}")
            
            if 'combined_score' in details:
                print(f"   Combined score: {details['combined_score']:.3f}")
            
            # Print result
            if is_relevant == expected_relevant:
                if expected_relevant:
                    tax_correct += 1
                    print(f"   Result: ✅ Relevant (Expected relevant)")
                else:
                    non_tax_correct += 1
                    print(f"   Result: ✅ Not relevant (Expected not relevant)")
            else:
                if expected_relevant:
                    print(f"   Result: ❌ Not relevant (Expected relevant)")
                else:
                    print(f"   Result: ❌ Relevant (Expected not relevant)")
                    
        except Exception as e:
            print(f"\n❌ Error processing query: {e}")
    
    # Print statistics
    print("\n📊 Results:")
    tax_accuracy = (tax_correct / tax_total * 100) if tax_total > 0 else 0
    non_tax_accuracy = (non_tax_correct / non_tax_total * 100) if non_tax_total > 0 else 0
    overall_accuracy = ((tax_correct + non_tax_correct) / len(TEST_QUERIES) * 100)
    
    print(f"Tax-related queries accuracy: {tax_accuracy:.1f}%")
    print(f"Non-tax queries accuracy: {non_tax_accuracy:.1f}%")
    print(f"Overall accuracy: {overall_accuracy:.1f}%")

if __name__ == "__main__":
    main() 