"""
Test script for the FastText relevance checker.
"""

import os
import sys
import logging
import pytest
from unittest.mock import Mock, patch

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
def mock_fasttext():
    """Fixture to create a mock FastText model."""
    with patch('fasttext.load_model') as mock_load:
        mock_model = Mock()
        # Mock the predict method to return proper format: (labels, probabilities)
        def mock_predict(text, k=-1):
            if any(kw in text.lower() for kw in ['iva', 'tax', 'fiscal', 'redditi', 'fattur']):
                return (["__label__IVA"], [0.8])
            return (["__label__Other"], [0.9])
        mock_model.predict = mock_predict
        mock_load.return_value = mock_model
        yield mock_model

@pytest.fixture
def relevance_checker(mock_fasttext):
    """Fixture to create and return a FastTextRelevanceChecker instance with mocked model."""
    model_path = "fast_text/models/tax_classifier.bin"
    checker = FastTextRelevanceChecker(model_path)
    # Override the model to use our mock
    checker.model = mock_fasttext
    return checker

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
    is_relevant, details = relevance_checker.is_relevant(query)
    
    assert details['keyword_score'] > 0.5, "Tax-related query should have high keyword score"
    assert 'iva' in details['preprocessed_text'].lower(), "Preprocessed text should contain 'iva'"

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
    assert details2['keyword_score'] > 0.5, "Follow-up question should have high keyword score due to context"

if __name__ == "__main__":
    pytest.main([__file__]) 