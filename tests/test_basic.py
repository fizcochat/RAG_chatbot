"""
Basic tests for the Fiscozen chatbot
"""
import os
import sys
import pytest
from unittest.mock import patch, MagicMock

# Set testing environment flag
os.environ["PYTEST_CURRENT_TEST"] = "yes"

# Import the module with get_response function
from main import get_response, RelevanceChecker

# Mock dependencies to avoid actual API calls and model loading during tests
@pytest.fixture(autouse=True)
def mock_dependencies(monkeypatch):
    # Mock RelevanceChecker
    mock_checker = MagicMock()
    mock_checker.check_relevance.return_value = {
        'is_relevant': True,
        'topic': 'IVA',
        'confidence': 0.9,
        'tax_related_probability': 0.9,
        'probabilities': {'IVA': 0.9, 'Fiscozen': 0.05, 'Other': 0.05}
    }
    monkeypatch.setattr("main.relevance_checker", mock_checker)
    
    # Mock conversation.predict
    mock_conversation = MagicMock()
    mock_conversation.predict.return_value = "This is a mock response about taxes."
    monkeypatch.setattr("main.conversation", mock_conversation)
    
    # Mock query_refiner
    monkeypatch.setattr("main.query_refiner", lambda client, conv, query: query)
    
    # Mock find_match
    monkeypatch.setattr("main.find_match", lambda vs, query: "Mock context")

def test_get_response_relevance():
    """Test that get_response correctly processes tax-related queries"""
    # Test with a tax-related query
    response = get_response("What is the IVA rate in Italy?")
    assert "This is a mock response about taxes" in response
    assert len(response) > 10

def test_get_response_off_topic():
    """Test that get_response correctly handles off-topic queries"""
    # Temporarily modify the mock to return off-topic
    with patch("main.relevance_checker") as mock_checker:
        mock_checker.check_relevance.return_value = {
            'is_relevant': False,
            'topic': 'Other',
            'confidence': 0.9,
            'tax_related_probability': 0.1,
            'probabilities': {'IVA': 0.05, 'Fiscozen': 0.05, 'Other': 0.9}
        }
        
        # Test with an off-topic query
        response = get_response("What's the weather like today?")
        assert "OFF-TOPIC DETECTED" in response
        
        # Test with a second off-topic query (should trigger redirection)
        response = get_response("Tell me about football", "same_user")
        assert "redirect" in response.lower()