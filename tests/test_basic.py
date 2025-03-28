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
    # Reset any existing counters before test
    if hasattr(get_response, 'off_topic_count'):
        get_response.off_topic_count = {}
    
    # Use a consistent conversation ID for tracking
    conversation_id = "test_conversation"
    
    with patch("main.relevance_checker") as mock_checker:
        # Configure the mock to always return off-topic
        mock_checker.check_relevance.return_value = {
            'is_relevant': False,
            'topic': 'Other',
            'confidence': 0.9,
            'tax_related_probability': 0.1,
            'probabilities': {'IVA': 0.05, 'Fiscozen': 0.05, 'Other': 0.9}
        }
        
        # First off-topic query should get a warning
        response1 = get_response("What's the weather like today?", conversation_id)
        assert "OFF-TOPIC DETECTED" in response1
        assert "redirect" not in response1.lower()
        
        # Second off-topic query with same conversation_id should trigger redirection
        response2 = get_response("Tell me about football", conversation_id)
        assert "OFF-TOPIC CONVERSATION DETECTED" in response2
        assert "redirect" in response2.lower()
        
        # Check that the counter got reset after redirection
        response3 = get_response("What's your favorite color?", conversation_id)
        assert "OFF-TOPIC DETECTED" in response3
        assert "redirect" not in response3.lower()