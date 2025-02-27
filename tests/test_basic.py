import pytest
from main import get_response, classify_query  # We'll need to add classify_query to main.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np

def test_basic():
    assert True

def test_iva_basic_query():
    # Test basic IVA questions
    query = "What is IVA?"
    response = get_response(query)
    assert response is not None
    assert "Individual Voluntary Arrangement" in response

def test_tax_payment_query():
    # Test tax payment related questions
    query = "How can I pay my taxes?"
    response = get_response(query)
    assert response is not None
    assert any(keyword in response.lower() for keyword in ["payment", "pay", "method"])

def test_empty_query():
    # Test handling of empty queries
    query = ""
    response = get_response(query)
    assert response is not None
    assert "please ask" in response.lower() or "can you rephrase" in response.lower()

def test_irrelevant_query():
    # Test handling of off-topic questions
    query = "What is the weather like on Mars?"
    response = get_response(query)
    assert response is not None
    assert "cannot help" in response.lower() or "focus on" in response.lower()

def test_query_classifier():
    # Training data
    queries = [
        # Fiscozen related (customer queries)
        "How do I contact Fiscozen support?",
        "What services does Fiscozen offer?",
        "Can Fiscozen help with my accounting?",
        "Fiscozen pricing plans",
        "How to register with Fiscozen",
        
        # IVA related
        "What is an IVA?",
        "How long does an IVA last?",
        "IVA payment terms",
        "Can I get an IVA if I'm self-employed?",
        "IVA debt minimum",
        
        # Tax related
        "How to pay taxes online?",
        "When is the tax deadline?",
        "Tax deductions for businesses",
        "VAT registration process",
        "Corporate tax rates",
        
        # Unrelated queries
        "What's the weather today?",
        "Best restaurants nearby",
        "How to make pasta",
        "Latest football scores",
        "Movie showtimes"
    ]
    
    # Labels: 0 for Fiscozen, 1 for IVA, 2 for Tax, 3 for Unrelated
    labels = [0,0,0,0,0, 1,1,1,1,1, 2,2,2,2,2, 3,3,3,3,3]
    
    # Test the classifier function
    result = classify_query("How can Fiscozen help my business?")
    assert result in [0, 1, 2, 3]  # Should return a valid category
    
    # Test specific cases
    assert classify_query("What is Fiscozen?") == 0  # Should be Fiscozen related
    assert classify_query("Tell me about IVA") == 1  # Should be IVA related
    assert classify_query("How do I pay my taxes?") == 2  # Should be Tax related
    assert classify_query("What's the weather like on Mars?") == 3  # Should be Unrelated

# Add more meaningful tests based on your chatbot functionality 