import pytest
from main import get_response  # Assuming this is your main chat function

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

# Add more meaningful tests based on your chatbot functionality 