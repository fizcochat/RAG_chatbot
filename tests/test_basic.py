import pytest
from main import get_response

# pytest tests/test_basic.py

def test_basic():
    assert True

def test_iva_basic_query():
    # Test basic IVA questions
    query = "What is IVA?"
    response = get_response(query)
    assert isinstance(response, str), "Response should be a string"
    assert response, "Response should not be empty"
    assert any(keyword in response.lower() for keyword in ["tax", "value-added", "vat"]), "Response should mention VAT or taxation"

def test_tax_payment_query():
    # Test tax payment related questions
    query = "How can I pay my taxes?"
    response = get_response(query)
    assert isinstance(response, str)
    assert response
    assert any(keyword in response.lower() for keyword in ["pay", "payment", "method", "process"]), "Response should mention payment methods"

def test_empty_query():
    # Test handling of empty queries
    query = ""
    response = get_response(query)
    assert isinstance(response, str)
    assert response == "Please enter a valid question.", "Response should prompt user to enter a valid question"

def test_irrelevant_query():
    # Test handling of off-topic questions
    query = "What is the weather like on Mars?"
    response = get_response(query)
    assert isinstance(response, str)
    assert response is not None and len(response.strip()) > 0, "Response should not be empty"
    assert query.lower() not in response.lower(), f"Unexpected response: {response}"
    # (Optional) Log response for debugging if needed
    print(f"Chatbot Response: {response}")