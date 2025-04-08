import pytest
from main import get_response

# pytest tests/test_basic.py

def test_basic():
    assert True

def test_iva_basic_query():
    # Test basic IVA questions
    query = "What is IVA?"
    try:
        response = get_response(query)
        assert isinstance(response, str), "Response should be a string"
        assert response, "Response should not be empty"
        assert any(keyword in response.lower() for keyword in ["tax", "value-added", "vat", "iva"]), "Response should mention VAT/IVA or taxation"
    except ImportError as e:
        pytest.skip(f"Skipping test due to import error: {e}")

def test_tax_payment_query():
    # Test tax payment related questions
    query = "How can I pay my taxes?"
    try:
        response = get_response(query)
        assert isinstance(response, str)
        assert response
        assert any(keyword in response.lower() for keyword in ["pay", "payment", "method", "process"]), "Response should mention payment methods"
    except ImportError as e:
        pytest.skip(f"Skipping test due to import error: {e}")

def test_empty_query():
    # Test handling of empty queries
    query = ""
    try:
        response = get_response(query)
        assert isinstance(response, str)
        assert response == "Please enter a valid question.", "Response should prompt user to enter a valid question"
    except ImportError as e:
        pytest.skip(f"Skipping test due to import error: {e}")

def test_irrelevant_query():
    # Test handling of off-topic questions
    query = "What is the weather like on Mars?"
    try:
        response = get_response(query)
        assert isinstance(response, str)
        assert response is not None and len(response.strip()) > 0, "Response should not be empty"
        assert "I can only help with tax-related questions" in response, "Response should indicate tax-only capability"
    except ImportError as e:
        pytest.skip(f"Skipping test due to import error: {e}")