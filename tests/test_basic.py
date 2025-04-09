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
        # The response should be in Italian since that's what the chatbot returns
        assert any(keyword in response.lower() for keyword in ["iva", "tasse", "fiscali"]), "Response should mention IVA or tax-related terms in Italian"
    except ImportError as e:
        pytest.skip(f"Skipping test due to import error: {e}")

def test_tax_payment_query():
    # Test tax payment related questions
    query = "How can I pay my taxes?"
    try:
        response = get_response(query)
        assert isinstance(response, str)
        assert response
        # The response should be in Italian
        assert any(keyword in response.lower() for keyword in ["pagare", "pagamento", "tasse"]), "Response should mention payment methods in Italian"
    except ImportError as e:
        pytest.skip(f"Skipping test due to import error: {e}")

def test_empty_query():
    # Test handling of empty queries
    query = ""
    try:
        response = get_response(query)
        assert isinstance(response, str)
        # The error message is in Italian
        assert response == "Mi dispiace, si è verificato un errore. Per favore, riprova.", "Response should indicate an error in Italian"
    except ImportError as e:
        pytest.skip(f"Skipping test due to import error: {e}")

def test_irrelevant_query():
    # Test handling of off-topic questions
    query = "What is the weather like on Mars?"
    try:
        response = get_response(query)
        assert isinstance(response, str)
        assert response is not None and len(response.strip()) > 0, "Response should not be empty"
        expected_response = "Mi dispiace, ma posso rispondere solo a domande relative a tasse, IVA e questioni fiscali. Posso aiutarti con domande su questi argomenti?"
        assert response == expected_response, "Response should indicate tax-only capability in Italian"
    except ImportError as e:
        pytest.skip(f"Skipping test due to import error: {e}")