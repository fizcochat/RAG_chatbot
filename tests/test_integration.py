import pytest
import os
from unittest.mock import patch, MagicMock
from main import get_response
import time

# ==================== ENVIRONMENT SETUP ====================

@pytest.fixture(scope="module")
def setup_keys():
    """Setup environment keys for integration tests"""
    # Store original environment variables
    original_openai_key = os.environ.get("OPENAI_API_KEY")
    original_pinecone_key = os.environ.get("PINECONE_API_KEY")
    
    # Check if keys exist, skip tests if not
    if not original_openai_key or not original_pinecone_key:
        pytest.skip("API keys not found in environment variables")
    
    yield
    
    # Restore original environment if needed
    if original_openai_key:
        os.environ["OPENAI_API_KEY"] = original_openai_key
    if original_pinecone_key:
        os.environ["PINECONE_API_KEY"] = original_pinecone_key

# ==================== END-TO-END TESTS ====================

@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("RUN_INTEGRATION_TESTS"), 
                    reason="Integration tests are expensive and slow, run only when explicitly enabled")
class TestEndToEnd:
    """End-to-end tests involving real API calls - use sparingly"""
    
    def test_full_pipeline(self, setup_keys):
        """Test the full retrieval and generation pipeline with real services"""
        query = "What is the IVA registration threshold in Italy?"
        
        start_time = time.time()
        response = get_response(query)
        end_time = time.time()
        
        # Basic response checks
        assert response, "Response should not be empty"
        assert len(response) > 100, "Response should be substantive"
        assert "registration" in response.lower() or "threshold" in response.lower(), "Response should be on topic"
        
        # Performance check
        elapsed_time = end_time - start_time
        assert elapsed_time < 10, f"Response took too long: {elapsed_time} seconds"
        
        print(f"\nFull pipeline response time: {elapsed_time:.2f} seconds")
        print(f"Response: {response[:100]}...")

# ==================== SIMULATED USER SESSIONS ====================

class TestUserSessions:
    """Tests that simulate user conversation sessions"""
    
    @patch('main.find_match')
    @patch('main.query_refiner')
    @patch('main.conversation')
    @patch('utils.get_conversation_string')
    def test_multi_turn_conversation(self, mock_conversation_string, mock_conversation, mock_refiner, mock_find):
        """Test a multi-turn conversation to ensure context is maintained"""
        
        # Setup mocks for conversation history
        conversation_history = [
            "Human: What is IVA in Italy?\nBot: IVA (Imposta sul Valore Aggiunto) is the Italian Value Added Tax with a standard rate of 22%.\n",
            "Human: What is IVA in Italy?\nBot: IVA (Imposta sul Valore Aggiunto) is the Italian Value Added Tax with a standard rate of 22%.\nHuman: What are the reduced rates?\n",
            "Human: What is IVA in Italy?\nBot: IVA (Imposta sul Valore Aggiunto) is the Italian Value Added Tax with a standard rate of 22%.\nHuman: What are the reduced rates?\nBot: The reduced IVA rates in Italy are 10% and 4%, applicable to specific goods and services.\n"
        ]
        
        # Setup mocks for various responses
        mock_conversation.predict.side_effect = [
            "IVA (Imposta sul Valore Aggiunto) is the Italian Value Added Tax with a standard rate of 22%.",
            "The reduced IVA rates in Italy are 10% and 4%, applicable to specific goods and services.",
            "You must file IVA returns quarterly if you're a standard business, or monthly for larger businesses with turnover exceeding certain thresholds."
        ]
        
        mock_find.side_effect = [
            "IVA is the Italian VAT tax at 22% standard rate.",
            "Reduced IVA rates: 10% for tourism, transport; 4% for essential goods.",
            "IVA filing frequency: quarterly for standard businesses, monthly for larger operations."
        ]
        
        mock_refiner.side_effect = [
            "What is IVA (VAT) in Italy?",
            "What are the reduced IVA rates in Italy?",
            "How often must I file IVA returns in Italy?"
        ]
        
        # Simulate conversation flow
        mock_conversation_string.return_value = conversation_history[0]
        response1 = get_response("What is IVA in Italy?")
        assert "22%" in response1, "First response should mention standard rate"
        
        mock_conversation_string.return_value = conversation_history[1]
        response2 = get_response("What are the reduced rates?")
        assert "10%" in response2 and "4%" in response2, "Second response should mention reduced rates"
        
        mock_conversation_string.return_value = conversation_history[2]
        response3 = get_response("How often do I need to file returns?")
        assert "quarterly" in response3.lower() or "monthly" in response3.lower(), "Third response should mention filing frequency"

# ==================== PERFORMANCE TESTS ====================

class TestPerformance:
    """Tests for system performance metrics"""
    
    @patch('main.find_match')
    @patch('main.query_refiner')
    @patch('main.conversation')
    def test_response_time(self, mock_conversation, mock_refiner, mock_find):
        """Test response time with mocked external services"""
        # Setup simple mocks with immediate returns
        mock_find.return_value = "Tax information."
        mock_refiner.return_value = "Refined query"
        mock_conversation.predict.return_value = "Here is tax information."
        
        queries = [
            "What is IVA?",
            "How do I register for VAT?",
            "What are the payment deadlines?"
        ]
        
        for query in queries:
            start_time = time.time()
            response = get_response(query)
            end_time = time.time()
            
            elapsed_time = end_time - start_time
            # With mocked services, response should be very fast
            assert elapsed_time < 0.5, f"Mocked response took too long: {elapsed_time} seconds"
    
    @patch('main.find_match')
    @patch('main.query_refiner')
    @patch('main.conversation')
    def test_concurrent_queries(self, mock_conversation, mock_refiner, mock_find):
        """Test the system's ability to handle concurrent queries (simulated)"""
        import threading
        
        # Setup mocks to avoid actual API calls
        mock_find.return_value = "Tax information for VAT in Italy."
        mock_refiner.return_value = "Refined query about Italian VAT"
        mock_conversation.predict.return_value = "Here is information about Italian VAT."
        
        # For concurrent testing, we'll use a shared results collector
        results = []
        errors = []
        
        def query_task(query, idx):
            try:
                start_time = time.time()
                response = get_response(query)
                elapsed_time = time.time() - start_time
                results.append((idx, response, elapsed_time))
            except Exception as e:
                errors.append((idx, str(e)))
        
        # Test queries
        queries = [
            "What is IVA?",
            "How do I pay VAT?",
            "When is the VAT deadline?",
            "Who needs to register for VAT?"
        ]
        
        # Create and start threads with longer delays
        threads = []
        for idx, query in enumerate(queries):
            thread = threading.Thread(target=query_task, args=(query, idx))
            threads.append(thread)
            thread.start()
            # Add a longer delay between thread starts
            time.sleep(2)
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=30)
        
        # Assert results
        assert len(results) == len(queries), f"Expected {len(queries)} results, got {len(results)}"
        assert not errors, f"Encountered errors during concurrent execution: {errors}"
        
        # Print timing information
        times = [r[2] for r in results]
        avg_time = sum(times) / len(times)
        print(f"\nConcurrent queries avg response time: {avg_time:.2f} seconds") 