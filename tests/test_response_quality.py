import pytest
import os
from unittest.mock import patch, MagicMock
from main import get_response
import nltk
from nltk.tokenize import sent_tokenize

# Ensure NLTK data is downloaded at import time
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# ==================== FIXTURES ====================

@pytest.fixture(scope="module")
def benchmark_qa_pairs():
    """
    Provides benchmark question-answer pairs with ground truth responses
    for evaluation purposes.
    """
    return [
        {
            "question": "What is IVA in Italy?",
            "ground_truth": [
                "IVA stands for Imposta sul Valore Aggiunto",
                "Italian Value Added Tax",
                "22% standard rate",
                "VAT",
                "consumption tax",
                "indirect tax",
                "added value",
                "applied to goods",
                "services",
                "imports",
                "reduced rates",
                "10%",
                "4%",
                "tax identification number",
                "partita IVA",
                "EU directive",
                "European Union",
                "tax system",
                "fiscal",
                "Italian tax authority"
            ],
            "irrelevant": [
                "corporate tax",
                "income tax",
                "IRAP",
                "personal finance",
                "inheritance tax",
                "property tax"
            ]
        },
        {
            "question": "When should I submit my VAT return?",
            "ground_truth": [
                "quarterly",
                "monthly",
                "deadlines",
                "filing period",
                "VAT settlement",
                "periodic VAT payment",
                "tax calendar",
                "VAT declaration",
                "annual return",
                "electronic submission",
                "F24 form",
                "liquidazione IVA",
                "dichiarazione IVA",
                "tax office",
                "fiscal year",
                "VAT period",
                "reporting obligations",
                "tax compliance",
                "penalties for late filing",
                "annual summary"
            ],
            "irrelevant": [
                "property tax",
                "dividend",
                "capital gains",
                "inheritance tax",
                "corporate structure",
                "IRPEF"
            ]
        },
        {
            "question": "Who can use the flat-rate tax regime in Italy?",
            "ground_truth": [
                "flat rate",
                "forfettario",
                "small business",
                "annual revenue limit",
                "simplified regime",
                "€65,000",
                "sole proprietor",
                "freelancer",
                "self-employed",
                "artisan",
                "professional",
                "income threshold",
                "new business",
                "startup",
                "reduced taxation",
                "substitute tax",
                "15% tax rate",
                "5% for new activities",
                "no VAT charged",
                "simplified accounting",
                "forfeit deduction",
                "coefficient",
                "activity code",
                "ATECO"
            ],
            "irrelevant": [
                "corporations",
                "large enterprises",
                "public companies",
                "multinational",
                "VAT exemption",
                "corporate tax"
            ]
        }
    ]

# ==================== RESPONSE QUALITY TESTS ====================

class TestResponseQuality:
    """Tests for evaluating the quality of responses"""
    
    def test_response_completeness(self, benchmark_qa_pairs):
        """Test that responses contain expected information from ground truth"""
        for qa_pair in benchmark_qa_pairs:
            response = get_response(qa_pair["question"])
            
            # Check that response contains at least some of the ground truth keywords
            ground_truth_matches = sum(1 for keyword in qa_pair["ground_truth"] 
                                      if keyword.lower() in response.lower())
            
            assert ground_truth_matches >= 2, f"Response to '{qa_pair['question']}' should contain at least 2 ground truth elements"
    
    def test_response_relevance(self, benchmark_qa_pairs):
        """Test that responses don't contain irrelevant information"""
        for qa_pair in benchmark_qa_pairs:
            response = get_response(qa_pair["question"])
            
            # Check that response doesn't contain irrelevant keywords
            irrelevant_matches = sum(1 for keyword in qa_pair["irrelevant"] 
                                    if keyword.lower() in response.lower())
            
            assert irrelevant_matches <= 1, f"Response to '{qa_pair['question']}' should not contain irrelevant information"
    
    def test_response_coherence(self):
        """Test that responses are coherent and well-structured"""
        test_questions = [
            "What is IVA?",
            "How do I register for VAT in Italy?",
            "What are the tax deadlines for sole proprietors?"
        ]
        
        for question in test_questions:
            response = get_response(question)
            
            # Split response into sentences
            sentences = sent_tokenize(response)
            
            # Basic coherence checks
            assert len(sentences) >= 2, f"Response to '{question}' should have multiple sentences"
            assert len(response) >= 50, f"Response to '{question}' should be substantive"
            
            # Check for sentence length variation (a sign of natural language)
            sentence_lengths = [len(s) for s in sentences]
            length_variation = max(sentence_lengths) - min(sentence_lengths)
            assert length_variation > 10, f"Response to '{question}' should have natural sentence variation"

# ==================== METADATA TESTS ====================

class TestMetadataHandling:
    """Tests for how the system handles document metadata"""
    
    @patch('main.find_match')
    @patch('main.query_refiner')
    @patch('main.conversation')
    def test_citation_handling(self, mock_conversation, mock_refiner, mock_find):
        """Test how the system handles source citations from metadata"""
        # Mock a response with metadata
        mock_find.return_value = """[Document(page_content='The standard VAT rate in Italy is 22%.', metadata={'source': 'tax_guide_2023.pdf', 'page': 42}), 
                                   Document(page_content='Reduced VAT rates of 10% and 4% apply to specific categories.', metadata={'source': 'vat_regulations.pdf', 'page': 13})]"""
        mock_refiner.return_value = "What is the VAT rate in Italy?"
        mock_conversation.predict.return_value = "The standard VAT rate in Italy is 22%. There are also reduced rates of 10% and 4% for specific categories."
        
        response = get_response("What is the VAT percentage in Italy?")
        
        # Check response contains information but not raw metadata
        assert "22%" in response, "Response should contain the information from documents"
        assert "standard" in response.lower(), "Response should mention standard rate"
        assert "reduced" in response.lower(), "Response should mention reduced rates"
        assert "metadata" not in response.lower(), "Response should not contain raw metadata"
        assert "page_content" not in response.lower(), "Response should not contain raw document structure"

# ==================== COMPARATIVE TESTS ====================

class TestResponseComparison:
    """Tests comparing responses across different input variations"""
    
    @pytest.mark.parametrize(
        "query_variations", [
            ["What is IVA?", "Tell me about IVA", "IVA explanation"],
            ["How to pay taxes in Italy?", "Italian tax payment methods", "Methods to pay taxes in Italy"],
            ["VAT registration process", "How to register for VAT?", "Steps for VAT registration"]
        ]
    )
    def test_response_consistency(self, query_variations):
        """Test that semantically similar questions get consistent responses"""
        responses = [get_response(query) for query in query_variations]
        
        # Extract key information from each response
        keywords = []
        for response in responses:
            # Take words longer than 5 chars as potential keywords
            words = [word.lower() for word in response.split() if len(word) > 5]
            keywords.append(set(words))
        
        # Calculate overlap between responses
        for i in range(len(keywords)):
            for j in range(i+1, len(keywords)):
                intersection = keywords[i].intersection(keywords[j])
                assert len(intersection) >= 5, f"Responses to similar queries should have significant overlap" 