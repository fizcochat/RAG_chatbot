import pytest
import os
import unittest
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

def test_basic():
    assert True

def test_main_exists():
    # Test if main.py exists
    assert os.path.exists("main.py"), "main.py should exist"

def test_requirements_exists():
    # Test if requirements.txt exists
    assert os.path.exists("requirements.txt"), "requirements.txt should exist"

def test_dockerfile_exists():
    # Test if Dockerfile exists
    assert os.path.exists("Dockerfile"), "Dockerfile should exist"

def test_imports():
    # Test if we can import main modules without errors
    try:
        import main
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import main: {e}")

class TestQueryClassification(unittest.TestCase):
    def setUp(self):
        # Example training data, this is where we should load the data from the csv file given by gloria
        self.queries = [
            "What is my IVA balance?",
            "How do I calculate IVA?",
            "When is my next IVA payment due?",
            "What's Fiscozen's return policy?",
            "How do I contact Fiscozen support?",
            "What are Fiscozen's business hours?",
            # Add more example queries as needed
        ]
        
        self.labels = [
            "IVA",      # 0
            "IVA",      # 1
            "IVA",      # 2
            "Fiscozen",  # 3
            "Fiscozen",  # 4
            "Fiscozen",  # 5
        ]

    def test_query_classification(self):
        # Create and train the model
        model = Pipeline([
            ('vectorizer', TfidfVectorizer()),
            ('classifier', LogisticRegression())
        ])

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            self.queries, 
            self.labels, 
            test_size=0.2, 
            random_state=42
        )

        # Train the model
        model.fit(X_train, y_train)

        # Test some example queries
        test_queries = [
            "What is IVA?",
            "Where is Fiscozen located?",
        ]
        predictions = model.predict(test_queries)

        # Basic assertions to verify the model works
        self.assertIsNotNone(predictions)
        self.assertEqual(len(predictions), len(test_queries))
        
        # Test model accuracy
        accuracy = model.score(X_test, y_test)
        self.assertGreater(accuracy, 0.5)  # Expecting better than random chance

    def test_prediction_probabilities(self):
        # Create and train the model
        model = Pipeline([
            ('vectorizer', TfidfVectorizer()),
            ('classifier', LogisticRegression())
        ])

        model.fit(self.queries, self.labels)

        # Test prediction probabilities
        test_query = "What is IVA tax?"
        proba = model.predict_proba([test_query])[0]

        # Verify we get probability scores
        self.assertEqual(len(proba), len(model.classes_))
        self.assertTrue(all(0 <= p <= 1 for p in proba))
        self.assertAlmostEqual(sum(proba), 1.0)

if __name__ == '__main__':
    unittest.main() 

