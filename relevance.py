from bert_classifier import BERTQueryClassifier
import os

class RelevanceChecker:
    """A minimal class for checking if messages are relevant to tax matters"""
    
    def __init__(self, model_path=None, threshold=0.6):
        """Initialize the relevance checker"""
        self.classifier = BERTQueryClassifier(threshold=threshold)
        
        # Load model if available
        if model_path and os.path.exists(model_path):
            self.classifier.load_model(model_path)
        
        # Track consecutive off-topic messages
        self.off_topic_count = 0
        self.max_consecutive_off_topic = 3
        
    def check_relevance(self, message):
        """Check if a message is relevant to tax matters"""
        # Get prediction, relevance status and warning
        result, is_relevant, warning = self.classifier.predict_with_warning(message)
        
        # Update off-topic counter
        if not is_relevant:
            self.off_topic_count += 1
        else:
            self.off_topic_count = 0
        
        # Check if we should redirect user
        should_redirect = self.off_topic_count >= self.max_consecutive_off_topic
        
        # Prepare response
        return {
            'is_relevant': is_relevant,
            'warning': warning,
            'should_redirect': should_redirect,
            'topic': result['predicted_class'],
            'confidence': max(result['probabilities'].values()),
            'probabilities': result['probabilities']
        }
        
    def save_model(self, path):
        """Save the trained model"""
        self.classifier.save_model(path)
        
    def train_with_data(self, texts, labels):
        """Train the classifier with provided data"""
        self.classifier.train(texts, labels)