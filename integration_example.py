"""
Example of how to integrate the relevance checker into your existing chatbot.
This is just an example - modify based on your actual chatbot architecture.
"""

from relevance import RelevanceChecker
from collections import deque

# -----------------------------------------------------
# CONVERSATION TRACKING
# -----------------------------------------------------

class ConversationTracker:
    """Tracks conversation context to detect topic drift over time"""
    
    def __init__(self, window_size=3):
        self.window_size = window_size
        self.messages = deque(maxlen=window_size)
        self.relevance_history = deque(maxlen=window_size)
        self.tax_probabilities = deque(maxlen=window_size)
        self.off_topic_count = 0
        self.total_messages = 0
    
    def add_message(self, message, is_relevant, tax_probability):
        """Add a message to the conversation history"""
        self.messages.append(message)
        self.relevance_history.append(is_relevant)
        self.tax_probabilities.append(tax_probability)
        
        # Update counters
        self.total_messages += 1
        if not is_relevant:
            self.off_topic_count += 1
        
    def consecutive_off_topic(self):
        """Count consecutive off-topic messages (most recent first)"""
        count = 0
        for is_relevant in reversed(list(self.relevance_history)):
            if not is_relevant:
                count += 1
            else:
                break
        return count
    
    def is_conversation_drifting(self):
        """Check if the entire conversation is drifting off-topic"""
        if not self.tax_probabilities:
            return False
            
        # Calculate weighted average (recent messages have more weight)
        weights = [1.0 + (i/len(self.tax_probabilities)) for i in range(len(self.tax_probabilities))]
        weighted_avg = sum(p * w for p, w in zip(self.tax_probabilities, weights)) / sum(weights)
        
        # Calculate off-topic percentage
        if self.total_messages > 0:
            off_topic_percentage = (self.off_topic_count / self.total_messages) * 100
        else:
            off_topic_percentage = 0
        
        # A conversation is drifting if:
        # 1. The weighted average tax probability is low, OR
        # 2. We have a high percentage of off-topic messages
        return weighted_avg < 0.4 or (off_topic_percentage > 50 and self.total_messages >= 3)
    
    def should_redirect(self):
        """Determine if we should redirect the user"""
        consecutive = self.consecutive_off_topic()
        drifting = self.is_conversation_drifting()
        
        # More aggressive redirect logic:
        # Redirect if:
        # 1. We have 2+ consecutive off-topic messages, OR
        # 2. The conversation is drifting AND at least 1 message is off-topic
        return consecutive >= 2 or (drifting and consecutive >= 1)
    
    def reset(self):
        """Reset the conversation tracking"""
        self.messages.clear()
        self.relevance_history.clear()
        self.tax_probabilities.clear()
        self.off_topic_count = 0
        self.total_messages = 0

# Dictionary to store conversation trackers by user ID
conversation_trackers = {}

def get_tracker(user_id):
    """Get or create a conversation tracker for a user"""
    if user_id not in conversation_trackers:
        conversation_trackers[user_id] = ConversationTracker()
    return conversation_trackers[user_id]

# -----------------------------------------------------
# Example integration - MODIFY THIS FOR YOUR CHATBOT
# -----------------------------------------------------

# 1. Initialize the relevance checker in your chatbot setup
relevance_checker = RelevanceChecker(model_path="models/enhanced_bert")  # Updated model path

# 2. In your message processing function (modify to match your architecture)
def process_message(user_message, user_id="default"):
    """
    Process a user message with enhanced off-topic detection.
    
    Args:
        user_message: The message from the user
        user_id: Unique identifier for the user/conversation
        
    Returns:
        Dictionary with response information
    """
    # First check if the message is relevant
    # Lower threshold (0.5) for more aggressive detection
    relevance = relevance_checker.check_relevance(user_message, tax_threshold=0.5)
    
    # Get conversation tracker for this user
    tracker = get_tracker(user_id)
    
    # Add message to conversation history
    tracker.add_message(
        message=user_message,
        is_relevant=relevance['is_relevant'],
        tax_probability=relevance['tax_related_probability']
    )
    
    # Check if we should redirect based on conversation context
    should_redirect = tracker.should_redirect()
    consecutive_off_topic = tracker.consecutive_off_topic()
    is_drifting = tracker.is_conversation_drifting()
    
    # Build detailed warnings for debugging
    warnings = {
        "is_relevant": relevance['is_relevant'],
        "tax_probability": relevance['tax_related_probability'],
        "topic": relevance['topic'],
        "consecutive_off_topic": consecutive_off_topic,
        "is_conversation_drifting": is_drifting,
        "should_redirect": should_redirect
    }
    
    # Handle relevance status
    if should_redirect:
        # Reset conversation tracking after redirection
        tracker.reset()
        
        # Redirect the user
        return {
            'response': "OFF-TOPIC CONVERSATION DETECTED: I notice our conversation has moved away from tax-related topics. I'm specialized in Italian tax and Fiscozen-related matters only. Let me redirect you to a Customer Success Consultant who can help with general inquiries.",
            'warnings': warnings,
            'redirect': True
        }
    elif not relevance['is_relevant']:
        # Message is off-topic but not enough to redirect
        return {
            'response': "OFF-TOPIC DETECTED: I'm specialized in Italian tax matters and Fiscozen services. Could you please ask something related to taxes, IVA, or Fiscozen?",
            'warnings': warnings,
            'redirect': False
        }
    
    # Message is relevant - process normally with your existing pipeline
    topic = relevance['topic']  # Can be 'IVA', 'Fiscozen', or 'Other'
    
    # Use topic to route to appropriate handler
    if topic == 'IVA':
        # Your existing IVA processing logic
        response = handle_iva_query(user_message)
    elif topic == 'Fiscozen':
        # Your existing Fiscozen processing logic
        response = handle_fiscozen_query(user_message)
    else:
        # Your existing general tax processing logic
        response = handle_tax_query(user_message)
    
    # Add conversation drift warning if drifting but not redirecting
    if is_drifting and not should_redirect:
        drift_note = (
            "\n\nNote: Our conversation seems to be moving away from tax topics. "
            "I'm specialized in Italian tax matters and Fiscozen services."
        )
        response += drift_note
    
    return {
        'response': response,
        'topic': topic,
        'confidence': relevance['confidence'],
        'warnings': warnings
    }

# Reset conversation for a user
def reset_conversation(user_id="default"):
    """Reset the conversation tracking for a user"""
    if user_id in conversation_trackers:
        conversation_trackers[user_id].reset()

# -----------------------------------------------------
# REPLACE THESE PLACEHOLDER FUNCTIONS WITH YOUR ACTUAL LOGIC
# -----------------------------------------------------
def handle_iva_query(message):
    return "Regarding IVA (Italian VAT): This is a placeholder response about IVA. Replace with your actual IVA handling logic."

def handle_fiscozen_query(message):
    return "About Fiscozen services: This is a placeholder response about Fiscozen. Replace with your actual Fiscozen handling logic."

def handle_tax_query(message):
    return "This is a placeholder response about general tax matters. Replace with your actual tax query handling logic."

# -----------------------------------------------------
# Example usage
# -----------------------------------------------------
if __name__ == "__main__":
    # Test with a series of messages
    user_id = "test_user"
    messages = [
        "What is the IVA rate in Italy?",
        "How do I register for IVA?",
        "What's the weather like in Rome today?",
        "Can you recommend a good restaurant?",
        "What is Fiscozen?"
    ]
    
    print("Testing conversation-aware off-topic detection:")
    print("-" * 60)
    
    for i, message in enumerate(messages):
        result = process_message(message, user_id)
        
        print(f"\nMessage {i+1}: {message}")
        print(f"Response: {result['response'][:100]}...")
        
        if 'warnings' in result:
            w = result['warnings']
            print(f"Is relevant: {w['is_relevant']}")
            print(f"Topic: {w['topic']}")
            print(f"Tax probability: {w['tax_probability']:.2f}")
            print(f"Consecutive off-topic: {w['consecutive_off_topic']}")
            print(f"Conversation drifting: {w['is_conversation_drifting']}")
            print(f"Redirect needed: {w['should_redirect']}")
        
        if result.get('redirect', False):
            print("ACTION: User redirected to customer support!")
            print("Resetting conversation tracking...")
            # In a real implementation, you would redirect here 