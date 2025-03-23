"""
Example of how to integrate the relevance checker into your existing chatbot.
This is just an example - modify based on your actual chatbot architecture.
"""

from relevance import RelevanceChecker

# -----------------------------------------------------
# Example integration - MODIFY THIS FOR YOUR CHATBOT
# -----------------------------------------------------

# 1. Initialize the relevance checker in your chatbot setup
relevance_checker = RelevanceChecker(model_path="models/bert_classifier")  # change path if needed

# 2. In your message processing function (modify to match your architecture)
def process_message(user_message, conversation_state):
    # First check if the message is relevant
    relevance = relevance_checker.check_relevance(user_message)
    
    # Handle relevance status
    if not relevance['is_relevant']:
        # Message is off-topic
        if relevance['should_redirect']:
            # Too many off-topic messages - redirect
            return {
                'response': "I notice we've gone off-topic. Let me redirect you to general support.",
                'warning': relevance['warning'],
                'redirect': True
            }
        else:
            # First or second off-topic message - warn user
            return {
                'response': "I'm specialized in tax matters. Can you ask about taxes, IVA, or Fiscozen?",
                'warning': relevance['warning'],
                'redirect': False
            }
    
    # Message is relevant - process normally with your existing pipeline
    topic = relevance['topic']  # Can be 'IVA', 'Fiscozen', or 'Other Tax Matter'
    
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
    
    return {
        'response': response,
        'topic': topic,
        'confidence': relevance['confidence']
    }

# -----------------------------------------------------
# REPLACE THESE PLACEHOLDER FUNCTIONS WITH YOUR ACTUAL LOGIC
# -----------------------------------------------------
def handle_iva_query(message):
    return "IVA response"

def handle_fiscozen_query(message):
    return "Fiscozen response"

def handle_tax_query(message):
    return "Tax response" 