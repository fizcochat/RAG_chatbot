"""
Test script for the FastText relevance checker.
"""

import logging
from relevance import FastTextRelevanceChecker

# Configure logging
logging.basicConfig(level=logging.INFO)

# Test queries and their expected relevance
TEST_QUERIES = [
    # Tax-related queries (should be relevant)
    ("Come funziona l'IVA per un libero professionista?", True),
    ("Quali sono le aliquote IVA in Italia?", True),
    ("Ho bisogno di aiuto con la dichiarazione dei redditi", True),
    ("Come posso ridurre il carico fiscale della mia attività?", True),
    ("Devo pagare l'IVA sulle fatture estere?", True),
    ("Quali spese posso detrarre come freelancer?", True),
    ("Come funziona il regime forfettario?", True),
    ("Vorrei sapere di più sui servizi di Fiscozen", True),
    
    # Non-tax queries (should not be relevant)
    ("Che tempo farà domani a Roma?", False),
    ("Come si prepara una buona pasta alla carbonara?", False),
    ("Quali sono i migliori ristoranti in città?", False),
    ("Mi puoi consigliare un buon film da vedere?", False),
    ("Come posso migliorare il mio inglese?", False),
    ("Dove posso trovare un idraulico?", False),
    ("Quali sono gli orari del supermercato?", False),
    ("Come si allena un cane?", False)
]

def main():
    print("\n🔹 Testing FastText Relevance Checker 🔹\n")
    
    # Initialize the checker
    model_path = "fast_text/models/tax_classifier.bin"
    print(f"Model path: {model_path}")
    checker = FastTextRelevanceChecker(model_path)
    
    # Test all queries
    print(f"Testing {len(TEST_QUERIES)} queries...\n")
    
    tax_correct = 0
    non_tax_correct = 0
    tax_total = sum(1 for _, is_tax in TEST_QUERIES if is_tax)
    non_tax_total = sum(1 for _, is_tax in TEST_QUERIES if not is_tax)
    
    for query, expected_relevant in TEST_QUERIES:
        print(f"\n📝 Query: {query}")
        try:
            # Get relevance check result and details
            is_relevant, details = checker.is_relevant(query)
            
            # Print details
            print(f"   Preprocessed: {details['preprocessed_text']}")
            print(f"   Keyword score: {details['keyword_score']:.3f}")
            
            if 'model_predictions' in details:
                print("   Model predictions:")
                for label, prob in details['model_predictions'].items():
                    print(f"   - {label}: {prob:.3f}")
            
            if details['keywords_found']:
                print(f"   Keywords found: {details['keywords_found']}")
            if details['phrases_found']:
                print(f"   Phrases found: {details['phrases_found']}")
            
            if 'combined_score' in details:
                print(f"   Combined score: {details['combined_score']:.3f}")
            
            # Print result
            if is_relevant == expected_relevant:
                if expected_relevant:
                    tax_correct += 1
                    print(f"   Result: ✅ Relevant (Expected relevant)")
                else:
                    non_tax_correct += 1
                    print(f"   Result: ✅ Not relevant (Expected not relevant)")
            else:
                if expected_relevant:
                    print(f"   Result: ❌ Not relevant (Expected relevant)")
                else:
                    print(f"   Result: ❌ Relevant (Expected not relevant)")
                    
        except Exception as e:
            print(f"\n❌ Error processing query: {e}")
    
    # Print statistics
    print("\n📊 Results:")
    tax_accuracy = (tax_correct / tax_total * 100) if tax_total > 0 else 0
    non_tax_accuracy = (non_tax_correct / non_tax_total * 100) if non_tax_total > 0 else 0
    overall_accuracy = ((tax_correct + non_tax_correct) / len(TEST_QUERIES) * 100)
    
    print(f"Tax-related queries accuracy: {tax_accuracy:.1f}%")
    print(f"Non-tax queries accuracy: {non_tax_accuracy:.1f}%")
    print(f"Overall accuracy: {overall_accuracy:.1f}%")

if __name__ == "__main__":
    main() 