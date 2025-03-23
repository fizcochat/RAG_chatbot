"""
Test script to demonstrate the improved relevance checker with tax-related probability threshold.
Run this script to see how the enhanced model performs on various examples.
"""

import re
from relevance import RelevanceChecker

def preprocess_text(text):
    """Clean and normalize text for better relevance detection"""
    if not text:
        return ""
        
    # Convert to lowercase
    text = text.lower()
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Replace common abbreviations and variants
    replacements = {
        "iva's": "iva",
        "i.v.a": "iva",
        "i.v.a.": "iva",
        "fiscozen's": "fiscozen",
        "fisco zen": "fiscozen",
        "fisco-zen": "fiscozen",
        "fisco zen's": "fiscozen",
        "v.a.t": "vat",
        "v.a.t.": "vat",
        "partita iva": "partita iva",
        "p. iva": "partita iva",
        "p.iva": "partita iva",
        "imposta sul valore aggiunto": "iva"
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    return text

def test_relevance_checker(model_path="models/enhanced_bert", tax_threshold=0.6):
    """
    Test the improved relevance checker with various examples
    
    Args:
        model_path: Path to the enhanced model
        tax_threshold: Threshold for tax-related probability
    """
    # Create the checker
    checker = RelevanceChecker(model_path=model_path)
    
    # Test examples for each category
    test_examples = {
        "IVA Examples": [
            "What is the current IVA rate in Italy?",
            "How do I register for IVA?",
            "When do I need to pay my IVA taxes?",
            "I need help with my I.V.A. paperwork",
            "How much IVA do I need to charge my customers?",
            "What's the procedure for filing IVA returns?",
            "Is my business required to register for partita IVA?"
        ],
        "Fiscozen Examples": [
            "What services does Fiscozen offer?",
            "How much does Fiscozen charge for tax preparation?",
            "Can Fiscozen help me with my tax return?",
            "I want to sign up for Fiscozen",
            "Does Fisco-Zen work with freelancers?",
            "How do I contact Fiscozen support?",
            "What makes Fiscozen different from traditional accountants?"
        ],
        "Other Tax Examples": [
            "What's the income tax rate in Italy?",
            "How do I claim tax deductions?",
            "When is the tax filing deadline?",
            "Can you explain the flat tax regime?",
            "What tax benefits do I get as a freelancer?",
            "How are capital gains taxed in Italy?",
            "What's the difference between IRPEF and IRES?"
        ],
        "Non-Tax Examples": [
            "What's the weather like today in Rome?",
            "Can you recommend a good restaurant?",
            "How do I book a flight to Milan?",
            "What's the capital of France?",
            "Tell me a joke",
            "What time is it in New York?",
            "How tall is the Eiffel Tower?"
        ],
        "Ambiguous Examples": [
            "I need some help with my documents",
            "Can you assist me with a problem?",
            "I have a question about a form",
            "What's the deadline for submission?",
            "How do I register my business?",
            "What are the requirements for this?",
            "Is there a fee for this service?"
        ]
    }
    
    # Store results for summary
    results = {
        "IVA Correct": 0,
        "Fiscozen Correct": 0,
        "Other Tax Correct": 0,
        "Non-Tax Correct": 0,
        "Ambiguous Results": []
    }
    
    # Test all examples
    print(f"\n{'='*80}")
    print(f"TESTING RELEVANCE CHECKER WITH TAX THRESHOLD: {tax_threshold}")
    print(f"{'='*80}")
    
    for category, examples in test_examples.items():
        print(f"\n{'-'*40}")
        print(f"{category}")
        print(f"{'-'*40}")
        
        for example in examples:
            # Preprocess the text
            preprocessed = preprocess_text(example)
            
            # Check relevance
            result = checker.check_relevance(preprocessed, tax_threshold=tax_threshold)
            
            # Format output
            relevance = "✓ RELEVANT" if result["is_relevant"] else "✗ NOT RELEVANT"
            
            # Print detailed results
            print(f"\nQuery: '{example}'")
            print(f"Result: {relevance} ({result['topic']})")
            print(f"Confidence: {result['confidence']:.4f}")
            print(f"Tax-related probability: {result['tax_related_probability']:.4f}")
            print(f"Class probabilities: IVA: {result['probabilities']['IVA']:.4f}, "
                  f"Fiscozen: {result['probabilities']['Fiscozen']:.4f}, "
                  f"Other: {result['probabilities']['Other']:.4f}")
            
            # Update results for summary
            if category == "IVA Examples" and result["topic"] == "IVA":
                results["IVA Correct"] += 1
            elif category == "Fiscozen Examples" and result["topic"] == "Fiscozen":
                results["Fiscozen Correct"] += 1
            elif category == "Other Tax Examples" and result["is_relevant"]:
                results["Other Tax Correct"] += 1
            elif category == "Non-Tax Examples" and not result["is_relevant"]:
                results["Non-Tax Correct"] += 1
            elif category == "Ambiguous Examples":
                results["Ambiguous Results"].append({
                    "query": example,
                    "is_relevant": result["is_relevant"],
                    "topic": result["topic"],
                    "tax_prob": result["tax_related_probability"]
                })
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"IVA Examples: {results['IVA Correct']}/{len(test_examples['IVA Examples'])} correct")
    print(f"Fiscozen Examples: {results['Fiscozen Correct']}/{len(test_examples['Fiscozen Examples'])} correct")
    print(f"Other Tax Examples (detected as relevant): {results['Other Tax Correct']}/{len(test_examples['Other Tax Examples'])}")
    print(f"Non-Tax Examples (detected as not relevant): {results['Non-Tax Correct']}/{len(test_examples['Non-Tax Examples'])}")
    
    print("\nAmbiguous Examples Results:")
    for i, result in enumerate(results["Ambiguous Results"]):
        print(f"{i+1}. '{result['query']}' → {'Relevant' if result['is_relevant'] else 'Not Relevant'} ({result['topic']}, Tax Prob: {result['tax_prob']:.4f})")

    # Overall accuracy
    total_clear_examples = len(test_examples["IVA Examples"]) + len(test_examples["Fiscozen Examples"]) + \
                         len(test_examples["Non-Tax Examples"])
    correct_clear_examples = results["IVA Correct"] + results["Fiscozen Correct"] + \
                           results["Non-Tax Correct"]
    
    accuracy = (correct_clear_examples / total_clear_examples) * 100
    print(f"\nOverall accuracy on clear examples: {accuracy:.2f}%")
    
    print(f"\n{'='*80}")
    print("CONCLUSIONS")
    print(f"{'='*80}")
    
    # Provide interpretation based on results
    if accuracy >= 85:
        print("The model performs very well across all categories.")
    elif accuracy >= 70:
        print("The model performs adequately but could use more training data.")
    else:
        print("The model needs significant improvement. Consider retraining with more data.")
    
    # Check if there are specific weaknesses
    if results["IVA Correct"] < len(test_examples["IVA Examples"]) * 0.7:
        print("- The model has difficulty identifying IVA-related queries.")
    
    if results["Fiscozen Correct"] < len(test_examples["Fiscozen Examples"]) * 0.7:
        print("- The model has difficulty identifying Fiscozen-related queries.")
    
    if results["Non-Tax Correct"] < len(test_examples["Non-Tax Examples"]) * 0.7:
        print("- The model has difficulty identifying non-tax related queries.")
    
    print(f"\nUsing tax-related probability threshold: {tax_threshold}")
    print("Adjusting this threshold may improve results for your specific use case.")

if __name__ == "__main__":
    # Test with the enhanced model
    test_relevance_checker(model_path="models/enhanced_bert", tax_threshold=0.6)
    
    # Uncomment to test with a different threshold
    # print("\n\nTESTING WITH DIFFERENT THRESHOLD\n")
    # test_relevance_checker(model_path="models/enhanced_bert", tax_threshold=0.5) 