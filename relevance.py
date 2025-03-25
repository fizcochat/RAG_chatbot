"""
BERT-based relevance checker for tax-related queries.
This module provides a RelevanceChecker class that can determine if user queries 
are relevant to tax matters, specifically for IVA and Fiscozen topics.
"""

import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F
import numpy as np
import re
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

load_dotenv()


class RelevanceChecker:
    """A class for checking if messages are relevant to tax matters with advanced detection methods"""
    
    def __init__(self, model_path=None):
        """
        Initialize the relevance checker
        
        Args:
            model_path: Path to the model directory. If None, will use default BERT model.
        """
        # Initialize the tokenizer and model
        if model_path and os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            self.tokenizer = BertTokenizer.from_pretrained(model_path)
            self.model = BertForSequenceClassification.from_pretrained(model_path)
        else:
            print("Loading default BERT model")
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.model = BertForSequenceClassification.from_pretrained(
                'bert-base-uncased', num_labels=3
            )
        
        # Mapping from index to topic
        self.topics = {
            0: "IVA",
            1: "Fiscozen",
            2: "Other"
        }
        
        # Move model to device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def preprocess_text(self, text: str) -> str:
        """
        Clean and normalize text for better relevance detection
        
        Args:
            text: The input text to clean
            
        Returns:
            Cleaned and normalized text
        """
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
        
    def check_relevance(self, text: str, tax_threshold: float = 0.6, apply_preprocessing: bool = True) -> Dict[str, Any]:
        """
        Check if the text is relevant to tax matters using combined tax probability
        
        Args:
            text: The text to check
            tax_threshold: Threshold for combined tax-related probabilities
            apply_preprocessing: Whether to preprocess the text before checking
            
        Returns:
            Dictionary with results:
                is_relevant: Whether the text is relevant to tax matters
                topic: The predicted topic ("IVA", "Fiscozen", or "Other")
                confidence: Confidence score
                tax_related_probability: Combined probability of tax classes
                probabilities: Raw class probabilities
        """
        # Ensure model is in evaluation mode
        self.model.eval()
        
        # Optionally preprocess the text
        if apply_preprocessing:
            text = self.preprocess_text(text)
        
        # Tokenize the text
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=1)
            
        # Convert to numpy for easier handling
        probs = probs.cpu().numpy()[0]
        
        # Get the predicted class and confidence
        pred_class = np.argmax(probs)
        confidence = probs[pred_class]
        
        # Calculate the combined probability of tax-related classes (IVA and Fiscozen)
        tax_related_prob = probs[0] + probs[1]  # Sum of IVA and Fiscozen probabilities
        
        # Determine if the query is relevant to tax matters
        # A query is relevant if the combined probability of tax-related classes exceeds the threshold
        is_relevant = tax_related_prob >= tax_threshold
        
        # If it's relevant, determine the specific topic
        if is_relevant:
            # If one tax class has higher probability, use that as the topic
            if probs[0] > probs[1]:
                topic = self.topics[0]  # IVA
                topic_confidence = probs[0]
            else:
                topic = self.topics[1]  # Fiscozen
                topic_confidence = probs[1]
        else:
            topic = self.topics[2]  # Other
            topic_confidence = probs[2]
        
        return {
            "is_relevant": is_relevant,
            "topic": topic,
            "confidence": float(topic_confidence),
            "tax_related_probability": float(tax_related_prob),
            "probabilities": {
                "IVA": float(probs[0]),
                "Fiscozen": float(probs[1]),
                "Other": float(probs[2])
            }
        }
    
    def train_with_data(self, texts, labels, batch_size=16, epochs=3):
        """
        Train the model with custom data
        
        Args:
            texts: List of text samples
            labels: List of corresponding labels (0=IVA, 1=Fiscozen, 2=Other)
            batch_size: Batch size for training
            epochs: Number of training epochs
        """
        from torch.utils.data import DataLoader, TensorDataset
        from transformers import AdamW
        
        # Ensure model is in training mode
        self.model.train()
        
        # Tokenize all texts
        encodings = self.tokenizer(texts, truncation=True, padding=True, return_tensors="pt")
        
        # Create dataset and dataloader
        dataset = TensorDataset(
            encodings["input_ids"], 
            encodings["attention_mask"], 
            torch.tensor(labels)
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Setup optimizer
        optimizer = AdamW(self.model.parameters(), lr=5e-5)
        
        # Training loop
        for epoch in range(epochs):
            total_loss = 0
            
            for batch in dataloader:
                # Get batch and move to device
                b_input_ids, b_attention_mask, b_labels = [b.to(self.device) for b in batch]
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(
                    input_ids=b_input_ids,
                    attention_mask=b_attention_mask,
                    labels=b_labels
                )
                
                # Get loss and perform backprop
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # Print epoch results
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{epochs} - Average Loss: {avg_loss:.4f}")
        
        # Move model back to CPU for saving
        self.model.to(torch.device('cpu'))
        
        print("Training completed!")
    
    def save_model(self, output_dir):
        """
        Save the model and tokenizer to the specified directory
        
        Args:
            output_dir: Directory to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model and tokenizer
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"Model saved to {output_dir}")
    
    def test_with_examples(self, examples: Optional[Dict[str, List[str]]] = None, 
                           tax_threshold: float = 0.6, verbose: bool = True) -> Dict[str, Any]:
        """
        Test the relevance checker with predefined or custom examples
        
        Args:
            examples: Dictionary of examples by category. If None, uses default examples.
            tax_threshold: Threshold for tax-related probability
            verbose: Whether to print detailed results
            
        Returns:
            Dictionary with test results and statistics
        """
        # Default test examples if none provided
        if examples is None:
            examples = {
                "IVA Examples": [
                    "What is the current IVA rate in Italy?",
                    "How do I register for IVA?",
                    "When do I need to pay my IVA taxes?",
                    "I need help with my I.V.A. paperwork",
                    "How much IVA do I need to charge my customers?"
                ],
                "Fiscozen Examples": [
                    "What services does Fiscozen offer?",
                    "How much does Fiscozen charge for tax preparation?",
                    "Can Fiscozen help me with my tax return?",
                    "I want to sign up for Fiscozen",
                    "Does Fisco-Zen work with freelancers?"
                ],
                "Other Tax Examples": [
                    "What's the income tax rate in Italy?",
                    "How do I claim tax deductions?",
                    "When is the tax filing deadline?",
                    "Can you explain the flat tax regime?",
                    "What tax benefits do I get as a freelancer?"
                ],
                "Non-Tax Examples": [
                    "What's the weather like today in Rome?",
                    "Can you recommend a good restaurant?",
                    "How do I book a flight to Milan?",
                    "What's the capital of France?",
                    "Tell me a joke"
                ]
            }
        
        # Store results for summary
        results = {
            "IVA Correct": 0,
            "Fiscozen Correct": 0,
            "Other Tax Correct": 0,
            "Non-Tax Correct": 0,
            "All Results": []
        }
        
        # Test all examples
        if verbose:
            print(f"\n{'='*80}")
            print(f"TESTING RELEVANCE CHECKER WITH TAX THRESHOLD: {tax_threshold}")
            print(f"{'='*80}")
        
        for category, category_examples in examples.items():
            if verbose:
                print(f"\n{'-'*40}")
                print(f"{category}")
                print(f"{'-'*40}")
            
            for example in category_examples:
                # Preprocess the text
                preprocessed = self.preprocess_text(example)
                
                # Check relevance
                result = self.check_relevance(preprocessed, tax_threshold=tax_threshold, apply_preprocessing=False)
                
                # Store the full result with metadata
                full_result = {
                    "query": example,
                    "category": category,
                    "is_relevant": result["is_relevant"],
                    "topic": result["topic"],
                    "confidence": result["confidence"],
                    "tax_probability": result["tax_related_probability"],
                    "probabilities": result["probabilities"]
                }
                results["All Results"].append(full_result)
                
                # Format output for verbose mode
                if verbose:
                    relevance = "✓ RELEVANT" if result["is_relevant"] else "✗ NOT RELEVANT"
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
        
        # Calculate overall accuracy (excluding "Other Tax Examples" since they're ambiguous)
        total_clear_examples = len(examples.get("IVA Examples", [])) + \
                             len(examples.get("Fiscozen Examples", [])) + \
                             len(examples.get("Non-Tax Examples", []))
                             
        correct_clear_examples = results["IVA Correct"] + \
                               results["Fiscozen Correct"] + \
                               results["Non-Tax Correct"]
        
        if total_clear_examples > 0:
            accuracy = (correct_clear_examples / total_clear_examples) * 100
            results["accuracy"] = accuracy
        else:
            results["accuracy"] = 0
        
        # Print summary in verbose mode
        if verbose:
            print(f"\n{'='*80}")
            print("SUMMARY")
            print(f"{'='*80}")
            
            iva_examples = examples.get("IVA Examples", [])
            if iva_examples:
                print(f"IVA Examples: {results['IVA Correct']}/{len(iva_examples)} correct")
            
            fiscozen_examples = examples.get("Fiscozen Examples", [])
            if fiscozen_examples:
                print(f"Fiscozen Examples: {results['Fiscozen Correct']}/{len(fiscozen_examples)} correct")
            
            other_tax_examples = examples.get("Other Tax Examples", [])
            if other_tax_examples:
                print(f"Other Tax Examples (detected as relevant): {results['Other Tax Correct']}/{len(other_tax_examples)}")
            
            non_tax_examples = examples.get("Non-Tax Examples", [])
            if non_tax_examples:
                print(f"Non-Tax Examples (detected as not relevant): {results['Non-Tax Correct']}/{len(non_tax_examples)}")
            
            if total_clear_examples > 0:
                print(f"\nOverall accuracy on clear examples: {accuracy:.2f}%")
            
            print(f"\nUsing tax-related probability threshold: {tax_threshold}")
        
        return results

# Example usage
if __name__ == "__main__":
    # Create the relevance checker
    checker = RelevanceChecker()
    
    # Option 1: Test with individual messages
    print("\nTesting individual messages:")
    test_texts = [
        "What is the current IVA rate in Italy?",
        "Can you help me with my Fiscozen account?",
        "What's the weather like today in Rome?",
        "How do I register for IVA?",
        "What services does Fiscozen offer?",
        "I need to book a flight to Milan"
    ]
    
    for text in test_texts:
        result = checker.check_relevance(text)
        relevance = "Relevant" if result["is_relevant"] else "Not relevant"
        print(f"\nText: '{text}'")
        print(f"Result: {relevance} ({result['topic']}, confidence: {result['confidence']:.4f})")
        print(f"Tax-related probability: {result['tax_related_probability']:.4f}")
        print(f"Probabilities: IVA: {result['probabilities']['IVA']:.4f}, "
              f"Fiscozen: {result['probabilities']['Fiscozen']:.4f}, "
              f"Other: {result['probabilities']['Other']:.4f}")
    
    # Option 2: Run comprehensive tests with built-in examples
    print("\nRunning comprehensive tests:")
    checker.test_with_examples(tax_threshold=0.6)
    
    # Option 3: Try with different threshold
    # print("\nTesting with different threshold:")
    # checker.test_with_examples(tax_threshold=0.5)