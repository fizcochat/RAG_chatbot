import os
import sys
import fasttext
import tempfile
import re
from dotenv import load_dotenv
from tqdm import tqdm
import random

def train_with_pinecone():
    """Train the FastText model using data from Pinecone vector database"""
    print("\n=== Training FastText Model with Pinecone Data ===\n")
    
    # Load environment variables
    load_dotenv()
    pinecone_api_key = os.getenv('PINECONE_API_KEY')
    
    if not pinecone_api_key:
        print("❌ Error: PINECONE_API_KEY not found in environment variables")
        print("Please set your Pinecone API key in the .env file")
        return False
    
    try:
        from pinecone import Pinecone
        
        # Initialize Pinecone client
        print("Connecting to Pinecone...")
        pc = Pinecone(api_key=pinecone_api_key)
        
        # Connect to the index
        index_name = "fiscozen"  # Use the index name you created
        print(f"Connecting to index '{index_name}'...")
        
        # Check if index exists
        try:
            index = pc.Index(index_name)
            # Test a simple query to verify connection
            test_vector = [0.1] * 3072  # 3072 dimensions for text-embedding-3-large
            results = index.query(vector=test_vector, top_k=1, include_metadata=True)
            
            if not results.matches:
                print("⚠️ Warning: No vectors found in the index. The index might be empty.")
                choice = input("Do you want to continue anyway? (y/n): ")
                if choice.lower() != 'y':
                    return False
            else:
                print(f"✅ Successfully connected to Pinecone index with {results.matches[0].score:.2f} similarity score")
                
        except Exception as e:
            print(f"❌ Error connecting to Pinecone index: {e}")
            print(f"Please make sure the index '{index_name}' exists and your API key has access to it.")
            return False
            
        # Create temporary files for training data
        print("Preparing training data files...")
        tax_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.txt')
        other_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.txt')
        
        # Function to clean text for FastText training
        def clean_text(text):
            # Remove special characters and excessive whitespace
            text = re.sub(r'[^\w\s]', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip().lower()
            return text
        
        # Get tax-related examples from Pinecone
        print("Fetching tax-related examples from Pinecone...")
        
        # Get sample IDs from the index 
        # Using query with a neutral vector to get random samples
        # We'll retrieve batches to get a good variety of data
        samples_count = 0
        batch_size = 100
        neutral_vector = [0.01] * 3072  # Nearly neutral vector in all dimensions
        
        # Tax-related keywords to filter relevant content
        tax_keywords = ["iva", "fiscale", "tasse", "fattura", "partita iva", "dichiarazione", 
                       "detrazioni", "forfettario", "fiscozen", "agenzia entrate"]
        
        # Progress bar
        pbar = tqdm(total=500, desc="Collecting tax examples")
        
        # Collect multiple samples with different queries to get variety
        for _ in range(5):  # Multiple queries to get diverse samples
            # Slightly randomize the vector for variety
            query_vector = [0.01 + random.uniform(-0.005, 0.005) for _ in range(3072)]
            
            results = index.query(
                vector=query_vector,
                top_k=batch_size,
                include_metadata=True
            )
            
            for match in results.matches:
                if "text" in match.metadata:
                    text = match.metadata["text"]
                    
                    # Keep only examples with tax-related content
                    text_lower = text.lower()
                    if any(keyword in text_lower for keyword in tax_keywords):
                        cleaned_text = clean_text(text)
                        if len(cleaned_text.split()) > 5:  # Ensure text has enough words
                            tax_file.write(f"__label__Tax {cleaned_text}\n")
                            samples_count += 1
                            pbar.update(1)
                            
                    if samples_count >= 500:  # Limit to a reasonable number of examples
                        break
            
            if samples_count >= 500:
                break
                
        pbar.close()
        
        # If we couldn't get enough examples, warn the user
        if samples_count < 50:
            print(f"⚠️ Warning: Only found {samples_count} tax-related examples in Pinecone.")
            print("This might not be enough for effective training.")
            choice = input("Do you want to continue anyway? (y/n): ")
            if choice.lower() != 'y':
                # Clean up temporary files
                os.unlink(tax_file.name)
                os.unlink(other_file.name)
                return False
        else:
            print(f"✅ Collected {samples_count} tax-related examples from Pinecone")
        
        # Generate non-tax examples to balance the dataset
        print("Generating non-tax examples...")
        
        # General topics for non-tax examples
        topics = [
            "weather", "sports", "food", "travel", "movies", "music", "technology",
            "health", "education", "politics", "environment", "science", "art",
            "fashion", "cars", "animals", "gardening", "history", "literature"
        ]
        
        general_questions = [
            "What is the capital of {}?",
            "How do I cook {}?",
            "When was {} invented?",
            "Who is the best {} player?",
            "Why is {} important?",
            "Can you recommend a good {}?",
            "What's the best time to visit {}?",
            "How do I learn {}?",
            "What are the benefits of {}?",
            "Is {} worth the money?"
        ]
        
        countries = ["Italy", "France", "Spain", "Germany", "USA", "Japan", "Brazil", "Australia"]
        cities = ["Rome", "Paris", "Madrid", "Berlin", "New York", "Tokyo", "Rio", "Sydney"]
        
        # Generate non-tax examples to match the number of tax examples (for balance)
        for _ in range(samples_count):
            topic = random.choice(topics)
            template = random.choice(general_questions)
            example = template.format(topic)
            other_file.write(f"__label__Other {clean_text(example)}\n")
            
            # Add some country/city specific questions too
            if random.random() < 0.3:  # 30% chance
                country = random.choice(countries)
                city = random.choice(cities)
                example = f"What's the weather like in {city}, {country}?"
                other_file.write(f"__label__Other {clean_text(example)}\n")
        
        tax_file.close()
        other_file.close()
        
        # Combine files for training
        print("Combining files for training...")
        combined_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.txt')
        with open(tax_file.name, 'r') as f:
            combined_file.write(f.read())
        with open(other_file.name, 'r') as f:
            combined_file.write(f.read())
        combined_file.close()
        
        # Train FastText model
        print("\nTraining FastText model...")
        os.makedirs("fast_text/models", exist_ok=True)
        model_path = "fast_text/models/tax_classifier.bin"
        
        try:
            # Train the model
            model = fasttext.train_supervised(
                input=combined_file.name,
                lr=0.5,
                epoch=25,
                wordNgrams=2,
                bucket=200000,
                dim=100,
                loss='softmax'
            )
            
            # Save the model
            model.save_model(model_path)
            print(f"✅ Model successfully trained and saved to {model_path}")
            
            # Test some examples
            print("\nTesting model with examples:")
            test_examples = [
                "Come funziona l'IVA?",
                "Quali spese posso dedurre?",
                "Quando devo presentare la dichiarazione dei redditi?",
                "Qual è la differenza tra regime forfettario e ordinario?",
                "What's the weather like today?",
                "Who won the world cup?",
                "Can you recommend a good restaurant?"
            ]
            
            for text in test_examples:
                prediction = model.predict(text)
                label = prediction[0][0].replace("__label__", "")
                confidence = prediction[1][0]
                print(f"Text: '{text}'")
                print(f"Prediction: {label} (Confidence: {confidence:.4f})")
                print("----")
            
            # Clean up temporary files
            os.unlink(combined_file.name)
            os.unlink(tax_file.name)
            os.unlink(other_file.name)
            
            return True
            
        except Exception as e:
            print(f"❌ Error training FastText model: {e}")
            
            # Clean up temporary files
            os.unlink(combined_file.name)
            os.unlink(tax_file.name)
            os.unlink(other_file.name)
            
            return False
            
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("Please install required packages: pip install pinecone-client fasttext tqdm")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = train_with_pinecone()
    if success:
        print("\n✅ FastText model successfully trained with Pinecone data!")
    else:
        print("\n❌ Failed to train FastText model with Pinecone data.")
        sys.exit(1)
