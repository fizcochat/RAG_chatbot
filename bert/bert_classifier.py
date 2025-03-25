import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from typing import Dict, Tuple
from simple_integration import check_message_relevance

class QueryClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class BERTQueryClassifier:
    def __init__(self, num_labels=3, model_name='bert-base-uncased', threshold=0.6):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_labels
        )
        self.model.to(self.device)
        self.threshold = threshold
        self.label_map = {
            0: 'IVA',
            1: 'Fiscozen',
            2: 'Other Tax Matter',
        }

    def train(self, train_texts, train_labels, val_texts=None, val_labels=None, 
              epochs=5, batch_size=16, learning_rate=2e-5):
        
        if val_texts is None or val_labels is None:
            train_texts, val_texts, train_labels, val_labels = train_test_split(
                train_texts, train_labels, test_size=0.2, random_state=42
            )

        train_dataset = QueryClassificationDataset(
            train_texts, 
            train_labels,
            self.tokenizer
        )
        
        val_dataset = QueryClassificationDataset(
            val_texts,
            val_labels,
            self.tokenizer
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False
        )

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            for batch in train_loader:
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss
                total_loss += loss.item()

                loss.backward()
                optimizer.step()

            # Validation
            self.model.eval()
            val_loss = 0
            correct_predictions = 0
            total_predictions = 0

            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)

                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )

                    val_loss += outputs.loss.item()
                    preds = torch.argmax(outputs.logits, dim=1)
                    correct_predictions += (preds == labels).sum().item()
                    total_predictions += labels.shape[0]

            avg_train_loss = total_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            accuracy = correct_predictions / total_predictions

            print(f'Epoch {epoch + 1}:')
            print(f'Average training loss: {avg_train_loss:.4f}')
            print(f'Average validation loss: {avg_val_loss:.4f}')
            print(f'Validation Accuracy: {accuracy:.4f}')

    def predict(self, text):
        self.model.eval()
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=128,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = F.softmax(outputs.logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()

        return {
            'predicted_class': self.label_map[predicted_class],
            'probabilities': {
                self.label_map[i]: prob.item()
                for i, prob in enumerate(probabilities[0])
            }
        }

    def save_model(self, path):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load_model(self, path):
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model.to(self.device)

    def predict_with_warning(self, text: str) -> Tuple[Dict, bool, str]:
        """
        Predict the category of the query and return a warning if it's off-topic.
        """
        result = self.predict(text)
        probabilities = result['probabilities']
        
        # Check if any tax-related category has high confidence
        max_prob = max(probabilities.values())
        is_relevant = max_prob >= self.threshold
        
        warning_message = ""
        if not is_relevant:
            warning_message = (
                "⚠️ Warning: This conversation appears to be unrelated to Fiscozen, "
                "IVA, or tax matters. Please keep queries focused on tax-related topics."
            )
        
        return result, is_relevant, warning_message

def process_message(message, user_id):
    # Check relevance
    relevance = check_message_relevance(message, user_id)
    
    if not relevance['is_relevant']:
        # Handle off-topic message
        if relevance['should_redirect']:
            # Too many off-topic messages in a row
            return "I notice we've gone off-topic. Let me redirect you to general support."
        else:
            # First off-topic message - just warn
            return f"I'm specialized in tax matters. {relevance['warning']}"
    
    # Continue with your existing logic for relevant messages
    # You can also use relevance['topic'] to route to specific handlers
    # (IVA, Fiscozen, or Other Tax Matter)