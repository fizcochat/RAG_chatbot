#!/usr/bin/env python3
"""
Run FastText training

This script provides a command-line interface to train the FastText model.
It allows the user to choose between the standard and improved training methods.

NOTE: This is a maintenance utility script for retraining the model when needed.
      It is not required for normal operation of the chatbot.
"""

import os
import sys
import argparse

def main():
    print("📢 MAINTENANCE UTILITY - FastText Model Training")
    print("⚠️  This script is for retraining the FastText model and is not required for normal chatbot operation.")
    print("⚠️  Training can take several minutes and requires significant memory.")
    print("")
    
    parser = argparse.ArgumentParser(description="Train the FastText classifier for the Fiscozen tax chatbot")
    parser.add_argument(
        "--method", 
        choices=["improved"], 
        default="improved",
        help="Training method to use: 'improved' (default)"
    )
    parser.add_argument(
        "--force", 
        action="store_true",
        help="Force retraining even if model already exists"
    )
    
    args = parser.parse_args()
    
    # Check if model already exists
    model_path = "fast_text/models/tax_classifier.bin"
    if os.path.exists(model_path) and not args.force:
        print("✅ FastText model already exists.")
        print("Use --force to retrain the model anyway.")
        return 0
        
    # If forcing retraining and model exists, remove it
    if args.force and os.path.exists(model_path):
        print("🔄 Removing existing model for retraining...")
        try:
            os.remove(model_path)
        except Exception as e:
            print(f"⚠️ Warning: Could not remove existing model: {e}")
    
    # Use improved training method
    script_path = "fast_text/train_improved_model.py"
    if not os.path.exists(script_path):
        print(f"❌ Training script not found at {script_path}")
        return 1
        
    print("🚀 Starting FastText training...")
    try:
        import subprocess
        result = subprocess.run([sys.executable, script_path], check=True)
        if result.returncode == 0:
            print("✅ FastText model trained successfully!")
            return 0
        else:
            print(f"❌ Training failed with return code {result.returncode}")
            return result.returncode
    except Exception as e:
        print(f"❌ Error during training: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 