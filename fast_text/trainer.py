import os
import subprocess
import sys

MODEL_PATH = "fast_text/models/tax_classifier.bin"

def train_fasttext_if_needed():
    if os.path.exists(MODEL_PATH):
        print("✅ FastText model already exists.")
        return
    print("🚧 FastText model not found. Training...")

    # Use the improved training script
    if os.path.exists("fast_text/train_improved_model.py"):
        try:
            print("📊 Using improved FastText training method...")
            subprocess.check_call([sys.executable, "fast_text/train_improved_model.py"])
            print("✅ Training completed successfully with improved method.")
            return
        except Exception as e:
            print(f"⚠️ Improved training failed: {e}. Trying standard method...")
    
    # Fall back to the original training script if available
    if os.path.exists("fast_text/train_with_real_data.py"):
        try:
            subprocess.check_call([sys.executable, "fast_text/train_with_real_data.py"])
            print("✅ Training completed with standard method.")
        except Exception as e:
            print(f"❌ Failed to train FastText model: {e}")
    else:
        print("❌ Training scripts not found.")
