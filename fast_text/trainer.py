import os
import subprocess
import sys

MODEL_PATH = "fast_text/models/tax_classifier.bin"

def train_fasttext_if_needed():
    if os.path.exists(MODEL_PATH):
        print("✅ FastText model already exists.")
        return
    print("🚧 FastText model not found. Training...")

    if not os.path.exists("fast_text/train_with_real_data.py"):
        print("❌ Training script not found.")
        return

    try:
        subprocess.check_call([sys.executable, "fast_text/train_with_real_data.py"])
        print("✅ Training completed.")
    except Exception as e:
        print(f"❌ Failed to train FastText model: {e}")
