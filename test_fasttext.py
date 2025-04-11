#!/usr/bin/env python
"""
Simple test to verify that fasttext is installed correctly
and that train_supervised is available.
"""

import sys
import os

# Remove the current directory from sys.path to avoid importing local fasttext module
cwd = os.getcwd()
if cwd in sys.path:
    sys.path.remove(cwd)

# Try to import fasttext from site-packages
print(f"Python version: {sys.version}")
print(f"sys.path: {sys.path}")

try:
    # Try to import fasttext directly
    print("Trying to import fasttext...")
    import fasttext
    print(f"Imported fasttext from: {fasttext.__file__}")
    
    # Try to get version if available
    version = getattr(fasttext, "__version__", "Not available")
    print(f"FastText version: {version}")

    print("FastText module attributes:", dir(fasttext))

    if hasattr(fasttext, "train_supervised"):
        print("✅ train_supervised method is available")
    else:
        print("❌ train_supervised method is NOT available")

    if hasattr(fasttext, "load_model"):
        print("✅ load_model method is available")
    else:
        print("❌ load_model method is NOT available")

    # Try creating a simple model
    try:
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
            temp_file.write("__label__hello world hello\n")
            temp_file.write("__label__bye world goodbye\n")
            temp_file_path = temp_file.name
        
        # Try to train
        print("Attempting to train a simple model...")
        model = fasttext.train_supervised(input=temp_file_path)
        print("✅ Model training successful")
        
        # Test prediction
        result = model.predict("hello world")
        print(f"Prediction result: {result}")
        
    except Exception as e:
        print(f"❌ Error while testing model training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        if 'temp_file_path' in locals():
            try:
                os.unlink(temp_file_path)
            except:
                pass
except Exception as e:
    print(f"Error importing fasttext: {e}")
    import traceback
    traceback.print_exc() 