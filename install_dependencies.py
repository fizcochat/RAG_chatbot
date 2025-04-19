#!/usr/bin/env python
"""
Cross-platform dependency installer for FiscoChat
This script installs all required dependencies for FiscoChat,
handling platform-specific requirements automatically.
"""

import sys
import platform
import subprocess
import os

def install_dependencies():
    """Install dependencies based on platform detection"""
    print("🔍 Detecting platform...")
    system = platform.system()
    machine = platform.machine()
    
    print(f"Detected: {system} on {machine}")
    
    # Base dependencies that work across platforms
    base_deps = [
        "python-dotenv",
        "streamlit",
        "streamlit-chat",
        "openai",
        "pinecone-client",
        "langchain",
        "langchain-core",
        "langchain-openai",
        "langchain-pinecone",
        "pandas",
        "numpy",
        "pydantic",
        "tqdm",
        "PyPDF2"
    ]
    
    # Install base dependencies
    print("📦 Installing base dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade"] + base_deps)
    
    # FastText installation (platform-specific)
    print("🧠 Installing FastText...")
    
    if system == "Darwin":  # macOS
        if "arm64" in machine:  # Apple Silicon
            print("🍎 Detected Apple Silicon Mac")
            try:
                # Try official fasttext package first
                subprocess.check_call([sys.executable, "-m", "pip", "install", "fasttext"])
            except subprocess.CalledProcessError:
                # Fallback
                subprocess.check_call([sys.executable, "-m", "pip", "install", "fasttext-wheel", "--no-build-isolation"])
        else:  # Intel Mac
            print("🍎 Detected Intel Mac")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "fasttext"])
            except subprocess.CalledProcessError:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "fasttext-wheel"])
    
    elif system == "Windows":
        print("🪟 Detected Windows")
        try:
            # Windows often needs specific build tools
            print("Installing pybind11 first...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pybind11"])
            subprocess.check_call([sys.executable, "-m", "pip", "install", "fasttext"])
        except subprocess.CalledProcessError:
            print("⚠️ FastText installation failed. The chatbot will fall back to keyword-based filtering.")
            print("For full functionality, please install Build Tools for Visual Studio with C++ support")
    
    elif system == "Linux":
        print("🐧 Detected Linux")
        try:
            # Ensure build essentials are installed
            print("Make sure you have build-essential packages installed on your system")
            print("On Ubuntu/Debian: sudo apt-get install build-essential python3-dev")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "fasttext"])
        except subprocess.CalledProcessError:
            print("⚠️ FastText installation failed. Trying alternative methods...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "fasttext-wheel"])
    
    # Install any remaining requirements
    print("📑 Installing remaining dependencies from requirements.txt...")
    if os.path.exists("requirements.txt"):
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    print("✅ Dependencies installed successfully!")
    print("You can now run the chatbot with: streamlit run app.py")

if __name__ == "__main__":
    install_dependencies() 