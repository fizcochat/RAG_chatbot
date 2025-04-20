#!/usr/bin/env python
"""
Fiscozen Tax Chatbot - Launcher

This script provides an easy way to launch the Fiscozen Tax Chatbot without
having to remember the streamlit command. It performs the same setup and
initialization as the original main.py but launches the app.py Streamlit app.
"""

import os
import sys
import argparse
import subprocess
import platform

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Fiscozen Tax Chatbot")
    parser.add_argument("--train-pinecone", action="store_true", 
                        help="Train the FastText model using data from Pinecone")
    parser.add_argument("--monitor", action="store_true",
                        help="Launch the monitoring dashboard")
    args = parser.parse_args()
    
    # Handle Pinecone training
    if args.train_pinecone:
        from main import train_fasttext_with_pinecone
        train_fasttext_with_pinecone()
        sys.exit(0)
    
    # Import main module
    try:
        from main import launch_chatbot, terminate_streamlit_processes
    except ImportError:
        print("❌ Error importing main module. Make sure you're in the correct directory.")
        sys.exit(1)
    
    # Kill any existing Streamlit processes
    terminate_streamlit_processes()
    
    # Launch the chatbot
    launch_chatbot(monitor_mode=args.monitor)

def launch_chatbot(monitor_mode=False):
    """Launch Fiscozen Chatbot using streamlit run app.py"""
    print("\n🔹🔹🔹 FISCOZEN TAX CHATBOT 🔹🔹🔹\n")
    
    # Check for dependencies
    try:
        import streamlit
        import fasttext
        print("✅ Required dependencies found")
    except ImportError as e:
        missing_module = str(e).split("'")[1]
        print(f"❌ Missing dependency: {missing_module}")
        
        if os.path.exists("install_dependencies.py"):
            print("📦 Running dependency installer...")
            subprocess.check_call([sys.executable, "install_dependencies.py"])
        else:
            print("📦 Installing core dependencies...")
            try:
                # Install core dependencies
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", 
                    "streamlit", "streamlit-chat", "langchain", "langchain-openai", 
                    "langchain-pinecone", "openai", "python-dotenv"
                ])
                print("✅ Core dependencies installed")
            except subprocess.CalledProcessError:
                print("❌ Failed to install dependencies")
                sys.exit(1)
    
    # Terminate any existing Streamlit processes
    try:
        import psutil
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if proc.info['cmdline'] and any('streamlit' in cmd.lower() for cmd in proc.info['cmdline']):
                    print(f"Terminating existing Streamlit process: {proc.info['pid']}")
                    proc.terminate()
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
    except ImportError:
        print("psutil not available, skipping process termination")
    
    # Set environment variables
    os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
    os.environ["FISCOZEN_HIDE_DEMO_WARNING"] = "true"
    os.environ["FISCOZEN_HIDE_AVATARS"] = "true"
    
    try:
        # Get path to the app.py file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        app_path = os.path.join(script_dir, "app.py")
        
        # Check if app.py exists
        if not os.path.exists(app_path):
            print(f"❌ Could not find app.py at {app_path}")
            sys.exit(1)
        
        # Get correct command based on platform
        system = platform.system()
        
        # Launch Streamlit with app.py
        command = [
            sys.executable, 
            "-m", 
            "streamlit", 
            "run",
            app_path,
            "--server.port=8501"
        ]
        
        # Add platform-specific options
        if system == "Windows":
            # On Windows, just use localhost binding
            command.append("--server.address=localhost")
        elif system == "Darwin":  # macOS
            # On macOS, use localhost binding
            command.append("--server.address=localhost")
        else:  # Linux and others
            # On Linux, bind to all interfaces
            command.append("--server.address=0.0.0.0")
        
        print(f"Running command: {' '.join(command)}")
        process = subprocess.Popen(command)
        
        print("\n✅ Fiscozen Tax Chatbot started successfully!")
        print("📱 Access the chatbot at: http://localhost:8501")
        print("\nPress Ctrl+C to quit the application\n")
        
        # Wait for the process to complete
        process.wait()
        
    except KeyboardInterrupt:
        print("\n⚠️ Chatbot terminated by user")
    except Exception as e:
        print(f"\n❌ Error launching the chatbot: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 