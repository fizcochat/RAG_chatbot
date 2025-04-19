#!/usr/bin/env python
"""
Fiscozen Tax Chatbot - Launch Script

This script provides an easy way to launch the Fiscozen Tax Chatbot without
having to remember the streamlit command. It performs the same setup and
initialization as the original main.py but launches the app.py Streamlit app.
"""

import os
import sys
import subprocess

def launch_chatbot():
    """Launch Fiscozen Chatbot using streamlit run app.py"""
    print("\n🔹🔹🔹 FISCOZEN TAX CHATBOT 🔹🔹🔹\n")
    print("Launching the Fiscozen Tax Chatbot...")
    
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
        
        # Launch Streamlit with app.py
        command = [
            sys.executable, 
            "-m", 
            "streamlit", 
            "run",
            app_path,
            "--server.port=8501",
            "--server.address=localhost"
        ]
        
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
    launch_chatbot() 