"""
Test client for the RAG Chatbot API
This script demonstrates how to call the API from other devices on the network.
"""

import requests
import json
import sys
import argparse
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

# Create a Rich console for better output formatting
console = Console()

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test client for RAG Chatbot API')
    parser.add_argument('--host', type=str, required=True, help='API host address (e.g., 192.168.1.5:8080)')
    parser.add_argument('--session', type=str, default='test_client', help='Session ID for conversation tracking')
    parser.add_argument('--language', type=str, choices=['italian', 'english'], default='italian', 
                       help='Language for communication (default: italian)')
    
    return parser.parse_args()

def health_check(host):
    """Check if the API is running."""
    try:
        response = requests.get(f"http://{host}/api/health")
        if response.status_code == 200:
            data = response.json()
            status = data.get('status')
            components = data.get('components', {})
            
            console.print(f"[bold green]API Status: {status}[/bold green]")
            for component, status in components.items():
                icon = "✓" if status else "✗"
                color = "green" if status else "red"
                console.print(f"[{color}]{icon} {component.upper()}[/{color}]")
            
            return True
        else:
            console.print(f"[bold red]API returned error code: {response.status_code}[/bold red]")
            return False
    except Exception as e:
        console.print(f"[bold red]Error connecting to API: {e}[/bold red]")
        return False

def send_query(host, query, session_id, language):
    """Send a query to the API and return the response."""
    try:
        url = f"http://{host}/api/chat"
        
        # Convert language to language code (it/en)
        language_code = "it"
        if language.lower() == "english":
            language_code = "en"
        
        # Use the new message format
        data = {
            "message": query,
            "session_id": session_id,
            "language": language_code
        }
        
        console.print(f"[dim]Sending request to {url}...[/dim]")
        response = requests.post(url, json=data)
        
        if response.status_code == 200:
            return response.json()
        else:
            console.print(f"[bold red]API returned error code: {response.status_code}[/bold red]")
            if response.text:
                console.print(f"[red]Error message: {response.text}[/red]")
            return None
    except Exception as e:
        console.print(f"[bold red]Error calling API: {e}[/bold red]")
        return None

def clear_conversation(host, session_id):
    """Clear the conversation history for a session."""
    try:
        url = f"http://{host}/api/clear"
        data = {"session_id": session_id}
        
        console.print(f"[dim]Clearing conversation history for session {session_id}...[/dim]")
        response = requests.post(url, json=data)
        
        if response.status_code == 200:
            console.print("[green]Conversation history cleared successfully[/green]")
            return True
        else:
            console.print(f"[bold red]API returned error code: {response.status_code}[/bold red]")
            return False
    except Exception as e:
        console.print(f"[bold red]Error clearing conversation: {e}[/bold red]")
        return False

def interactive_mode(host, session_id, language):
    """Run an interactive chat session with the API."""
    console.print(Panel(
        f"[bold]RAG Chatbot API Client[/bold]\n\n"
        f"Connected to: [green]{host}[/green]\n"
        f"Session ID: [blue]{session_id}[/blue]\n"
        f"Language: [yellow]{language}[/yellow]\n\n"
        "Type your questions or use these commands:\n"
        "  [bold]/quit[/bold] - Exit the client\n"
        "  [bold]/clear[/bold] - Clear conversation history\n"
        "  [bold]/language italian|english[/bold] - Change language",
        title="Interactive Mode",
        border_style="blue"
    ))
    
    # Run interactive loop
    while True:
        query = Prompt.ask("\n[bold blue]You[/bold blue]")
        
        # Handle commands
        if query.lower() == '/quit':
            console.print("[yellow]Exiting...[/yellow]")
            break
        
        elif query.lower() == '/clear':
            clear_conversation(host, session_id)
            continue
        
        elif query.lower().startswith('/language '):
            new_lang = query.lower().split(' ')[1].strip()
            if new_lang in ['italian', 'english']:
                language = new_lang
                console.print(f"[green]Language changed to: {language}[/green]")
            else:
                console.print("[red]Invalid language. Use 'italian' or 'english'[/red]")
            continue
        
        # Send regular query
        result = send_query(host, query, session_id, language)
        
        if result:
            response = result.get('response', 'No response received')
            result_session_id = result.get('session_id', session_id)
            
            # Update the session ID if a new one was returned
            if result_session_id != session_id:
                session_id = result_session_id
                console.print(f"[dim]Session ID updated to: {session_id}[/dim]")
            
            console.print(Panel(
                response,
                title="Response",
                border_style="green"
            ))

def main():
    """Main function to run the test client."""
    args = parse_arguments()
    host = args.host
    session_id = args.session
    language = args.language
    
    # Check if API is running
    console.print(f"[bold]Connecting to RAG Chatbot API at {host}...[/bold]")
    if not health_check(host):
        console.print("[bold red]Could not connect to the API. Exiting...[/bold red]")
        sys.exit(1)
    
    # Run interactive mode
    interactive_mode(host, session_id, language)

if __name__ == "__main__":
    main() 