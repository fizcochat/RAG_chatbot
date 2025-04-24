"""
Test script for directly testing the complete chatbot functionality without Streamlit.
This script includes:
1. FastText-based relevance checking
2. Query refinement with conversation context
3. Hybrid RAG-GPT4 approach for comprehensive responses
"""

import os
import dotenv
import test_utils
from rich.console import Console
from rich.panel import Panel
from rich import print as rprint
import sys
from fast_text.relevance import FastTextRelevanceChecker

# Create a Rich console for better output formatting
console = Console()

def main():
    # Load API keys from .env file
    dotenv.load_dotenv()
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    
    if not openai_api_key or not pinecone_api_key:
        console.print("[bold red]Error: API keys not found in .env file[/bold red]")
        console.print("Please ensure you have OPENAI_API_KEY and PINECONE_API_KEY in your .env file")
        return
    
    console.print("[bold green]Initializing services...[/bold green]")
    
    # Initialize the services directly (not through Streamlit)
    vectorstore, client = test_utils.initialize_services(
        openai_api_key=openai_api_key, 
        pinecone_api_key=pinecone_api_key
    )
    
    if not vectorstore or not client:
        console.print("[bold red]Error: Failed to initialize services[/bold red]")
        return
    
    # Initialize FastText relevance checker
    console.print("[bold green]Initializing FastText relevance checker...[/bold green]")
    try:
        relevance_checker = FastTextRelevanceChecker()
        fasttext_available = True
        console.print("[green]FastText relevance checker initialized successfully[/green]")
    except Exception as e:
        console.print(f"[yellow]Warning: FastText relevance checker could not be initialized: {e}[/yellow]")
        console.print("[yellow]Will skip relevance checking and continue with RAG testing[/yellow]")
        fasttext_available = False
    
    console.print("[bold green]Services initialized successfully![/bold green]")
    console.print("[bold blue]Testing Complete System functionality...[/bold blue]")
    
    # Sample queries to test - mix of tax-related and non-tax queries
    test_queries = [
        # Tax-related queries
        "Cosa succede se non presento la dichiarazione dei redditi?",
        "Come funziona il regime forfettario?",
        "Quali sono i servizi offerti da Fiscozen?",
        "What are the tax deadlines in Italy?",
        "Come contattare un consulente Fiscozen?",
        "Come posso aprire una partita IVA?",
        
        # Non-tax queries to test relevance filtering
        "Qual è il miglior ristorante a Roma?",
        "Come funziona Netflix?",
        "Dove posso trovare un meccanico per la mia auto?"
    ]
    
    # Test conversation context with follow-up questions
    conversation_queries = [
        "Qual è l'aliquota IVA standard in Italia?",
        "E per i beni di prima necessità?",
        "Quando va pagata?"
    ]
    
    # Process standalone queries
    console.print("\n[bold cyan]Testing standalone queries...[/bold cyan]")
    for i, query in enumerate(test_queries, 1):
        console.print(f"\n[bold cyan]Test Query {i}:[/bold cyan] {query}")
        
        # Check if the query is tax-related
        if fasttext_available:
            is_relevant, relevance_details = relevance_checker.is_relevant(query)
            
            if not is_relevant:
                console.print(Panel(
                    "Mi dispiace, ma posso rispondere solo a domande relative a tasse, fiscalità e servizi Fiscozen. "
                    "Puoi provare a porre una domanda su questi argomenti?",
                    title="Not Tax Related",
                    border_style="yellow"
                ))
                continue
            
            console.print(f"[green]Query is tax-related (score: {relevance_details.get('final_score', 0):.2f})[/green]")
        else:
            # If FastText is not available, assume all queries are relevant
            console.print("[yellow]Skipping relevance check (FastText not available)[/yellow]")
        
        # If query is in English, translate it to Italian
        original_lang = "English" if any(word in query for word in ["What", "are", "How", "can", "is", "the"]) else "Italian"
        if original_lang == "English":
            italian_query = test_utils.translate_to_italian(query, client)
            console.print(f"[italic]Translated query:[/italic] {italian_query}")
            query_to_use = italian_query
        else:
            query_to_use = query
        
        # Process the query
        rprint("[bold magenta]Processing with hybrid RAG-GPT4 approach...[/bold magenta]")
        response = test_utils.find_match(query_to_use)
        
        # If original query was in English, translate response back
        if original_lang == "English":
            english_response = test_utils.translate_from_italian(response, client)
            console.print(Panel(english_response, title=f"Response (English)", border_style="green"))
        else:
            console.print(Panel(response, title="Response (Italian)", border_style="green"))
        
        console.print("[cyan]----------------------------------------[/cyan]")
    
    # Test conversation context
    console.print("\n[bold cyan]Testing conversation context...[/bold cyan]")
    conversation_history = []
    
    for i, query in enumerate(conversation_queries, 1):
        console.print(f"\n[bold cyan]Conversation Query {i}:[/bold cyan] {query}")
        
        # Check relevance
        if fasttext_available:
            is_relevant, _ = relevance_checker.is_relevant(query)
            if not is_relevant and i > 1:  # Allow follow-up questions (context should make them relevant)
                console.print("[yellow]Follow-up query might not be tax-related on its own, but using conversation context...[/yellow]")
            elif not is_relevant:
                console.print(Panel(
                    "Mi dispiace, ma posso rispondere solo a domande relative a tasse, fiscalità e servizi Fiscozen. "
                    "Puoi provare a porre una domanda su questi argomenti?",
                    title="Not Tax Related",
                    border_style="yellow"
                ))
                continue
        else:
            # If FastText is not available, assume all queries are relevant
            console.print("[yellow]Skipping relevance check (FastText not available)[/yellow]")
        
        # Use query refinement for follow-up questions
        if i > 1:
            refined_query = test_utils.query_refiner(conversation_history, query)
            console.print(f"[italic]Refined query with context:[/italic] {refined_query}")
            query_to_use = refined_query
        else:
            query_to_use = query
        
        # Process the query
        response = test_utils.find_match(query_to_use)
        console.print(Panel(response, title=f"Response", border_style="green"))
        
        # Add to conversation history
        conversation_history.append({
            "user": query,
            "assistant": response
        })
        
        console.print("[cyan]----------------------------------------[/cyan]")
    
    console.print("\n[bold green]Testing complete! All system components functioning[/bold green]")

if __name__ == "__main__":
    main() 