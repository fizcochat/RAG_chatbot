"""
Individual component tests for the RAG-GPT4 system

This script tests each component of the system individually:
1. FastText relevance checker
2. LangChain query refinement
3. RAG document retrieval
4. GPT-4 integration
5. Translation functionality
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

def test_fasttext_relevance():
    """Test the FastText relevance checker."""
    console.print("\n[bold cyan]Testing FastText Relevance Checker...[/bold cyan]")
    
    try:
        relevance_checker = FastTextRelevanceChecker()
        
        # Test with obvious tax queries
        tax_queries = [
            "Come funziona l'IVA?",
            "Quali sono le scadenze fiscali?",
            "Fiscozen offre servizi di dichiarazione dei redditi?",
            "Quanto costa aprire una partita IVA?"
        ]
        
        # Test with non-tax queries
        non_tax_queries = [
            "Qual è la capitale dell'Italia?",
            "Come si prepara la pasta carbonara?",
            "Quali sono i migliori film del 2023?",
            "Dove posso comprare un computer?"
        ]
        
        console.print("[bold]Tax-related queries:[/bold]")
        for query in tax_queries:
            is_relevant, details = relevance_checker.is_relevant(query)
            score = details.get('final_score', 0)
            icon = "✓" if is_relevant else "✗"
            color = "green" if is_relevant else "red"
            console.print(f"[{color}]{icon} {query} (Score: {score:.2f})[/{color}]")
        
        console.print("\n[bold]Non-tax queries:[/bold]")
        for query in non_tax_queries:
            is_relevant, details = relevance_checker.is_relevant(query)
            score = details.get('final_score', 0)
            icon = "✓" if not is_relevant else "✗"
            color = "green" if not is_relevant else "red"
            console.print(f"[{color}]{icon} {query} (Score: {score:.2f})[/{color}]")
    except Exception as e:
        console.print(f"[bold red]Error: FastText relevance checker could not be initialized: {e}[/bold red]")
        console.print("[yellow]Skipping FastText relevance tests. You may need to train the model first.[/yellow]")
        console.print("[yellow]Run: python fast_text/train_fasttext_model.py[/yellow]")

def test_query_refinement(client):
    """Test the query refinement functionality."""
    console.print("\n[bold cyan]Testing Query Refinement...[/bold cyan]")
    
    # Create conversation history
    conversation = [
        {
            "user": "Come funziona il regime forfettario?",
            "assistant": "Il regime forfettario è un regime fiscale agevolato per le partite IVA con ricavi/compensi fino a 85.000€. Prevede un'imposta sostitutiva del 15% (5% per i primi 5 anni) calcolata sul reddito determinato applicando un coefficiente di redditività variabile in base al codice ATECO. Non si applica l'IVA sulle fatture e ci sono semplificazioni contabili."
        },
        {
            "user": "Quali sono i limiti di fatturato?",
            "assistant": "Il limite di fatturato per il regime forfettario è di 85.000€ annui. Se si supera questa soglia, si esce dal regime nell'anno successivo. Se si supera la soglia di 100.000€, si esce dal regime immediatamente, dovendo applicare l'IVA e la tassazione ordinaria dalla data di superamento."
        }
    ]
    
    # Test follow-up questions
    follow_up_queries = [
        "E per le spese?",
        "Cosa succede se supero il limite?",
        "Posso dedurre i costi?"
    ]
    
    for query in follow_up_queries:
        console.print(f"\n[bold]Original query:[/bold] {query}")
        refined_query = test_utils.query_refiner(conversation, query)
        console.print(f"[green]Refined query:[/green] {refined_query}")

def test_document_retrieval(vectorstore):
    """Test document retrieval from Pinecone."""
    console.print("\n[bold cyan]Testing Pinecone Document Retrieval...[/bold cyan]")
    
    queries = [
        "regime forfettario",
        "scadenze fiscali",
        "dichiarazione dei redditi",
        "fatturazione elettronica"
    ]
    
    for query in queries:
        console.print(f"\n[bold]Query:[/bold] {query}")
        # Test expansion first
        expanded_query = test_utils.expand_query(query)
        console.print(f"[green]Expanded query:[/green] {expanded_query}")
        
        # Retrieve documents
        docs = vectorstore.similarity_search(expanded_query, k=2)
        
        if docs:
            console.print(f"[green]Retrieved {len(docs)} documents[/green]")
            for i, doc in enumerate(docs):
                preview = doc.page_content[:150].replace("\n", " ")
                console.print(f"[bold]Document {i+1}:[/bold] {preview}...")
        else:
            console.print("[red]No documents retrieved[/red]")

def test_translation(client):
    """Test translation functionality."""
    console.print("\n[bold cyan]Testing Translation...[/bold cyan]")
    
    english_texts = [
        "How does the flat tax regime work in Italy?",
        "What are the tax deadlines for VAT payments?",
        "Is it possible to deduct expenses for a freelancer?"
    ]
    
    for text in english_texts:
        console.print(f"\n[bold]English:[/bold] {text}")
        italian = test_utils.translate_to_italian(text, client)
        console.print(f"[green]Italian:[/green] {italian}")
        back_to_english = test_utils.translate_from_italian(italian, client)
        console.print(f"[blue]Back to English:[/blue] {back_to_english}")

def test_hybrid_response(client):
    """Test the hybrid RAG-GPT4 response generation."""
    console.print("\n[bold cyan]Testing Hybrid RAG-GPT4 Response...[/bold cyan]")
    
    queries = [
        "Come funziona il regime forfettario?",
        "Quali sono i vantaggi fiscali per un libero professionista?"
    ]
    
    for query in queries:
        console.print(f"\n[bold]Query:[/bold] {query}")
        response = test_utils.find_match(query)
        console.print(Panel(response, title="Hybrid Response", border_style="green"))

def main():
    """Main function to run all tests."""
    # Load API keys from .env file
    dotenv.load_dotenv()
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    
    if not openai_api_key or not pinecone_api_key:
        console.print("[bold red]Error: API keys not found in .env file[/bold red]")
        console.print("Please ensure you have OPENAI_API_KEY and PINECONE_API_KEY in your .env file")
        return
    
    console.print("[bold green]Initializing services for component testing...[/bold green]")
    
    # Initialize the services
    vectorstore, client = test_utils.initialize_services(
        openai_api_key=openai_api_key, 
        pinecone_api_key=pinecone_api_key
    )
    
    if not vectorstore or not client:
        console.print("[bold red]Error: Failed to initialize services[/bold red]")
        return
    
    console.print("[bold green]Services initialized successfully![/bold green]")
    
    # Run individual component tests
    console.print("[bold blue]Running individual component tests...[/bold blue]")
    
    try:
        # Test FastText relevance
        test_fasttext_relevance()
        
        # Test query refinement
        test_query_refinement(client)
        
        # Test document retrieval
        test_document_retrieval(vectorstore)
        
        # Test translation
        test_translation(client)
        
        # Test hybrid response
        test_hybrid_response(client)
        
        console.print("\n[bold green]All component tests completed successfully![/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]Error during testing: {str(e)}[/bold red]")
        import traceback
        console.print(traceback.format_exc())

if __name__ == "__main__":
    main() 