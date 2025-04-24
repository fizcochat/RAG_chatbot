from langchain_openai import OpenAIEmbeddings
import openai
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain.schema import HumanMessage, AIMessage
import streamlit as st
from monitor.db_logger import log_event

# Constants for Pinecone configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MODEL = "text-embedding-ada-002"
PINECONE_ENV = "gcp-starter"
INDEX = 'ragtest'

# Load the environment variables from the .env file
def initialize_services(openai_api_key, pinecone_api_key):
    """Initialize OpenAI and Pinecone services"""
    # Set OpenAI API key
    openai.api_key = openai_api_key

    # Initialize OpenAI Embeddings model
    model = OpenAIEmbeddings(model=MODEL, openai_api_key=openai.api_key)

    # Initialize Pinecone with API key
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(INDEX)

    # Set up Pinecone VectorStore
    vectorstore = PineconeVectorStore(index, model, "text")

    # Initialize OpenAI client
    client = openai.OpenAI(api_key=openai.api_key)

    return vectorstore, client

def translate_to_italian(text, client=None):
    """Translate text from English to Italian using OpenAI."""
    if not text:
        return ""
        
    try:
        # Get OpenAI client from session state if not provided
        if client is None:
            if 'openai_client' not in st.session_state:
                raise ValueError("OpenAI client not initialized")
            client = st.session_state['openai_client']
        
        # Use OpenAI to translate
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a professional translator. Translate the following text from English to Italian, maintaining the same tone and style. Only return the translated text without any additional commentary."},
                {"role": "user", "content": text}
            ],
            temperature=0.3
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Translation error (to Italian): {e}")
        return text  # Return original text if translation fails

def translate_from_italian(text, client=None):
    """Translate text from Italian to English using OpenAI."""
    if not text:
        return ""
        
    try:
        # Get OpenAI client from session state if not provided
        if client is None:
            if 'openai_client' not in st.session_state:
                raise ValueError("OpenAI client not initialized")
            client = st.session_state['openai_client']
        
        # Use OpenAI to translate
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a professional translator. Translate the following text from Italian to English, maintaining the same tone and style. Only return the translated text without any additional commentary."},
                {"role": "user", "content": text}
            ],
            temperature=0.3
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Translation error (from Italian): {e}")
        return text  # Return original text if translation fails

def find_match(query, k=3):
    """Find similar documents in the vector store and format them into a readable response"""
    try:
        # Check for contact/support queries
        contact_keywords = [
            "contact", "speak with", "talk to", "email", "phone", "expert", "advisor", 
            "consultant", "support", "help", "real person", "contattare", "parlare con", 
            "consulente", "esperto"
        ]
        
        # If this is a contact request, provide standard contact info
        is_contact_query = any(keyword in query.lower() for keyword in contact_keywords) and (
            "fiscozen" in query.lower() or 
            "expert" in query.lower() or 
            "advisor" in query.lower() or
            "consulente" in query.lower() or
            "esperto" in query.lower()
        )
        
        if is_contact_query:
                return ("Per parlare con un consulente Fiscozen, puoi utilizzare uno dei seguenti canali:\n\n"
                       "📧 Email: supporto@fiscozen.it\n"
                       "📞 Telefono: 02 8738 8787\n"
                       "💬 Chat: Visita fiscozen.it e utilizza la chat dal vivo\n\n"
                       "Un consulente fiscale ti ricontatterà il prima possibile per rispondere a tutte le tue domande.")
        
        # Handle specific Fiscozen service queries
        service_keywords = [
            "fiscozen offer", "fiscozen service", "servizi fiscozen", "offre fiscozen", 
            "cosa fa fiscozen", "what does fiscozen", "servizio", "service"
        ]
        
        is_service_query = any(keyword in query.lower() for keyword in service_keywords) or (
            "fiscozen" in query.lower() and any(kw in query.lower() for kw in ["offer", "servizi", "cosa", "what"])
        )
        
        if is_service_query:
                return ("Fiscozen offre un'ampia gamma di servizi fiscali e contabili per professionisti e piccole imprese:\n\n"
                        "📊 **Gestione della contabilità**: Registrazione delle fatture, adempimenti fiscali e contabili\n"
                        "📑 **Dichiarazione dei redditi**: Compilazione e invio della dichiarazione annuale\n"
                        "📱 **Fatturazione elettronica**: Gestione completa delle fatture elettroniche\n"
                        "💼 **Consulenza fiscale**: Supporto personalizzato da parte di consulenti esperti\n"
                        "🧮 **Calcolo delle imposte**: Determinazione delle imposte da versare\n"
                        "📆 **Scadenzario fiscale**: Promemoria per tutte le scadenze fiscali\n"
                        "🚀 **Apertura Partita IVA**: Assistenza completa per l'avvio dell'attività\n\n"
                        "Per maggiori dettagli sui costi e sui piani disponibili, visita il sito fiscozen.it o contatta il supporto.")
        
        # Handle specific electronic invoice queries
        invoice_keywords = [
            "fattur", "invoice", "electronic", "elettronic", "e-fattura", "e-invoice"
        ]
        
        is_invoice_query = any(keyword in query.lower() for keyword in invoice_keywords)
        
        if is_invoice_query and "fiscozen" in query.lower():
                return ("Sì, Fiscozen offre un servizio completo di gestione delle fatture elettroniche che include:\n\n"
                        "✅ **Creazione e invio**: Generazione automatica delle fatture e invio al Sistema di Interscambio (SDI)\n"
                        "✅ **Ricezione**: Gestione delle fatture passive ricevute dai fornitori\n"
                        "✅ **Archivio digitale**: Conservazione a norma di legge di tutte le fatture\n"
                        "✅ **Monitoraggio stato**: Controllo dello stato di consegna delle tue fatture\n"
                        "✅ **Integrazione contabile**: Le fatture vengono automaticamente registrate nella contabilità\n\n"
                        "Il servizio è incluso nei piani Fiscozen e ti permette di gestire facilmente la fatturazione elettronica senza preoccuparti degli aspetti tecnici.")
        
        # Import needed modules
        import openai
        import streamlit as st
        from monitor.db_logger import log_event
        
        # Get vectorstore and OpenAI client
        vectorstore = st.session_state.get('vectorstore')
        if not vectorstore:
            import inspect
            # We're not in a Streamlit app, check if we're in a Flask app with global_store
            calling_frame = inspect.currentframe().f_back
            while calling_frame:
                if 'global_store' in calling_frame.f_locals:
                    vectorstore = calling_frame.f_locals['global_store'].get('vectorstore')
                    openai_client = calling_frame.f_locals['global_store'].get('openai_client')
                    if vectorstore:
                        break
                calling_frame = calling_frame.f_back
                
        if not vectorstore:
            return "Mi dispiace, ma non riesco a connettermi al database delle conoscenze. Riprova più tardi."

        # Get OpenAI client for potential fallback
        client = openai_client
        if not client:
            client = openai.OpenAI(api_key=openai.api_key)
            if not client:
                return "Mi dispiace, non riesco a elaborare la tua richiesta al momento. Riprova più tardi."

        # Perform similarity search with specified k value
        docs = vectorstore.similarity_search(query, k=k)
        
        # Log the retrieval attempt
        chunk_preview = [doc.page_content[:60].replace("\n", " ") for doc in docs] if docs else []
        try:
            log_event("rag_search_attempt", {
            "query": query,
                "retrieved_chunks": len(docs) if docs else 0,
            "chunks_preview": chunk_preview
        })
        except Exception as e:
            print(f"Warning: Could not log event: {e}")
        
        # Generate response using RAG if documents are found
        if docs:
            # Format docs for GPT-4
            context = "\n\n---\n\n".join([f"Documento {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])
            
            # Create system message for GPT-4 with RAG context
            system_message = """Sei un assistente fiscale italiano di Fiscozen. Utilizza le informazioni nei documenti forniti per rispondere alle domande fiscali dell'utente.
            Se i documenti non contengono l'informazione necessaria, dì esplicitamente che ti scuserai di usare la tua conoscenza generale piuttosto che i documenti specifici.
            Utilizza un tono professionale ma amichevole. Fornisci risposte complete ma concise.
            Se la domanda riguarda contattare un consulente Fiscozen, fornisci le informazioni di contatto: Email: supporto@fiscozen.it, Telefono: 02 8738 8787."""
            
            # Generate answer with GPT-4 using RAG context
            try:
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": f"Contesto:\n{context}\n\nDomanda dell'utente: {query}\n\nRispondi in modo esaustivo basandoti sulle informazioni nel contesto se disponibili:"}
                    ],
                    temperature=0.3
                )
                
                rag_response = response.choices[0].message.content
                
                # Check if the RAG response is insufficient (contains phrases like "mi dispiace", "non ho trovato", etc.)
                insufficient_indicators = [
                    "mi dispiace", "non ho trovato", "non contengono informazioni", 
                    "non ho informazioni", "non posso fornire", "non sono in grado",
                    "non sono presenti", "non è presente", "non è menzionato"
                ]
                
                if any(indicator in rag_response.lower() for indicator in insufficient_indicators):
                    # RAG doesn't have the info, use GPT-4 as fallback
                    log_event("rag_insufficient", {"query": query, "response": rag_response[:100]})
                    return generate_gpt4_fallback(query, client)
                
                # RAG has the info, return it
                log_event("rag_success", {"query": query, "response": rag_response[:100]})
                return rag_response
                
            except Exception as e:
                print(f"Error generating GPT-4 response with RAG: {e}")
                # Fall through to GPT-4 fallback
        
        # If no documents found or error in processing RAG, use GPT-4 fallback
        return generate_gpt4_fallback(query, client)
        
    except Exception as e:
        print(f"Error in find_match: {e}")
        return "Mi dispiace, si è verificato un errore durante la ricerca delle informazioni. Riprova più tardi."

def generate_gpt4_fallback(query, client):
    """Generate a response using GPT-4 as fallback when RAG doesn't have relevant information"""
    try:
        log_event("using_gpt4_fallback", {"query": query})
        
        # System message for GPT-4 as fallback
        system_message = """Sei un esperto consulente fiscale italiano che lavora per Fiscozen, una società che offre servizi di consulenza fiscale a liberi professionisti e piccole imprese in Italia.

        INFORMAZIONI SU FISCOZEN:
        - Fiscozen offre servizi di gestione della contabilità, dichiarazione dei redditi, fatturazione elettronica, e consulenza fiscale
        - Si specializza nell'assistenza a liberi professionisti, freelancer e piccole imprese
        - Aiuta con apertura partita IVA, gestione fiscale, e ottimizzazione fiscale
        - Contatti: Email: supporto@fiscozen.it, Telefono: 02 8738 8787

        CONOSCENZE FISCALI ITALIANE:
        - Sei molto esperto del sistema fiscale italiano, incluse le norme su IVA, tassazione, detrazioni e regime forfettario
        - Conosci le leggi fiscali aggiornate al 2023 per l'Italia
        - Sai come funzionano: regime forfettario, regime dei minimi, partita IVA, dichiarazione dei redditi, e fatturazione elettronica

        LINEE GUIDA PER RISPOSTE:
        - Fornisci risposte accurate basate sulla tua conoscenza del sistema fiscale italiano
        - Usa un tono professionale ma accessibile
        - Offri informazioni pratiche e utili
        - Mantieni le risposte concise ma complete
        - Quando appropriato, suggerisci di contattare un consulente Fiscozen per assistenza più specifica
        - Non inventare informazioni

        Rispondi in italiano, a meno che la domanda non sia in inglese."""
        
        # Generate answer with GPT-4 as fallback
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"Domanda dell'utente: {query}\n\nRispondi in modo esaustivo come esperto fiscale:"}
            ],
            temperature=0.5
        )
        
        gpt4_response = response.choices[0].message.content
        
        # Don't explicitly tell the user we're using GPT-4 fallback, just return the response
        return gpt4_response
        
    except Exception as e:
        print(f"Error in GPT-4 fallback: {e}")
        return "Mi dispiace, non ho informazioni sufficienti per rispondere alla tua domanda in questo momento. Puoi contattare il supporto Fiscozen all'indirizzo email supporto@fiscozen.it o al numero 02 8738 8787 per assistenza personalizzata."

def query_refiner(conversation, query):
    """
    Refine the query based on conversation history to handle follow-up questions
    
    Args:
        conversation: List of conversation exchanges in format [{"user": "...", "assistant": "..."}]
        query: The current query to refine
        
    Returns:
        Refined query that incorporates context from conversation history
    """
    try:
        # If no conversation history, return the original query
        if not conversation:
            return query
            
        # Import needed modules
        import openai
        import inspect
        import streamlit as st
        
        # Get OpenAI client
        client = None
        if 'openai_client' in st.session_state:
            client = st.session_state.openai_client
        else:
            # We're not in a Streamlit app, check if we're in a Flask app with global_store
            calling_frame = inspect.currentframe().f_back
            while calling_frame:
                if 'global_store' in calling_frame.f_locals:
                    client = calling_frame.f_locals['global_store'].get('openai_client')
                    if client:
                        break
                calling_frame = calling_frame.f_back
                
        if not client:
            # Use default client if available
            client = openai.OpenAI(api_key=openai.api_key)
            
        # Format conversation history
        conversation_text = ""
        for i, exchange in enumerate(conversation):
            conversation_text += f"Utente: {exchange['user']}\nAssistente: {exchange['assistant']}\n\n"
            
        # Create the system message
        system_message = """Sei un assistente che aiuta a riformulare domande di follow-up in domande autonome e complete.
        Basati sulla cronologia della conversazione per comprendere il contesto e riformulare l'ultima domanda dell'utente.
        Se la domanda è già completa e autonoma, restituiscila invariata.
        Non aggiungere dettagli inventati. Integra solo le informazioni presenti nella cronologia della conversazione."""
        
        # Create the user message
        user_message = f"""Cronologia della conversazione:
        {conversation_text}
        
        Domanda attuale dell'utente: {query}
        
        Riformula questa domanda in una domanda completa e autonoma che includa tutto il contesto necessario dalla conversazione:"""
        
        # Generate the refined query
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Using a smaller model for efficiency
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=0.3
        )
        
        refined_query = response.choices[0].message.content.strip()
        
        # If the refinement failed or produced something odd, return the original
        if not refined_query or len(refined_query) < len(query) / 2 or len(refined_query) > len(query) * 3:
            return query
            
        return refined_query
        
    except Exception as e:
        print(f"Error in query_refiner: {e}")
        return query  # Return original query if refinement fails

def get_conversation_string(conversation_id=None):
    """Get the conversation history as a string"""
    try:
        conversation_string = ""
        if 'responses' in st.session_state and 'requests' in st.session_state:
            for i in range(len(st.session_state['responses'])-1):
                conversation_string += "Human: " + st.session_state['requests'][i] + "\n"
                conversation_string += "Bot: " + st.session_state['responses'][i+1] + "\n"
        return conversation_string
    except Exception as e:
        print(f"Error in get_conversation_string: {e}")
        return ""