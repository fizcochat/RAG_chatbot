"""
Modified utils.py file for testing purposes that directly initializes services
without relying on Streamlit session state.
"""

from langchain_openai import OpenAIEmbeddings
import openai
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain.schema import HumanMessage, AIMessage

def generate_local_fallback(query):
    """Generate a simple fallback response when OpenAI API is unavailable."""
    # Dictionary of common tax terms and simple definitions
    tax_knowledge = {
        "regime forfettario": "Il regime forfettario è un regime fiscale agevolato per partite IVA con ricavi/compensi sotto certe soglie. Prevede un'imposta sostitutiva del 5% o 15% sul reddito calcolato applicando un coefficiente di redditività ai ricavi.",
        "partita iva": "La Partita IVA è un codice identificativo necessario per esercitare attività d'impresa, arte o professione. Può essere aperta in diversi regimi fiscali.",
        "iva": "L'IVA (Imposta sul Valore Aggiunto) è un'imposta indiretta sui consumi applicata sulla cessione di beni e prestazione di servizi. In Italia l'aliquota ordinaria è del 22%, con aliquote ridotte al 10%, 5% e 4% per specifiche categorie.",
        "fatturazione": "La fatturazione è l'emissione di documenti fiscali obbligatori per attestare vendite o prestazioni professionali. Dal 2019 la fatturazione elettronica è obbligatoria per la maggior parte delle operazioni.",
        "fiscozen": "Fiscozen è una piattaforma digitale che offre servizi di gestione fiscale per partite IVA, inclusi contabilità, dichiarazione dei redditi, fatturazione elettronica e consulenza fiscale.",
        "detrazioni": "Le detrazioni fiscali sono importi che possono essere sottratti direttamente dall'imposta dovuta, riducendo l'ammontare da versare.",
        "deduzioni": "Le deduzioni fiscali sono spese che possono essere sottratte dal reddito imponibile prima del calcolo dell'imposta.",
        "coefficiente di redditività": "Nel regime forfettario, è la percentuale applicata ai ricavi/compensi per determinare il reddito imponibile. Varia dal 40% all'86% in base al codice ATECO dell'attività.",
        "dichiarazione dei redditi": "La dichiarazione dei redditi è il documento con cui contribuenti comunicano all'Agenzia delle Entrate i redditi percepiti, le ritenute, e le detrazioni per determinare l'imposta dovuta."
    }
    
    # Prepare lowercase version of the query for matching
    query_lower = query.lower()
    
    # Determine which topics might be in the query
    relevant_topics = []
    for topic, explanation in tax_knowledge.items():
        if topic in query_lower:
            relevant_topics.append((topic, explanation))
    
    if relevant_topics:
        # If we found relevant topics, construct a response
        response = "Ecco alcune informazioni che potrebbero esserti utili:\n\n"
        for topic, explanation in relevant_topics:
            response += f"**{topic.capitalize()}**: {explanation}\n\n"
        response += "Per informazioni più dettagliate ti consiglio di consultare un consulente fiscale."
        return response
    else:
        # Generic response when no specific topics are detected
        return ("Mi dispiace, al momento sto riscontrando problemi di connessione con il sistema di risposta avanzato. "
                "Posso fornirti solo informazioni di base sui temi fiscali italiani come regime forfettario, "
                "partita IVA, fatturazione elettronica e servizi di Fiscozen. "
                "Puoi riprovare più tardi o riformulare la tua domanda in modo più specifico.")

# Constants for Pinecone configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MODEL = "text-embedding-ada-002"
PINECONE_ENV = "gcp-starter"
INDEX = 'ragtest'

# Global variables to store initialized services
_global_vectorstore = None
_global_client = None

# Load the environment variables from the .env file
def initialize_services(openai_api_key, pinecone_api_key):
    """Initialize OpenAI and Pinecone services"""
    global _global_vectorstore, _global_client
    
    # Set OpenAI API key
    openai.api_key = openai_api_key

    # Initialize OpenAI Embeddings model
    model = OpenAIEmbeddings(model=MODEL, openai_api_key=openai.api_key)

    # Initialize Pinecone with API key
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(INDEX)

    # Set up Pinecone VectorStore
    _global_vectorstore = PineconeVectorStore(index, model, "text")

    # Initialize OpenAI client
    _global_client = openai.OpenAI(api_key=openai.api_key)

    return _global_vectorstore, _global_client

def translate_to_italian(text, client=None):
    """Translate text from English to Italian using OpenAI."""
    global _global_client
    
    if not text:
        return ""
        
    try:
        # Get OpenAI client
        if client is None:
            client = _global_client
        
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
    global _global_client
    
    if not text:
        return ""
        
    try:
        # Get OpenAI client
        if client is None:
            client = _global_client
        
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
    global _global_vectorstore, _global_client
    
    try:
        # Check for explicit contact/support queries
        contact_keywords = [
            "contattare", "parlare con", "email", "telefono", "consulente", 
            "contatto", "contatti", "chiamare", "scrivere a", "numero di telefono",
            "indirizzo email", "posso chiamare", "posso scrivere"
        ]
        
        # Only trigger contact info if the query is explicitly about contacting someone
        is_contact_query = (
            # Must have a contact-specific keyword (not just general help)
            any(keyword in query.lower() for keyword in contact_keywords) and
            # Must explicitly mention contacting someone - not just asking if they can help
            ("fiscozen" in query.lower() or "consulente" in query.lower() or "esperto" in query.lower()) and
            # Additional check to avoid false positives on general help queries
            not any(term in query.lower() for term in ["può aiutarmi", "potete aiutare", "mi aiuta con", "servizi per"])
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
            "cosa fa fiscozen", "what does fiscozen", "servizio", "service", 
            "fiscozen può aiutarmi", "fiscozen mi aiuta", "fiscozen assiste"
        ]
        
        is_service_query = any(keyword in query.lower() for keyword in service_keywords) or (
            "fiscozen" in query.lower() and any(kw in query.lower() for kw in ["offer", "servizi", "cosa", "what", "come", "aiutare", "assistere"])
        )
        
        if is_service_query:
            # Check if the query is about a specific tax topic
            tax_topics = {
                "iva": "Sì, Fiscozen offre assistenza completa per la gestione dell'IVA, inclusi il calcolo delle aliquote corrette, la compilazione delle dichiarazioni IVA periodiche, la gestione del versamento dell'imposta e la consulenza sulle operazioni soggette a regimi IVA speciali come reverse charge o split payment.",
                "regime forfettario": "Sì, Fiscozen è specializzata nell'assistenza a professionisti in regime forfettario. I servizi includono l'apertura della partita IVA, la gestione della contabilità semplificata, il calcolo del coefficiente di redditività, la dichiarazione dei redditi annuale e la consulenza fiscale continua per ottimizzare i vantaggi di questo regime.",
                "partita iva": "Sì, Fiscozen offre un servizio completo di gestione della partita IVA, dall'apertura alla gestione ordinaria, inclusi tutti gli adempimenti fiscali, la fatturazione elettronica, e la consulenza personalizzata per scegliere il regime fiscale più adatto alle tue esigenze.",
                "fatturazione": "Sì, Fiscozen fornisce un sistema completo per la fatturazione elettronica che consente di creare, inviare e archiviare fatture in conformità con la normativa vigente. Il servizio include anche la gestione delle fatture passive e il monitoraggio costante delle scadenze.",
                "dichiarazione dei redditi": "Sì, Fiscozen offre assistenza completa per la dichiarazione dei redditi di liberi professionisti e piccole imprese, garantendo il corretto adempimento degli obblighi fiscali e l'ottimizzazione del carico fiscale attraverso tutte le detrazioni e deduzioni applicabili.",
                "detrazioni": "Sì, Fiscozen fornisce consulenza specializzata per identificare e applicare correttamente tutte le detrazioni fiscali a cui hai diritto, aiutandoti a ridurre il carico fiscale in modo legittimo e ottimizzare la tua situazione tributaria.",
                "deduzioni": "Sì, Fiscozen ti assiste nell'individuare e documentare correttamente tutte le spese deducibili per la tua attività professionale, garantendo che il tuo reddito imponibile sia calcolato nel modo più vantaggioso possibile nel rispetto della normativa."
            }
            
            # Check if the query mentions specific tax topics
            for topic, specific_response in tax_topics.items():
                if topic in query.lower():
                    return specific_response
            
            # General response about Fiscozen services if no specific topic is mentioned
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
        
        # Check for vector store
        vectorstore = _global_vectorstore
        if not vectorstore:
            return generate_local_fallback(query)
        
        # Get OpenAI client for potential fallback
        client = _global_client
        if not client:
            try:
                client = openai.OpenAI(api_key=openai.api_key)
            except Exception as e:
                print(f"Error creating OpenAI client: {e}")
                return generate_local_fallback(query)
            
            if not client:
                return generate_local_fallback(query)
        
        # Expand the query for better retrieval
        try:
            expanded_query = expand_query(query)
        except Exception as expansion_error:
            print(f"Error expanding query: {expansion_error}")
            expanded_query = query
        
        # Perform similarity search with specified k value
        try:
            docs = vectorstore.similarity_search(expanded_query, k=k)
            # Log the retrieval attempt (simplified for testing)
            print(f"Retrieved {len(docs)} document chunks for expanded query: {expanded_query}")
        except Exception as search_error:
            print(f"Error in vector search: {search_error}")
            return generate_local_fallback(query)
        
        # HYBRID APPROACH: Always combine RAG with GPT-4
        if docs:
            # Format docs for GPT-4
            context = "\n\n---\n\n".join([f"Documento {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])
            
            # Create system message for hybrid approach
            system_message = """Sei un assistente fiscale italiano di Fiscozen. La tua risposta deve combinare due fonti di informazioni:
            
            1) Le informazioni nei documenti forniti (prioritarie quando contengono dati rilevanti)
            2) La tua conoscenza generale sul sistema fiscale italiano (da usare quando i documenti non contengono risposte complete)
            
            LINEE GUIDA IMPORTANTI:
            - Non menzionare esplicitamente queste due fonti nella tua risposta
            - Crea una risposta fluida e coerente che integri le informazioni in modo naturale
            - Usa un tono professionale ma amichevole e risposte complete ma concise
            - Interpreta domande con "questo" o "quello" facendo riferimento al contesto della conversazione
            - NON indirizzare l'utente a contattare Fiscozen a meno che non lo chieda esplicitamente
            - Se l'utente chiede se Fiscozen può aiutare con un determinato argomento, spiega i servizi specifici offerti
              per quell'area, NON limitarti a fornire informazioni di contatto
            - Se ti chiedono un'opinione o un consiglio, forniscilo basandoti sui fatti disponibili
            
            Informazioni di contatto Fiscozen (da fornire SOLO se richieste esplicitamente):
            Email: supporto@fiscozen.it, Telefono: 02 8738 8787"""
            
            # Generate hybrid answer
            try:
                response = client.chat.completions.create(
                    model="gpt-4o", # Changed to GPT-4o for better performance
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": f"Documenti dal database Fiscozen:\n{context}\n\nDomanda dell'utente: {query}\n\nRispondi integrando le informazioni dai documenti con la tua conoscenza generale:"}
                    ],
                    temperature=0.2,  # Lower temperature for more consistent and predictable responses
                    max_tokens=800
                )
                    
                hybrid_response = response.choices[0].message.content
                print("Using hybrid RAG approach")
                return hybrid_response
                
            except Exception as e:
                print(f"Error generating hybrid response: {e}")
                # Fall back to local fallback
                return generate_local_fallback(query)
        
        # If no documents found or error in processing hybrid approach, use local fallback
        print("No relevant documents found or RAG processing failed. Using local fallback.")
        return generate_local_fallback(query)
            
    except Exception as e:
        print(f"Error in find_match: {e}")
        return generate_local_fallback(query)

def generate_gpt4_fallback(query, client=None):
    """Generate a response using GPT-4 as fallback when RAG doesn't have relevant information"""
    global _global_client
    
    try:
        print("Using GPT-4 fallback for query:", query)
        
        # Get OpenAI client
        if client is None:
            client = _global_client
        
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
    global _global_client
    
    try:
        # If no conversation history, expand the query directly
        if not conversation:
            return expand_query(query)
            
        # Check for follow-up markers in the query
        follow_up_markers = [
            "questo", "questa", "questi", "queste", "quello", "quella", "quelli", "quelle",
            "lo stesso", "la stessa", "mi aiuta con questo", "può aiutarmi con questo",
            "e riguardo a", "e per quanto riguarda", "cosa ne pensi di", "potete"
        ]
        
        # Detect if this is likely a follow-up question
        is_follow_up = any(marker in query.lower() for marker in follow_up_markers)
        
        # Get OpenAI client
        client = _global_client
        if not client:
            client = openai.OpenAI(api_key=openai.api_key)
            
        # Format conversation history
        conversation_text = ""
        for i, exchange in enumerate(conversation):
            conversation_text += f"Utente: {exchange['user']}\nAssistente: {exchange['assistant']}\n\n"
            
        # Create the system message
        system_message = """Sei un assistente specializzato nella riformulazione di domande fiscali per un sistema di ricerca documentale.
        
        Il tuo obiettivo è:
        1. Incorporare il contesto dalla cronologia della conversazione per trasformare domande di follow-up in domande complete
        2. Espandere la domanda con termini fiscali italiani correlati per migliorare la ricerca
        3. Includere sinonimi e termini tecnici pertinenti alla domanda
        
        ISTRUZIONI IMPORTANTI:
        - Quando la domanda è un follow-up vago come "Fiscozen può aiutarmi con questo?", devi capire a cosa si riferisce "questo" 
          dal contesto della conversazione precedente e riformulare la domanda in modo completo
        - Mantieni tutti i dettagli specifici dalle domande precedenti quando elabori follow-up
        - Se c'è un riferimento a "Fiscozen" insieme a "può aiutarmi", interpreta come se l'utente stesse chiedendo se Fiscozen offre
          servizi relativi all'argomento discusso in precedenza
        
        Ad esempio, se la conversazione riguardava l'IVA per liberi professionisti e la nuova domanda è "Fiscozen può aiutarmi con questo?",
        riformula come "Fiscozen offre servizi di consulenza e assistenza per la gestione dell'IVA per liberi professionisti?"
        
        Se la domanda è già completa ed espansa, restituiscila con lievi miglioramenti.
        Mantieni la domanda in italiano (a meno che l'originale non sia in inglese)."""
        
        # Create the user message
        user_message = f"""Cronologia della conversazione:
        {conversation_text}
        
        Domanda attuale dell'utente: {query}
        
        Riformula questa domanda in una domanda completa ed espansa includendo termini tecnici fiscali correlati:"""
        
        # Generate the refined query
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Using a smaller model for efficiency
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=0.3,
            max_tokens=300
        )
        
        refined_query = response.choices[0].message.content.strip()
        
        # If the refinement failed or produced something odd, use the base query expansion
        if not refined_query or len(refined_query) < len(query) / 2 or len(refined_query) > len(query) * 5:
            return expand_query(query)
            
        print(f"Original query: {query}")
        print(f"Refined query: {refined_query}")
        return refined_query
        
    except Exception as e:
        print(f"Error in query_refiner: {e}")
        # For follow-up queries with clear markers, attempt simple context fusion even if OpenAI fails
        if len(conversation) > 0 and any(marker in query.lower() for marker in ["questo", "questa", "questi", "queste"]):
            last_query = conversation[-1]["user"]
            print(f"Fallback simple context fusion with previous query: {last_query}")
            return f"{query} - in riferimento a: {last_query}"
        return query  # Return original query if refinement fails

def expand_query(query):
    """
    Expand a query with related fiscal terms to improve retrieval
    
    Args:
        query: The query to expand
        
    Returns:
        Expanded query with related fiscal terms
    """
    global _global_client
    
    try:
        # Get OpenAI client
        client = _global_client
        if not client:
            client = openai.OpenAI(api_key=openai.api_key)
            
        # Create the system message
        system_message = """Sei un assistente specializzato nell'espansione di query fiscali italiane.
        
        Il tuo obiettivo è arricchire la query con:
        1. Termini tecnici fiscali correlati
        2. Sinonimi appropriati
        3. Espressioni alternative che potrebbero essere presenti nei documenti
        
        Mantieni la query concisa ma completa. Non aggiungere più di 3-4 termini aggiuntivi.
        La query espansa deve mantenere lo stesso significato dell'originale."""
        
        # Generate the expanded query
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Using a smaller model for efficiency
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"Query originale: {query}\n\nEspandi questa query con termini fiscali correlati:"}
            ],
            temperature=0.3
        )
        
        expanded_query = response.choices[0].message.content.strip()
        
        # If the expansion failed or produced something odd, return the original
        if not expanded_query or len(expanded_query) < len(query) / 2 or len(expanded_query) > len(query) * 4:
            return query
            
        print(f"Original query: {query}")
        print(f"Expanded query: {expanded_query}")
        return expanded_query
        
    except Exception as e:
        print(f"Error in expand_query: {e}")
        return query  # Return original query if expansion fails 