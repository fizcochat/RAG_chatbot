import time
import pandas as pd
import matplotlib.pyplot as plt
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from ingestion.config import MODEL, INDEX, PINECONE_ENV

# Set up logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Sample queries representing different tax scenarios
TEST_QUERIES = [
    "What is the VAT rate in Italy for standard services?",
    "How do I register for a partita IVA as a freelancer?",
    "What are the filing deadlines for the forfettario regime?",
    "Can I deduct business expenses under the simplified regime?",
    "What happens if I exceed the revenue threshold for forfettario?",
    "How do quarterly VAT payments work in Italy?"
]

def initialize_vectorstore(openai_api_key, pinecone_api_key):
    """Initialize the vector store with API keys"""
    model = OpenAIEmbeddings(model=MODEL, openai_api_key=openai_api_key)
    pc = Pinecone(api_key=pinecone_api_key, environment=PINECONE_ENV)
    index = pc.Index(INDEX)
    return PineconeVectorStore(index, model, "text")

def evaluate_k_values(vectorstore, k_values=[1, 3, 5, 7, 10, 15, 20, 30, 50, 100]):
    """Test different k values and measure performance metrics"""
    results = []
    
    for query in TEST_QUERIES:
        query_results = []
        logger.info(f"Testing query: {query}")
        
        for k in k_values:
            start_time = time.time()
            
            # Perform search with current k value
            docs = vectorstore.similarity_search(query, k=k)
            
            # Calculate metrics
            retrieval_time = time.time() - start_time
            unique_sources = len(set(doc.metadata.get('source', '') for doc in docs if hasattr(doc, 'metadata')))
            total_content_length = sum(len(doc.page_content) for doc in docs if hasattr(doc, 'page_content'))
            token_estimate = total_content_length / 4  # Rough estimate of tokens
            
            # Log results
            logger.info(f"k={k}, retrieved {len(docs)} docs in {retrieval_time:.3f}s")
            logger.info(f"  - Unique sources: {unique_sources}")
            logger.info(f"  - Estimated tokens: {token_estimate:.0f}")
            
            # Preview first result
            if docs:
                first_doc = docs[0].page_content if hasattr(docs[0], 'page_content') else str(docs[0])
                preview = first_doc[:100] + "..." if len(first_doc) > 100 else first_doc
                logger.info(f"  - Top result preview: {preview}")
            
            # Store results
            query_results.append({
                'query': query,
                'k': k,
                'retrieval_time': retrieval_time,
                'doc_count': len(docs),
                'unique_sources': unique_sources,
                'total_content_length': total_content_length,
                'token_estimate': token_estimate
            })
        
        results.extend(query_results)
    
    return pd.DataFrame(results)

def analyze_and_visualize(results_df):
    """Analyze and visualize the k-value comparison results"""
    # Calculate aggregates by k value
    k_analysis = results_df.groupby('k').agg({
        'retrieval_time': ['mean', 'std'],
        'doc_count': 'mean',
        'unique_sources': 'mean',
        'token_estimate': 'mean'
    }).reset_index()
    
    print("\n==== K-Value Performance Analysis ====")
    print(k_analysis.to_string())
    
    # Plot retrieval time vs k
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 2, 1)
    plt.errorbar(k_analysis['k'], k_analysis[('retrieval_time', 'mean')], 
                 yerr=k_analysis[('retrieval_time', 'std')], marker='o')
    plt.title('Retrieval Time vs K Value')
    plt.xlabel('K Value')
    plt.ylabel('Time (seconds)')
    
    # Plot token estimate vs k
    plt.subplot(2, 2, 2)
    plt.plot(k_analysis['k'], k_analysis[('token_estimate', 'mean')], marker='o')
    plt.title('Estimated Tokens vs K Value')
    plt.xlabel('K Value')
    plt.ylabel('Estimated Tokens')
    
    # Plot unique sources vs k
    plt.subplot(2, 2, 3)
    plt.plot(k_analysis['k'], k_analysis[('unique_sources', 'mean')], marker='o')
    plt.title('Unique Sources vs K Value')
    plt.xlabel('K Value')
    plt.ylabel('Average Unique Sources')
    
    plt.tight_layout()
    plt.savefig('k_value_analysis.png')
    print("Analysis chart saved as 'k_value_analysis.png'")
    
    # Calculate information density (unique sources per k)
    k_analysis['info_density'] = k_analysis[('unique_sources', 'mean')] / k_analysis['k']
    
    # Find optimal k based on information density
    optimal_k = k_analysis.loc[k_analysis['info_density'].idxmax()]
    
    print("\n==== Optimal K Value Analysis ====")
    print(f"Optimal k for information density: {int(optimal_k['k'])}")
    print(f"  - Info density: {optimal_k[('info_density')].round(3)}")
    print(f"  - Avg retrieval time: {optimal_k[('retrieval_time', 'mean')].round(3)} seconds")
    print(f"  - Avg token estimate: {int(optimal_k[('token_estimate', 'mean')])}")
    
    # Trade-off analysis for context limits
    print("\n==== Context Window Trade-off Analysis ====")
    context_limit = 8000  # GPT-4 context limit after system prompts
    for _, row in k_analysis.iterrows():
        k = int(row['k'])
        tokens = int(row[('token_estimate', 'mean')])
        remaining = context_limit - tokens
        viability = "RECOMMENDED" if remaining >= 1500 else "POSSIBLE" if remaining > 0 else "EXCEEDS LIMIT"
        print(f"k={k}: ~{tokens} tokens, {remaining} remaining for other context. Status: {viability}")

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    
    # Run the evaluation
    vectorstore = initialize_vectorstore(openai_api_key, pinecone_api_key)
    results = evaluate_k_values(vectorstore)
    analyze_and_visualize(results)