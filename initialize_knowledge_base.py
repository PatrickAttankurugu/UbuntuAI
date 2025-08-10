#!/usr/bin/env python3
"""
Initialize the UbuntuAI Knowledge Base

This script sets up the vector database with initial data from:
- Internal funding database
- Regulatory information
- Sample documents for testing

Run this script once after setting up the environment.
"""

import sys
import os
from typing import List, Dict, Any

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def initialize_knowledge_base():
    try:
        print("Initializing UbuntuAI Knowledge Base...")
        
        # Import modules
        from api.vector_store import vector_store
        from data.processor import data_processor
        from config.settings import settings
        
        print("Processing internal data sources...")
        
        # Process funding data
        print("  Processing funding opportunities...")
        funding_chunks = data_processor.process_funding_data()
        print(f"     Generated {len(funding_chunks)} funding chunks")
        
        # Process regulatory data
        print("  Processing regulatory information...")
        regulatory_chunks = data_processor.process_regulatory_data()
        print(f"     Generated {len(regulatory_chunks)} regulatory chunks")
        
        # Generate sample documents for testing
        print("  Generating sample documents...")
        sample_chunks = data_processor.generate_sample_documents()
        print(f"     Generated {len(sample_chunks)} sample chunks")
        
        # Combine all chunks
        all_chunks = funding_chunks + regulatory_chunks + sample_chunks
        print(f"Total chunks to process: {len(all_chunks)}")
        
        # Prepare for vector store
        print("Preparing documents for vector storage...")
        vector_data = data_processor.prepare_documents_for_vectorstore(all_chunks)
        
        # Add to vector store
        print("Adding documents to vector database...")
        success = vector_store.add_documents(
            documents=vector_data["documents"],
            metadatas=vector_data["metadatas"],
            ids=vector_data["ids"]
        )
        
        if success:
            print("SUCCESS: Knowledge base initialized successfully!")
            
            # Get stats
            stats = vector_store.get_collection_stats()
            print(f"Database Statistics:")
            print(f"   - Total documents: {stats.get('total_documents', 0)}")
            print(f"   - Collection name: {stats.get('collection_name', 'N/A')}")
            print(f"   - Persist directory: {stats.get('persist_directory', 'N/A')}")
            
        else:
            print("ERROR: Failed to initialize knowledge base")
            return False
        
        # Test the system
        print("\nTesting the RAG system...")
        test_rag_system()
        
        return True
        
    except ImportError as e:
        print(f"ERROR - Import error: {e}")
        print("Please ensure all dependencies are installed:")
        print("pip install -r requirements.txt")
        return False
        
    except Exception as e:
        print(f"ERROR - Error during initialization: {e}")
        return False

def test_rag_system():
    try:
        from api.rag_engine import rag_engine
        
        # Test queries
        test_queries = [
            "What funding opportunities are available for fintech startups in Nigeria?",
            "How do I register a business in Kenya?",
            "Tell me about successful African startups",
        ]
        
        print("\nRunning test queries...")
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{i}. Testing: {query[:50]}...")
            
            try:
                response = rag_engine.query(query)
                
                if response and response.get("answer"):
                    print(f"   SUCCESS: Response generated ({len(response['answer'])} chars)")
                    print(f"   Sources found: {len(response.get('sources', []))}")
                    print(f"   Confidence: {response.get('confidence', 0.0):.2f}")
                else:
                    print("   WARNING: No response generated")
                    
            except Exception as e:
                print(f"   ERROR: Query failed: {e}")
        
        print("\nRAG system test completed!")
        
    except Exception as e:
        print(f"ERROR: RAG test failed: {e}")

def check_environment():
    """Check if the environment is properly set up"""
    print("Checking environment setup...")
    
    required_env_vars = ["OPENAI_API_KEY"]
    missing_vars = []
    
    for var in required_env_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"ERROR: Missing required environment variables: {', '.join(missing_vars)}")
        print("Please create a .env file with the required variables.")
        print("See .env.example for reference.")
        return False
    
    # Check if directories exist
    required_dirs = ["vector_db", "data", "api", "config", "utils", "knowledge_base"]
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            print(f"ERROR: Missing directory: {dir_name}")
            return False
    
    print("SUCCESS: Environment setup looks good!")
    return True

def main():
    print("=" * 60)
    print("UbuntuAI Knowledge Base Initialization")
    print("=" * 60)
    
    # Check environment first
    if not check_environment():
        print("\nERROR: Environment check failed. Please fix the issues above.")
        sys.exit(1)
    
    # Initialize knowledge base
    success = initialize_knowledge_base()
    
    if success:
        print("\n" + "=" * 60)
        print("SUCCESS: UbuntuAI is ready to use!")
        print("Run: streamlit run app.py")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("ERROR: Initialization failed. Please check the errors above.")
        print("=" * 60)
        sys.exit(1)

if __name__ == "__main__":
    main()