#!/usr/bin/env python3
"""
Ghanaian RAG System Initialization Script
Initializes the RAG system with documents from data/documents directory
Focuses on fintech, agritech, and healthtech for Ghanaian startup ecosystem
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ghana_rag_init.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from config.settings import settings
    from data.processor import data_processor
    from api.vector_store import initialize_vector_store
    from api.rag_engine import initialize_rag_engine
    from api.llm_providers import llm_manager
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    sys.exit(1)

def check_documents_directory():
    """Check if documents directory exists and contains files"""
    
    documents_dir = Path("data/documents")
    
    if not documents_dir.exists():
        logger.error(f"Documents directory {documents_dir} does not exist")
        return False
    
    # List all files in the directory
    files = list(documents_dir.iterdir())
    if not files:
        logger.error(f"No files found in {documents_dir}")
        return False
    
    # Check file types
    supported_extensions = {'.pdf', '.txt', '.csv', '.json', '.md', '.html'}
    supported_files = [f for f in files if f.is_file() and f.suffix.lower() in supported_extensions]
    
    if not supported_files:
        logger.error(f"No supported file types found in {documents_dir}")
        logger.info(f"Supported extensions: {supported_extensions}")
        return False
    
    logger.info(f"Found {len(supported_files)} supported documents:")
    for file in supported_files:
        logger.info(f"  - {file.name} ({file.suffix})")
    
    return True

def process_documents():
    """Process documents from the data/documents directory"""
    
    logger.info("Starting document processing...")
    
    try:
        # Process documents using the data processor
        documents = data_processor.process_documents_directory()
        
        if not documents:
            logger.error("No documents were processed")
            return False
        
        logger.info(f"Successfully processed {len(documents)} documents")
        
        # Display document statistics
        stats = data_processor.get_processing_stats()
        logger.info(f"Processing statistics: {stats}")
        
        return documents
        
    except Exception as e:
        logger.error(f"Error processing documents: {e}")
        return False

def initialize_system():
    """Initialize the complete Ghanaian RAG system"""
    
    logger.info("🚀 Initializing Ghanaian RAG System")
    logger.info("=" * 50)
    
    # Check system configuration
    logger.info("📋 Checking system configuration...")
    try:
        config_info = settings.to_dict()
        logger.info(f"Configuration loaded successfully:")
        logger.info(f"  - Primary LLM Provider: {config_info['primary_llm_provider']}")
        logger.info(f"  - Available LLM Providers: {config_info['available_llm_providers']}")
        logger.info(f"  - Vector Store Type: {config_info['vector_store_type']}")
        logger.info(f"  - Retrieval Strategy: {config_info['retrieval_strategy']}")
        logger.info(f"  - Ghanaian Sectors: {config_info['supported_sectors']}")
    except Exception as e:
        logger.error(f"Configuration error: {e}")
        return False
    
    # Check documents directory
    logger.info("📚 Checking documents directory...")
    if not check_documents_directory():
        return False
    
    # Process documents
    logger.info("🔄 Processing documents...")
    documents = process_documents()
    if not documents:
        return False
    
    # Initialize vector store
    logger.info("🗄️ Initializing vector store...")
    try:
        vector_store = initialize_vector_store()
        if not vector_store:
            logger.error("Failed to initialize vector store")
            return False
        logger.info("Vector store initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing vector store: {e}")
        return False
    
    # Initialize RAG engine
    logger.info("🤖 Initializing RAG engine...")
    try:
        rag_engine = initialize_rag_engine(vector_store)
        if not rag_engine:
            logger.error("Failed to initialize RAG engine")
            return False
        logger.info("RAG engine initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing RAG engine: {e}")
        return False
    
    # Test LLM providers
    logger.info("🧪 Testing LLM providers...")
    try:
        available_providers = llm_manager.get_available_providers()
        logger.info(f"Available LLM providers: {available_providers}")
        
        if available_providers:
            # Test the primary provider
            primary_provider = settings.PRIMARY_LLM_PROVIDER
            if primary_provider in available_providers:
                logger.info(f"Primary provider {primary_provider} is available")
            else:
                logger.warning(f"Primary provider {primary_provider} not available, using {available_providers[0]}")
        else:
            logger.warning("No LLM providers available")
    except Exception as e:
        logger.error(f"Error testing LLM providers: {e}")
    
    logger.info("=" * 50)
    logger.info("✅ Ghanaian RAG System initialization completed successfully!")
    
    return True

def main():
    """Main initialization function"""
    
    try:
        success = initialize_system()
        
        if success:
            logger.info("🎉 System is ready to use!")
            logger.info("You can now run the main application with: streamlit run app.py")
            
            # Display next steps
            print("\n" + "="*60)
            print("🇬🇭 Ghanaian RAG System Initialized Successfully!")
            print("="*60)
            print("📚 Documents processed from data/documents directory")
            print("🤖 RAG engine ready for Ghanaian startup ecosystem queries")
            print("🎯 Focus areas: Fintech, Agritech, Healthtech")
            print("📍 Coverage: All 16 regions of Ghana")
            print("\n🚀 Next steps:")
            print("1. Run the application: streamlit run app.py")
            print("2. Ask questions about Ghanaian startups")
            print("3. Explore fintech, agritech, and healthtech opportunities")
            print("="*60)
            
        else:
            logger.error("❌ System initialization failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Initialization interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during initialization: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 