#!/usr/bin/env python3
"""
UbuntuAI Knowledge Base Initialization

This script initializes the vector database with sample business data for UbuntuAI.
It processes various data sources and creates embeddings for retrieval.

Usage:
    python initialize_knowledge_base.py [--verbose] [--force]
    
Arguments:
    --verbose: Enable verbose logging
    --force: Force reinitialization even if data already exists
"""

import sys
import os
import argparse
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import traceback

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def setup_logging(verbose: bool = False) -> logging.Logger:
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('knowledge_base_init.log'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def check_prerequisites() -> tuple[bool, List[str]]:
    """
    Check if all prerequisites are met for initialization
    
    Returns:
        Tuple of (success, error_messages)
    """
    errors = []
    
    try:
        # Check settings
        from config.settings import settings
        
        if not settings.GOOGLE_API_KEY:
            errors.append("Google Gemini API key not configured")
        
        # Check if vector store is accessible
        try:
            from api.vector_store import vector_store
            stats = vector_store.get_collection_stats()
            if stats.get('error'):
                errors.append(f"Vector store error: {stats['error']}")
        except Exception as e:
            errors.append(f"Cannot access vector store: {e}")
        
        # Check data processors
        try:
            from data.processor import data_processor
        except Exception as e:
            errors.append(f"Cannot import data processor: {e}")
        
        return len(errors) == 0, errors
        
    except Exception as e:
        errors.append(f"Critical import error: {e}")
        return False, errors

class KnowledgeBaseInitializer:
    """
    Main class for initializing the UbuntuAI knowledge base
    """
    
    def __init__(self, logger: logging.Logger):
        """Initialize with logger"""
        self.logger = logger
        self.stats = {
            'total_documents': 0,
            'successful_embeddings': 0,
            'failed_embeddings': 0,
            'processing_time': 0,
            'start_time': None,
            'end_time': None
        }
        
        # Import required modules
        try:
            from api.vector_store import vector_store
            from data.processor import data_processor
            from config.settings import settings
            
            self.vector_store = vector_store
            self.data_processor = data_processor
            self.settings = settings
            
            self.logger.info("Initializer setup completed successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to setup initializer: {e}")
            raise
    
    def check_existing_data(self) -> Dict[str, Any]:
        """Check if knowledge base already has data"""
        try:
            stats = self.vector_store.get_collection_stats()
            existing_count = stats.get('total_documents', 0)
            
            self.logger.info(f"Found {existing_count} existing documents in knowledge base")
            
            return {
                'has_data': existing_count > 0,
                'document_count': existing_count,
                'collection_info': stats
            }
            
        except Exception as e:
            self.logger.error(f"Error checking existing data: {e}")
            return {'has_data': False, 'document_count': 0, 'error': str(e)}
    
    def initialize_knowledge_base(self, force: bool = False) -> bool:
        """
        Initialize the knowledge base with sample data
        
        Args:
            force: Force reinitialization even if data exists
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.stats['start_time'] = datetime.now()
            self.logger.info("Starting knowledge base initialization...")
            
            # Check existing data
            existing_data = self.check_existing_data()
            
            if existing_data['has_data'] and not force:
                self.logger.info(f"Knowledge base already contains {existing_data['document_count']} documents")
                self.logger.info("Use --force flag to reinitialize")
                return True
            
            if force and existing_data['has_data']:
                self.logger.info("Force flag detected - will reinitialize existing data")
            
            # Process different data sources
            success = True
            
            success &= self._process_sample_documents()
            success &= self._process_funding_data()
            success &= self._process_regulatory_data()
            
            # Final statistics
            self.stats['end_time'] = datetime.now()
            self.stats['processing_time'] = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
            
            self._log_final_stats()
            
            if success:
                self.logger.info("✅ Knowledge base initialization completed successfully!")
            else:
                self.logger.warning("⚠️ Knowledge base initialization completed with some errors")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Fatal error during initialization: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def _process_sample_documents(self) -> bool:
        """Process sample documents from data processor"""
        try:
            self.logger.info("Processing sample documents...")
            
            # Generate sample documents
            sample_docs = self.data_processor.generate_sample_documents()
            
            if not sample_docs:
                self.logger.warning("No sample documents generated")
                return True
            
            self.logger.info(f"Generated {len(sample_docs)} sample documents")
            
            # Prepare for vector store
            prepared_data = self.data_processor.prepare_documents_for_vectorstore(sample_docs)
            
            # Add to vector store
            success = self.vector_store.add_documents(
                documents=prepared_data['documents'],
                metadatas=prepared_data['metadatas'],
                ids=prepared_data['ids']
            )
            
            if success:
                self.stats['total_documents'] += len(prepared_data['documents'])
                self.stats['successful_embeddings'] += len(prepared_data['documents'])
                self.logger.info(f"✅ Successfully added {len(prepared_data['documents'])} sample documents")
            else:
                self.stats['failed_embeddings'] += len(prepared_data['documents'])
                self.logger.error("❌ Failed to add sample documents")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error processing sample documents: {e}")
            return False
    
    def _process_funding_data(self) -> bool:
        """Process funding database information"""
        try:
            self.logger.info("Processing funding data...")
            
            # Process funding opportunities
            funding_chunks = self.data_processor.process_funding_data()
            
            if not funding_chunks:
                self.logger.warning("No funding data to process")
                return True
            
            self.logger.info(f"Generated {len(funding_chunks)} funding document chunks")
            
            # Prepare for vector store
            prepared_data = self.data_processor.prepare_documents_for_vectorstore(funding_chunks)
            
            # Create unique IDs for funding data
            funding_ids = [f"funding_{i}" for i in range(len(prepared_data['documents']))]
            
            # Add to vector store
            success = self.vector_store.add_documents(
                documents=prepared_data['documents'],
                metadatas=prepared_data['metadatas'],
                ids=funding_ids
            )
            
            if success:
                self.stats['total_documents'] += len(prepared_data['documents'])
                self.stats['successful_embeddings'] += len(prepared_data['documents'])
                self.logger.info(f"✅ Successfully added {len(prepared_data['documents'])} funding documents")
            else:
                self.stats['failed_embeddings'] += len(prepared_data['documents'])
                self.logger.error("❌ Failed to add funding documents")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error processing funding data: {e}")
            return False
    
    def _process_regulatory_data(self) -> bool:
        """Process regulatory information"""
        try:
            self.logger.info("Processing regulatory data...")
            
            # Process regulatory information
            regulatory_chunks = self.data_processor.process_regulatory_data()
            
            if not regulatory_chunks:
                self.logger.warning("No regulatory data to process")
                return True
            
            self.logger.info(f"Generated {len(regulatory_chunks)} regulatory document chunks")
            
            # Prepare for vector store
            prepared_data = self.data_processor.prepare_documents_for_vectorstore(regulatory_chunks)
            
            # Create unique IDs for regulatory data
            regulatory_ids = [f"regulatory_{i}" for i in range(len(prepared_data['documents']))]
            
            # Add to vector store
            success = self.vector_store.add_documents(
                documents=prepared_data['documents'],
                metadatas=prepared_data['metadatas'],
                ids=regulatory_ids
            )
            
            if success:
                self.stats['total_documents'] += len(prepared_data['documents'])
                self.stats['successful_embeddings'] += len(prepared_data['documents'])
                self.logger.info(f"✅ Successfully added {len(prepared_data['documents'])} regulatory documents")
            else:
                self.stats['failed_embeddings'] += len(prepared_data['documents'])
                self.logger.error("❌ Failed to add regulatory documents")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error processing regulatory data: {e}")
            return False
    
    def _log_final_stats(self):
        """Log final initialization statistics"""
        self.logger.info("\n" + "="*50)
        self.logger.info("KNOWLEDGE BASE INITIALIZATION SUMMARY")
        self.logger.info("="*50)
        self.logger.info(f"Total Documents Processed: {self.stats['total_documents']}")
        self.logger.info(f"Successful Embeddings: {self.stats['successful_embeddings']}")
        self.logger.info(f"Failed Embeddings: {self.stats['failed_embeddings']}")
        self.logger.info(f"Processing Time: {self.stats['processing_time']:.2f} seconds")
        
        if self.stats['total_documents'] > 0:
            success_rate = (self.stats['successful_embeddings'] / self.stats['total_documents']) * 100
            self.logger.info(f"Success Rate: {success_rate:.1f}%")
        
        # Get final vector store stats
        try:
            final_stats = self.vector_store.get_collection_stats()
            self.logger.info(f"Final Document Count: {final_stats.get('total_documents', 'Unknown')}")
            self.logger.info(f"Collection Name: {final_stats.get('collection_name', 'Unknown')}")
        except Exception as e:
            self.logger.warning(f"Could not retrieve final stats: {e}")
        
        self.logger.info("="*50)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Initialize UbuntuAI Knowledge Base",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--force', '-f',
        action='store_true', 
        help='Force reinitialization even if data already exists'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.verbose)
    
    logger.info("Starting UbuntuAI Knowledge Base Initialization")
    logger.info(f"Arguments: verbose={args.verbose}, force={args.force}")
    
    try:
        # Check prerequisites
        logger.info("Checking prerequisites...")
        prereq_ok, errors = check_prerequisites()
        
        if not prereq_ok:
            logger.error("Prerequisites check failed:")
            for error in errors:
                logger.error(f"  - {error}")
            
            logger.error("\nPlease fix the above issues before running initialization.")
            logger.error("Make sure you have:")
            logger.error("1. Set OPENAI_API_KEY in your .env file")
            logger.error("2. Installed all required dependencies: pip install -r requirements.txt")
            logger.error("3. Proper file permissions for the vector_db directory")
            return 1
        
        logger.info("✅ Prerequisites check passed")
        
        # Initialize
        initializer = KnowledgeBaseInitializer(logger)
        success = initializer.initialize_knowledge_base(force=args.force)
        
        if success:
            logger.info("\n🎉 Knowledge base initialization completed successfully!")
            logger.info("You can now run the UbuntuAI application with: streamlit run app.py")
            return 0
        else:
            logger.error("\n❌ Knowledge base initialization failed")
            logger.error("Check the logs above for specific error details")
            return 1
            
    except KeyboardInterrupt:
        logger.info("\n⏹️ Initialization cancelled by user")
        return 1
    except Exception as e:
        logger.error(f"\n💥 Fatal error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return 1

if __name__ == "__main__":
    sys.exit(main())