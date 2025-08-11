"""
Modern Document Processing Pipeline for UbuntuAI
Uses LangChain document loaders and advanced chunking strategies
Dynamically loads documents from data/documents directory
"""

import json
import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import os
from datetime import datetime

# LangChain document loaders
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    TextLoader,
    PDFPlumberLoader,
    CSVLoader,
    JSONLoader,
    WebBaseLoader,
    DirectoryLoader,
    UnstructuredFileLoader
)

# Data processing
import pandas as pd

from utils.chunking import chunking_manager
from utils.context_enhancer import context_enhancer
from config.settings import settings

logger = logging.getLogger(__name__)

class ModernDataProcessor:
    """
    Modern data processing pipeline with LangChain integration
    Dynamically loads documents from data/documents directory
    """
    
    def __init__(self):
        self.chunking_manager = chunking_manager
        self.context_enhancer = context_enhancer
        self.supported_extensions = {
            '.txt': TextLoader,
            '.pdf': PDFPlumberLoader,
            '.csv': CSVLoader,
            '.json': JSONLoader,
            '.md': TextLoader,
            '.html': UnstructuredFileLoader
        }
        
        # Define the documents directory path
        self.documents_dir = Path("data/documents")
        
        logger.info("Modern data processor initialized")
    
    def process_documents_directory(self) -> List[Document]:
        """Process all documents from the data/documents directory"""
        
        logger.info(f"Processing documents from: {self.documents_dir}")
        documents = []
        
        if not self.documents_dir.exists():
            logger.warning(f"Documents directory {self.documents_dir} does not exist")
            return documents
        
        try:
            # Process all files in the documents directory
            for file_path in self.documents_dir.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in self.supported_extensions:
                    logger.info(f"Processing file: {file_path.name}")
                    file_docs = self.process_file(file_path)
                    documents.extend(file_docs)
            
            logger.info(f"Processed {len(documents)} documents from {self.documents_dir}")
            return documents
            
        except Exception as e:
            logger.error(f"Error processing documents directory: {e}")
            return []
    
    def process_file(self, file_path: Union[str, Path]) -> List[Document]:
        """Process a single file into LangChain documents"""
        
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                logger.warning(f"File does not exist: {file_path}")
                return []
            
            # Determine the appropriate loader based on file extension
            file_extension = file_path.suffix.lower()
            
            if file_extension not in self.supported_extensions:
                logger.warning(f"Unsupported file type: {file_extension}")
                return []
            
            loader_class = self.supported_extensions[file_extension]
            
            # Load the document
            if file_extension == '.pdf':
                loader = loader_class(str(file_path))
            else:
                loader = loader_class(str(file_path))
            
            # Load and process the document
            raw_docs = loader.load()
            
            # Enhance documents with metadata
            enhanced_docs = []
            for doc in raw_docs:
                enhanced_doc = self._enhance_document_metadata(doc, file_path)
                enhanced_docs.append(enhanced_doc)
            
            logger.info(f"Processed {len(enhanced_docs)} documents from {file_path.name}")
            return enhanced_docs
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return []
    
    def process_directory(self, 
                        directory_path: Union[str, Path],
                        recursive: bool = True,
                        file_pattern: str = "**/*") -> List[Document]:
        """Process all files in a directory using LangChain DirectoryLoader"""
        
        try:
            directory_path = Path(directory_path)
            
            if not directory_path.exists():
                logger.warning(f"Directory does not exist: {directory_path}")
                return []
            
            # Use LangChain DirectoryLoader for efficient processing
            loader = DirectoryLoader(
                str(directory_path),
                glob=file_pattern,
                recursive=recursive,
                show_progress=True,
                use_multithreading=True
            )
            
            documents = loader.load()
            
            # Enhance documents with metadata
            enhanced_docs = []
            for doc in documents:
                enhanced_doc = self._enhance_document_metadata(doc, directory_path)
                enhanced_docs.append(enhanced_doc)
            
            logger.info(f"Processed {len(enhanced_docs)} documents from directory: {directory_path}")
            return enhanced_docs
            
        except Exception as e:
            logger.error(f"Error processing directory {directory_path}: {e}")
            return []
    
    def process_web_content(self, urls: List[str]) -> List[Document]:
        """Process web content from URLs"""
        
        logger.info(f"Processing {len(urls)} web URLs")
        documents = []
        
        try:
            for url in urls:
                try:
                    loader = WebBaseLoader(url)
                    docs = loader.load()
                    
                    # Enhance documents with metadata
                    for doc in docs:
                        enhanced_doc = self._enhance_document_metadata(doc, url, source_type="web")
                        enhanced_docs.append(enhanced_doc)
                    
                    documents.extend(enhanced_docs)
                    logger.info(f"Processed web content from: {url}")
                    
                except Exception as e:
                    logger.error(f"Error processing URL {url}: {e}")
                    continue
            
            return documents
            
        except Exception as e:
            logger.error(f"Error processing web content: {e}")
            return []
    
    def chunk_documents(self, 
                       documents: List[Document],
                       strategy: str = None,
                       **kwargs) -> List[Document]:
        """Chunk documents using the chunking manager"""
        
        try:
            strategy = strategy or settings.CHUNKING_STRATEGY
            
            logger.info(f"Chunking {len(documents)} documents using {strategy} strategy")
            
            chunked_docs = self.chunking_manager.chunk_documents(
                documents=documents,
                strategy=strategy,
                chunk_size=kwargs.get('chunk_size', settings.CHUNK_SIZE),
                chunk_overlap=kwargs.get('chunk_overlap', settings.CHUNK_OVERLAP)
            )
            
            logger.info(f"Created {len(chunked_docs)} chunks from {len(documents)} documents")
            return chunked_docs
            
        except Exception as e:
            logger.error(f"Error chunking documents: {e}")
            return documents
    
    def enhance_documents_with_context(self, documents: List[Document]) -> List[Document]:
        """Enhance documents with additional context using the context enhancer"""
        
        try:
            logger.info(f"Enhancing {len(documents)} documents with context")
            
            enhanced_docs = []
            for doc in documents:
                enhanced_doc = self.context_enhancer.enhance_document(doc)
                enhanced_docs.append(enhanced_doc)
            
            logger.info(f"Enhanced {len(enhanced_docs)} documents with context")
            return enhanced_docs
            
        except Exception as e:
            logger.error(f"Error enhancing documents with context: {e}")
            return documents
    
    def process_pipeline(self, 
                        sources: Dict[str, Any] = None,
                        chunking_strategy: str = None,
                        enhance_context: bool = True) -> List[Document]:
        """Main processing pipeline for documents"""
        
        logger.info("Starting document processing pipeline")
        
        try:
            # Default to processing documents directory if no sources specified
            if not sources:
                sources = {"documents_directory": str(self.documents_dir)}
            
            all_documents = []
            
            # Process documents directory
            if "documents_directory" in sources:
                docs = self.process_documents_directory()
                all_documents.extend(docs)
            
            # Process additional sources if specified
            if "files" in sources:
                for file_path in sources["files"]:
                    docs = self.process_file(file_path)
                    all_documents.extend(docs)
            
            if "directories" in sources:
                for dir_path in sources["directories"]:
                    docs = self.process_directory(dir_path)
                    all_documents.extend(docs)
            
            if "urls" in sources:
                docs = self.process_web_content(sources["urls"])
                all_documents.extend(docs)
            
            # Remove duplicates based on content hash
            unique_documents = self._remove_duplicates(all_documents)
            
            # Chunk documents
            chunked_documents = self.chunk_documents(
                unique_documents, 
                strategy=chunking_strategy
            )
            
            # Enhance context if requested
            if enhance_context:
                final_documents = self.enhance_documents_with_context(chunked_documents)
            else:
                final_documents = chunked_documents
            
            logger.info(f"Pipeline completed. Final document count: {len(final_documents)}")
            return final_documents
            
        except Exception as e:
            logger.error(f"Error in processing pipeline: {e}")
            return []
    
    def _enhance_document_metadata(self, 
                                 doc: Document, 
                                 source_path: Union[str, Path],
                                 source_type: str = "file") -> Document:
        """Enhance document with additional metadata"""
        
        try:
            # Create enhanced metadata
            enhanced_metadata = {
                "source": str(source_path),
                "source_type": source_type,
                "processed_date": datetime.now().isoformat(),
                "file_extension": Path(source_path).suffix.lower() if source_type == "file" else None,
                "file_name": Path(source_path).name if source_type == "file" else None,
                "ghana_focus": True,  # All documents are Ghana-focused
                "sector_relevance": self._detect_sector_relevance(doc.page_content)
            }
            
            # Merge with existing metadata
            if hasattr(doc, 'metadata') and doc.metadata:
                enhanced_metadata.update(doc.metadata)
            
            # Create new document with enhanced metadata
            enhanced_doc = Document(
                page_content=doc.page_content,
                metadata=enhanced_metadata
            )
            
            return enhanced_doc
            
        except Exception as e:
            logger.error(f"Error enhancing document metadata: {e}")
            return doc
    
    def _detect_sector_relevance(self, content: str) -> List[str]:
        """Detect which Ghanaian sectors the document is most relevant to"""
        
        content_lower = content.lower()
        relevant_sectors = []
        
        # Fintech keywords
        fintech_keywords = [
            "fintech", "financial technology", "mobile money", "digital payments",
            "banking", "insurance", "lending", "investment", "blockchain",
            "bank of ghana", "gipc", "financial services"
        ]
        
        # Agritech keywords
        agritech_keywords = [
            "agritech", "agricultural technology", "farming", "crop",
            "livestock", "irrigation", "soil", "fertilizer", "pesticide",
            "ministry of food and agriculture", "mofa", "agriculture"
        ]
        
        # Healthtech keywords
        healthtech_keywords = [
            "healthtech", "health technology", "healthcare", "medical",
            "pharmaceutical", "telemedicine", "health", "medicine", "patient",
            "food and drugs authority", "fda", "ministry of health"
        ]
        
        # Check relevance
        if any(keyword in content_lower for keyword in fintech_keywords):
            relevant_sectors.append("fintech")
        
        if any(keyword in content_lower for keyword in agritech_keywords):
            relevant_sectors.append("agritech")
        
        if any(keyword in content_lower for keyword in healthtech_keywords):
            relevant_sectors.append("healthtech")
        
        return relevant_sectors if relevant_sectors else ["general"]
    
    def _remove_duplicates(self, documents: List[Document]) -> List[Document]:
        """Remove duplicate documents based on content hash"""
        
        try:
            seen_content = set()
            unique_documents = []
            
            for doc in documents:
                content_hash = hash(doc.page_content)
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    unique_documents.append(doc)
            
            logger.info(f"Removed {len(documents) - len(unique_documents)} duplicate documents")
            return unique_documents
            
        except Exception as e:
            logger.error(f"Error removing duplicates: {e}")
            return documents
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get statistics about the processing pipeline"""
        
        try:
            documents_dir = self.documents_dir
            
            if not documents_dir.exists():
                return {"error": "Documents directory does not exist"}
            
            # Count files by type
            file_counts = {}
            total_files = 0
            
            for file_path in documents_dir.iterdir():
                if file_path.is_file():
                    total_files += 1
                    ext = file_path.suffix.lower()
                    file_counts[ext] = file_counts.get(ext, 0) + 1
            
            return {
                "total_files": total_files,
                "file_types": file_counts,
                "supported_extensions": list(self.supported_extensions.keys()),
                "documents_directory": str(documents_dir),
                "ghana_focus": True,
                "target_sectors": settings.GHANA_STARTUP_SECTORS
            }
            
        except Exception as e:
            logger.error(f"Error getting processing stats: {e}")
            return {"error": str(e)}

# Global instance
data_processor = ModernDataProcessor()