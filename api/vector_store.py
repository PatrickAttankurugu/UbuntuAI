#!/usr/bin/env python3
"""
Vector store implementation for UbuntuAI
Uses ChromaDB for document storage and retrieval
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils import embedding_functions
import json
import uuid
from datetime import datetime

from config.settings import settings
from utils.embeddings import embedding_service

logger = logging.getLogger(__name__)

class VectorStore:
    """ChromaDB-based vector store for document storage and retrieval"""
    
    def __init__(self):
        """Initialize the vector store"""
        self.collection_name = settings.COLLECTION_NAME
        self.persist_directory = settings.VECTOR_STORE_PATH
        self.embedding_dimensions = settings.EMBEDDING_DIMENSIONS
        
        # Create persist directory
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Initialize ChromaDB client
        try:
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            logger.info(f"ChromaDB client initialized at {self.persist_directory}")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client: {e}")
            raise
        
        # Initialize collection
        self._initialize_collection()
    
    def _initialize_collection(self):
        """Initialize or get the document collection"""
        try:
            # Check if collection exists
            try:
                self.collection = self.client.get_collection(
                    name=self.collection_name,
                    embedding_function=self._get_embedding_function()
                )
                logger.info(f"Using existing collection: {self.collection_name}")
            except Exception:
                # Create new collection
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    embedding_function=self._get_embedding_function(),
                    metadata={"description": "UbuntuAI Knowledge Base"}
                )
                logger.info(f"Created new collection: {self.collection_name}")
        
        except Exception as e:
            logger.error(f"Failed to initialize collection: {e}")
            raise
    
    def _get_embedding_function(self):
        """Get the appropriate embedding function"""
        if embedding_service and embedding_service.is_initialized():
            # Use custom embedding function that calls our service
            return CustomEmbeddingFunction(embedding_service)
        else:
            # Fallback to default ChromaDB embedding function
            logger.warning("Using default ChromaDB embedding function")
            return embedding_functions.DefaultEmbeddingFunction()
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """
        Add documents to the vector store
        
        Args:
            documents: List of document dictionaries with 'text', 'metadata', etc.
            
        Returns:
            True if successful, False otherwise
        """
        if not documents:
            logger.warning("No documents provided for addition")
            return False
        
        try:
            # Prepare documents for ChromaDB
            ids = []
            texts = []
            metadatas = []
            
            for doc in documents:
                # Generate unique ID if not provided
                doc_id = doc.get('id', str(uuid.uuid4()))
                
                # Extract text content
                text = doc.get('text', '')
                if not text:
                    logger.warning(f"Document {doc_id} has no text content, skipping")
                    continue
                
                # Prepare metadata
                metadata = doc.get('metadata', {})
                metadata.update({
                    'added_at': datetime.now().isoformat(),
                    'document_type': doc.get('type', 'unknown'),
                    'source': doc.get('source', 'unknown')
                })
                
                ids.append(doc_id)
                texts.append(text)
                metadatas.append(metadata)
            
            if not ids:
                logger.warning("No valid documents to add")
                return False
            
            # Add to collection
            self.collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas
            )
            
            logger.info(f"Successfully added {len(ids)} documents to vector store")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add documents to vector store: {e}")
            return False
    
    def search(self, query: str, n_results: int = None, filter_metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Search for similar documents
        
        Args:
            query: Search query text
            n_results: Number of results to return
            filter_metadata: Metadata filters to apply
            
        Returns:
            List of search results with documents and metadata
        """
        if not query or not query.strip():
            logger.warning("Empty query provided")
            return []
        
        try:
            n_results = n_results or settings.MAX_RETRIEVAL_RESULTS
            
            # Perform search
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=filter_metadata
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    result = {
                        'document': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                        'distance': results['distances'][0][i] if results['distances'] else 0.0,
                        'id': results['ids'][0][i] if results['ids'] else None
                    }
                    formatted_results.append(result)
            
            logger.info(f"Search returned {len(formatted_results)} results for query: {query[:50]}...")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific document by ID
        
        Args:
            document_id: ID of the document to retrieve
            
        Returns:
            Document data or None if not found
        """
        try:
            result = self.collection.get(ids=[document_id])
            
            if result['documents'] and result['documents'][0]:
                return {
                    'id': document_id,
                    'document': result['documents'][0],
                    'metadata': result['metadatas'][0] if result['metadatas'] else {}
                }
            else:
                logger.warning(f"Document {document_id} not found")
                return None
                
        except Exception as e:
            logger.error(f"Failed to retrieve document {document_id}: {e}")
            return None
    
    def update_document(self, document_id: str, text: str = None, metadata: Dict[str, Any] = None) -> bool:
        """
        Update an existing document
        
        Args:
            document_id: ID of the document to update
            text: New text content (optional)
            metadata: New metadata (optional)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            update_data = {}
            
            if text is not None:
                update_data['documents'] = [text]
            
            if metadata is not None:
                metadata['updated_at'] = datetime.now().isoformat()
                update_data['metadatas'] = [metadata]
            
            if not update_data:
                logger.warning("No update data provided")
                return False
            
            self.collection.update(
                ids=[document_id],
                **update_data
            )
            
            logger.info(f"Successfully updated document {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update document {document_id}: {e}")
            return False
    
    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document from the vector store
        
        Args:
            document_id: ID of the document to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.collection.delete(ids=[document_id])
            logger.info(f"Successfully deleted document {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {e}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        try:
            count = self.collection.count()
            
            # Get sample metadata to understand structure
            sample = self.collection.peek(limit=1)
            metadata_keys = set()
            if sample['metadatas']:
                for metadata in sample['metadatas']:
                    metadata_keys.update(metadata.keys())
            
            return {
                'collection_name': self.collection_name,
                'total_documents': count,
                'persist_directory': self.persist_directory,
                'embedding_dimensions': self.embedding_dimensions,
                'metadata_keys': list(metadata_keys),
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {
                'error': str(e),
                'collection_name': self.collection_name
            }
    
    def clear_collection(self) -> bool:
        """Clear all documents from the collection"""
        try:
            self.collection.delete(where={})
            logger.info("Successfully cleared collection")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            return False
    
    def export_collection(self, filepath: str) -> bool:
        """Export collection data to a JSON file"""
        try:
            # Get all documents
            results = self.collection.get()
            
            export_data = {
                'collection_name': self.collection_name,
                'export_date': datetime.now().isoformat(),
                'documents': []
            }
            
            if results['documents']:
                for i in range(len(results['documents'])):
                    doc_data = {
                        'id': results['ids'][i],
                        'text': results['documents'][i],
                        'metadata': results['metadatas'][i] if results['metadatas'] else {}
                    }
                    export_data['documents'].append(doc_data)
            
            # Write to file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Collection exported to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export collection: {e}")
            return False

class CustomEmbeddingFunction:
    """Custom embedding function that uses our embedding service"""
    
    def __init__(self, embedding_service):
        self.embedding_service = embedding_service
    
    def __call__(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        if not self.embedding_service or not self.embedding_service.is_initialized():
            logger.error("Embedding service not available")
            return [[0.0] * self.embedding_service.dimensions] * len(texts)
        
        embeddings = []
        for text in texts:
            embedding = self.embedding_service.create_document_embedding(text)
            if embedding:
                embeddings.append(embedding)
            else:
                # Fallback to zero vector
                embeddings.append([0.0] * self.embedding_service.dimensions)
        
        return embeddings

# Global vector store instance
try:
    vector_store = VectorStore()
    logger.info("Vector store initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize vector store: {e}")
    vector_store = None

# Convenience functions
def initialize_vector_store():
    """Initialize and return the vector store"""
    global vector_store
    if vector_store is None:
        try:
            vector_store = VectorStore()
            logger.info("Vector store initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            return None
    return vector_store

def add_documents_to_store(documents: List[Dict[str, Any]]) -> bool:
    """Add documents to the vector store"""
    if vector_store:
        return vector_store.add_documents(documents)
    else:
        logger.error("Vector store not initialized")
        return False

def search_documents(query: str, n_results: int = None) -> List[Dict[str, Any]]:
    """Search for documents in the vector store"""
    if vector_store:
        return vector_store.search(query, n_results)
    else:
        logger.error("Vector store not initialized")
        return []