#!/usr/bin/env python3
"""
Embedding service for UbuntuAI
Handles text embeddings using Google's embedding model
"""

import os
import logging
from typing import List, Optional, Union
import numpy as np
from google.generativeai import embed
from config.settings import settings

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Service for creating and managing text embeddings"""
    
    def __init__(self):
        """Initialize the embedding service"""
        self.model_name = settings.EMBEDDING_MODEL
        self.dimensions = settings.EMBEDDING_DIMENSIONS
        self.task_type = "retrieval_query"
        
        # Configure Google API
        if not settings.GOOGLE_API_KEY:
            logger.error("Google API key not configured")
            self._initialized = False
            return
        
        try:
            # Test the embedding service
            test_embedding = self.create_embedding("test")
            if test_embedding is not None:
                logger.info(f"Embedding service initialized successfully with model: {self.model_name}")
                self._initialized = True
            else:
                logger.error("Failed to create test embedding")
                self._initialized = False
        except Exception as e:
            logger.error(f"Failed to initialize embedding service: {e}")
            self._initialized = False
    
    def is_initialized(self) -> bool:
        """Check if the embedding service is properly initialized"""
        return self._initialized
    
    def create_embedding(self, text: str, task_type: str = "retrieval_query") -> Optional[List[float]]:
        """
        Create an embedding for a single text
        
        Args:
            text: Text to embed
            task_type: Type of task (retrieval_query, retrieval_document, etc.)
            
        Returns:
            List of float values representing the embedding, or None if failed
        """
        if not self._initialized:
            logger.error("Embedding service not initialized")
            return None
        
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return None
        
        try:
            # Create embedding using Google's model
            result = embed(
                model=self.model_name,
                content=text,
                task_type=task_type
            )
            
            if result and hasattr(result, 'embedding'):
                embedding = result.embedding
                logger.debug(f"Created embedding for text of length {len(text)}")
                return embedding
            else:
                logger.error("Failed to get embedding from Google API response")
                return None
                
        except Exception as e:
            logger.error(f"Error creating embedding: {e}")
            return None
    
    def create_query_embedding(self, text: str) -> Optional[List[float]]:
        """Create embedding optimized for query retrieval"""
        return self.create_embedding(text, "retrieval_query")
    
    def create_document_embedding(self, text: str) -> Optional[List[float]]:
        """Create embedding optimized for document storage"""
        return self.create_embedding(text, "retrieval_document")
    
    def create_embeddings_batch(self, texts: List[str], task_type: str = "retrieval_document") -> List[Optional[List[float]]]:
        """
        Create embeddings for multiple texts
        
        Args:
            texts: List of texts to embed
            task_type: Type of task for all embeddings
            
        Returns:
            List of embeddings (some may be None if failed)
        """
        if not self._initialized:
            logger.error("Embedding service not initialized")
            return [None] * len(texts)
        
        if not texts:
            return []
        
        embeddings = []
        for text in texts:
            embedding = self.create_embedding(text, task_type)
            embeddings.append(embedding)
        
        successful_count = sum(1 for e in embeddings if e is not None)
        logger.info(f"Created {successful_count}/{len(texts)} embeddings successfully")
        
        return embeddings
    
    def cosine_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score between 0 and 1
        """
        if not embedding1 or not embedding2:
            return 0.0
        
        try:
            # Convert to numpy arrays
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Calculate cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {e}")
            return 0.0
    
    def euclidean_distance(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate Euclidean distance between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Euclidean distance (lower is more similar)
        """
        if not embedding1 or not embedding2:
            return float('inf')
        
        try:
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            distance = np.linalg.norm(vec1 - vec2)
            return float(distance)
            
        except Exception as e:
            logger.error(f"Error calculating Euclidean distance: {e}")
            return float('inf')
    
    def normalize_embedding(self, embedding: List[float]) -> List[float]:
        """
        Normalize an embedding vector to unit length
        
        Args:
            embedding: Input embedding vector
            
        Returns:
            Normalized embedding vector
        """
        if not embedding:
            return []
        
        try:
            vec = np.array(embedding)
            norm = np.linalg.norm(vec)
            
            if norm == 0:
                return embedding
            
            normalized = vec / norm
            return normalized.tolist()
            
        except Exception as e:
            logger.error(f"Error normalizing embedding: {e}")
            return embedding
    
    def get_embedding_dimensions(self) -> int:
        """Get the dimensionality of embeddings"""
        return self.dimensions
    
    def get_model_info(self) -> dict:
        """Get information about the embedding model"""
        return {
            "model_name": self.model_name,
            "dimensions": self.dimensions,
            "task_type": self.task_type,
            "initialized": self._initialized,
            "provider": "google"
        }

# Create global embedding service instance
try:
    embedding_service = EmbeddingService()
    if not embedding_service.is_initialized():
        logger.warning("Embedding service failed to initialize - some features may not work")
        embedding_service = None
except Exception as e:
    logger.error(f"Failed to create embedding service: {e}")
    embedding_service = None

# Fallback embedding function for compatibility
def create_embedding(text: str, task_type: str = "retrieval_query") -> Optional[List[float]]:
    """Fallback function for creating embeddings"""
    if embedding_service:
        return embedding_service.create_embedding(text, task_type)
    else:
        logger.error("Embedding service not available")
        return None

def create_embeddings_batch(texts: List[str], task_type: str = "retrieval_document") -> List[Optional[List[float]]]:
    """Fallback function for creating batch embeddings"""
    if embedding_service:
        return embedding_service.create_embeddings_batch(texts, task_type)
    else:
        logger.error("Embedding service not available")
        return [None] * len(texts)