"""
UbuntuAI Embeddings Service

This module provides embedding functionality for text processing in the UbuntuAI system.
It handles Google Gemini embeddings with proper error handling and retry logic.
"""

import google.generativeai as genai
import numpy as np
import logging
from typing import List, Dict, Any, Optional
import time
import re
from config.settings import settings

# Configure logging
logger = logging.getLogger(__name__)

class EmbeddingError(Exception):
    """Custom exception for embedding-related errors"""
    pass

class EmbeddingService:
    """
    Service for creating and managing text embeddings using Google Gemini API.
    
    This service handles:
    - Text preprocessing and truncation
    - Embedding creation with retry logic
    - Batch processing for efficiency
    - Error handling and logging
    - Multiple task types for different use cases
    """
    
    def __init__(self):
        """Initialize the embedding service with Google Gemini client and configuration"""
        try:
            if not settings.GOOGLE_API_KEY:
                raise EmbeddingError("Google Gemini API key not configured")
                
            genai.configure(api_key=settings.GOOGLE_API_KEY)
            
            # Get configuration from settings
            self.model_name = settings.EMBEDDING_MODEL
            self.dimensions = settings.EMBEDDING_DIMENSIONS
            self.task_type = settings.EMBEDDING_TASK_TYPE
            self.max_input_tokens = settings.MAX_INPUT_TOKENS
            
            # Rough character to token ratio for Gemini (approximately 1 token = 4 characters)
            self.char_to_token_ratio = 4
            self.max_input_chars = self.max_input_tokens * self.char_to_token_ratio
            
            logger.info(f"Gemini embedding service initialized with model: {self.model_name}")
            logger.info(f"Dimensions: {self.dimensions}, Task type: {self.task_type}")
            
        except Exception as e:
            logger.error(f"Failed to initialize embedding service: {e}")
            raise EmbeddingError(f"Initialization failed: {e}")
        
    def count_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in a text string.
        
        Args:
            text: Input text to count tokens for
            
        Returns:
            Estimated number of tokens in the text
        """
        try:
            if not isinstance(text, str):
                raise EmbeddingError("Input must be a string")
            
            # Simple estimation: 1 token ≈ 4 characters for most languages
            # This is a rough estimate since Gemini uses its own tokenizer
            return len(text) // self.char_to_token_ratio
            
        except Exception as e:
            logger.error(f"Error counting tokens: {e}")
            return len(text) // 4  # Fallback estimation
    
    def truncate_text(self, text: str, max_tokens: Optional[int] = None) -> str:
        """
        Truncate text to fit within token limits.
        
        Args:
            text: Input text to truncate
            max_tokens: Maximum number of tokens (uses default if not specified)
            
        Returns:
            Truncated text that fits within token limits
        """
        try:
            if not isinstance(text, str):
                raise EmbeddingError("Input must be a string")
                
            if max_tokens is None:
                max_tokens = self.max_input_tokens
                
            # Clean and normalize text
            text = self._clean_text(text)
            
            estimated_tokens = self.count_tokens(text)
            if estimated_tokens <= max_tokens:
                return text
                
            # Truncate to estimated character limit
            max_chars = max_tokens * self.char_to_token_ratio
            truncated_text = text[:max_chars]
            
            # Try to cut at sentence boundary
            truncated_text = self._truncate_at_sentence_boundary(truncated_text)
            
            if len(text) > len(truncated_text):
                logger.warning(f"Text truncated from {len(text)} to {len(truncated_text)} characters")
                
            return truncated_text
            
        except Exception as e:
            logger.error(f"Error truncating text: {e}")
            # Fallback: return first portion of text
            fallback_chars = (max_tokens or 2000) * 4
            return text[:fallback_chars]
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text for embedding"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove control characters but keep newlines
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        
        return text
    
    def _truncate_at_sentence_boundary(self, text: str) -> str:
        """Truncate text at the last complete sentence"""
        sentence_endings = ['. ', '! ', '? ', '.\n', '!\n', '?\n']
        
        for ending in sentence_endings:
            last_index = text.rfind(ending)
            if last_index > len(text) * 0.8:  # Only if we're keeping at least 80% of text
                return text[:last_index + len(ending)]
        
        return text
    
    def create_embedding(self, text: str, task_type: Optional[str] = None, retry_count: int = 3) -> Optional[List[float]]:
        """
        Create an embedding for a single text string using Gemini.
        
        Args:
            text: Input text to create embedding for
            task_type: Task type for embedding ("RETRIEVAL_QUERY", "RETRIEVAL_DOCUMENT", "SEMANTIC_SIMILARITY", etc.)
            retry_count: Number of retry attempts on failure
            
        Returns:
            List of embedding values or None if failed
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return None
            
        try:
            # Use provided task type or default
            embedding_task_type = task_type or self.task_type
            
            # Truncate text to fit within limits
            processed_text = self.truncate_text(text)
            
            for attempt in range(retry_count):
                try:
                    # Create embedding using Gemini
                    result = genai.embed_content(
                        model=self.model_name,
                        content=processed_text,
                        task_type=embedding_task_type,
                        output_dimensionality=self.dimensions if self.dimensions < 3072 else None
                    )
                    
                    if not result or 'embedding' not in result:
                        raise EmbeddingError("Empty response from Gemini API")
                    
                    embedding = result['embedding']
                    
                    if not embedding or len(embedding) == 0:
                        raise EmbeddingError("Empty embedding generated")
                    
                    # Validate embedding dimensions
                    expected_dims = self.dimensions
                    if len(embedding) != expected_dims:
                        logger.warning(f"Embedding dimension mismatch: expected {expected_dims}, got {len(embedding)}")
                        
                    logger.debug(f"Successfully created Gemini embedding with {len(embedding)} dimensions")
                    return embedding
                    
                except Exception as e:
                    logger.error(f"Gemini API error (attempt {attempt + 1}): {e}")
                    if attempt < retry_count - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    else:
                        raise EmbeddingError(f"All retry attempts failed: {e}")
                        
            return None
            
        except EmbeddingError:
            raise
        except Exception as e:
            logger.error(f"Fatal error in create_embedding: {e}")
            raise EmbeddingError(f"Embedding creation failed: {e}")
    
    def create_embeddings_batch(self, texts: List[str], task_type: Optional[str] = None, batch_size: int = 20) -> List[Optional[List[float]]]:
        """
        Create embeddings for multiple texts in batches.
        
        Args:
            texts: List of input texts
            task_type: Task type for embeddings
            batch_size: Number of texts to process in each batch
            
        Returns:
            List of embeddings (same order as input texts)
        """
        if not texts:
            logger.warning("Empty text list provided for batch embedding")
            return []
            
        try:
            embeddings = []
            total_batches = (len(texts) + batch_size - 1) // batch_size
            
            logger.info(f"Processing {len(texts)} texts in {total_batches} batches")
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_num = (i // batch_size) + 1
                
                logger.debug(f"Processing batch {batch_num}/{total_batches}")
                
                try:
                    # Process each text individually (Gemini embed_content processes one at a time)
                    batch_embeddings = []
                    for text in batch:
                        if text and text.strip():
                            try:
                                embedding = self.create_embedding(text, task_type)
                                batch_embeddings.append(embedding)
                                if embedding:
                                    logger.debug(f"Successfully created embedding for text: {text[:50]}...")
                                else:
                                    logger.warning(f"Failed to create embedding for text: {text[:50]}...")
                            except Exception as e:
                                logger.error(f"Failed to create embedding for text: {e}")
                                batch_embeddings.append(None)
                        else:
                            batch_embeddings.append(None)
                    
                    embeddings.extend(batch_embeddings)
                    
                    # Rate limiting between batches
                    if i + batch_size < len(texts):
                        time.sleep(0.1)
                        
                except Exception as e:
                    logger.error(f"Error in batch {batch_num}: {e}")
                    # Add None for failed batch
                    embeddings.extend([None] * len(batch))
            
            success_count = sum(1 for e in embeddings if e is not None)
            logger.info(f"Batch processing complete: {success_count}/{len(texts)} successful")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Fatal error in batch embedding: {e}")
            raise EmbeddingError(f"Batch embedding failed: {e}")
    
    def create_query_embedding(self, query: str) -> Optional[List[float]]:
        """
        Create an embedding optimized for search queries.
        
        Args:
            query: Search query text
            
        Returns:
            Query embedding or None if failed
        """
        return self.create_embedding(query, task_type="RETRIEVAL_QUERY")
    
    def create_document_embedding(self, document: str) -> Optional[List[float]]:
        """
        Create an embedding optimized for documents.
        
        Args:
            document: Document text
            
        Returns:
            Document embedding or None if failed
        """
        return self.create_embedding(document, task_type="RETRIEVAL_DOCUMENT")
    
    def cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """
        Calculate cosine similarity between two embedding vectors.
        
        Args:
            a: First embedding vector
            b: Second embedding vector
            
        Returns:
            Cosine similarity score between -1 and 1
        """
        try:
            if not a or not b:
                return 0.0
                
            if len(a) != len(b):
                raise EmbeddingError(f"Vector dimensions don't match: {len(a)} vs {len(b)}")
            
            a_np = np.array(a, dtype=np.float32)
            b_np = np.array(b, dtype=np.float32)
            
            dot_product = np.dot(a_np, b_np)
            norm_a = np.linalg.norm(a_np)
            norm_b = np.linalg.norm(b_np)
            
            if norm_a == 0 or norm_b == 0:
                logger.warning("Zero norm vector encountered in similarity calculation")
                return 0.0
                
            similarity = float(dot_product / (norm_a * norm_b))
            
            # Clamp to valid range
            similarity = max(-1.0, min(1.0, similarity))
            
            return similarity
            
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {e}")
            raise EmbeddingError(f"Similarity calculation failed: {e}")
    
    def semantic_search(self, query_embedding: List[float], 
                       document_embeddings: List[List[float]], 
                       top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Perform semantic search using embedding similarity.
        
        Args:
            query_embedding: Query vector to search with
            document_embeddings: List of document vectors to search in
            top_k: Number of top results to return
            
        Returns:
            List of search results with indices and similarity scores
        """
        try:
            if not query_embedding:
                raise EmbeddingError("Query embedding is empty")
                
            if not document_embeddings:
                logger.warning("No document embeddings provided for search")
                return []
            
            similarities = []
            
            for idx, doc_embedding in enumerate(document_embeddings):
                if doc_embedding is not None:
                    try:
                        similarity = self.cosine_similarity(query_embedding, doc_embedding)
                        similarities.append({
                            'index': idx,
                            'similarity': similarity
                        })
                    except Exception as e:
                        logger.warning(f"Error calculating similarity for document {idx}: {e}")
                        continue
            
            if not similarities:
                logger.warning("No valid similarities calculated")
                return []
            
            # Sort by similarity (highest first)
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            
            # Return top k results
            results = similarities[:top_k]
            
            logger.debug(f"Semantic search returned {len(results)} results")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            raise EmbeddingError(f"Semantic search failed: {e}")
    
    def get_embedding_stats(self, embeddings: List[List[float]]) -> Dict[str, Any]:
        """
        Calculate statistics for a collection of embeddings.
        
        Args:
            embeddings: List of embedding vectors
            
        Returns:
            Dictionary containing embedding statistics
        """
        try:
            valid_embeddings = [e for e in embeddings if e is not None]
            
            stats = {
                "count": len(embeddings),
                "valid": len(valid_embeddings),
                "invalid": len(embeddings) - len(valid_embeddings),
                "dimensions": 0,
                "mean_magnitude": 0.0,
                "std_magnitude": 0.0
            }
            
            if not valid_embeddings:
                return stats
            
            embeddings_array = np.array(valid_embeddings, dtype=np.float32)
            
            stats.update({
                "dimensions": embeddings_array.shape[1] if len(embeddings_array.shape) > 1 else 0,
                "mean_magnitude": float(np.mean(np.linalg.norm(embeddings_array, axis=1))),
                "std_magnitude": float(np.std(np.linalg.norm(embeddings_array, axis=1)))
            })
            
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating embedding stats: {e}")
            return {
                "count": len(embeddings) if embeddings else 0,
                "valid": 0,
                "invalid": len(embeddings) if embeddings else 0,
                "error": str(e)
            }

class BusinessContextEmbeddings:
    """
    Enhanced embedding service that incorporates business context into embeddings.
    
    This service adds metadata context to improve embedding quality for
    business-specific use cases in the African context.
    """
    
    def __init__(self, embedding_service: EmbeddingService):
        """
        Initialize with an embedding service instance.
        
        Args:
            embedding_service: Base embedding service to use
        """
        self.embedding_service = embedding_service
        logger.info("Business context embeddings service initialized")
        
    def create_context_aware_embedding(self, text: str, metadata: Dict[str, Any]) -> Optional[List[float]]:
        """
        Create an embedding with added business context.
        
        Args:
            text: Primary text content
            metadata: Business context metadata
            
        Returns:
            Context-enhanced embedding vector
        """
        try:
            enhanced_text = self._enhance_text_with_context(text, metadata)
            return self.embedding_service.create_document_embedding(enhanced_text)
            
        except Exception as e:
            logger.error(f"Error creating context-aware embedding: {e}")
            # Fallback to basic embedding
            return self.embedding_service.create_document_embedding(text)
    
    def _enhance_text_with_context(self, text: str, metadata: Dict[str, Any]) -> str:
        """
        Enhance text with business context from metadata.
        
        Args:
            text: Original text
            metadata: Context metadata
            
        Returns:
            Text enhanced with context information
        """
        try:
            if not text:
                return ""
                
            enhancements = []
            
            # Add context from metadata
            context_fields = {
                'country': 'Country',
                'sector': 'Business Sector', 
                'funding_stage': 'Funding Stage',
                'source_type': 'Source Type',
                'date': 'Date',
                'type': 'Content Type'
            }
            
            for field, label in context_fields.items():
                value = metadata.get(field)
                if value and str(value).strip():
                    if isinstance(value, list):
                        value = ", ".join(str(v) for v in value)
                    enhancements.append(f"{label}: {value}")
            
            if enhancements:
                context_string = " | ".join(enhancements)
                return f"{text}\n\nContext: {context_string}"
            
            return text
            
        except Exception as e:
            logger.warning(f"Error enhancing text with context: {e}")
            return text

# Global service instances
try:
    embedding_service = EmbeddingService()
    business_context_embeddings = BusinessContextEmbeddings(embedding_service)
    logger.info("Embedding services initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize embedding services: {e}")
    # Create fallback services that will raise errors when used
    embedding_service = None
    business_context_embeddings = None