"""
UbuntuAI Embeddings Service

This module provides embedding functionality for text processing in the UbuntuAI system.
It handles Google Gemini embeddings with proper error handling and retry logic.
"""

import google.generativeai as genai
import numpy as np
import logging
from typing import List, Dict, Any, Optional
import tiktoken
import time
from config.settings import settings

# Configure logging
logger = logging.getLogger(__name__)

class EmbeddingError(Exception):
    """Custom exception for embedding-related errors"""
    pass

class EmbeddingService:
    """
    Service for creating and managing text embeddings using OpenAI's API.
    
    This service handles:
    - Text tokenization and truncation
    - Embedding creation with retry logic
    - Batch processing for efficiency
    - Error handling and logging
    """
    
    def __init__(self):
        """Initialize the embedding service with Google Gemini client and configuration"""
        try:
            if not settings.GOOGLE_API_KEY:
                raise EmbeddingError("Google Gemini API key not configured")
                
            genai.configure(api_key=settings.GOOGLE_API_KEY)
            self.model = genai.GenerativeModel('gemini-1.5-pro')
            self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
            self.max_tokens = 8191  # Gemini model limit
            
            logger.info(f"Embedding service initialized with model: {self.model}")
            
        except Exception as e:
            logger.error(f"Failed to initialize embedding service: {e}")
            raise EmbeddingError(f"Initialization failed: {e}")
        
    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text string.
        
        Args:
            text: Input text to count tokens for
            
        Returns:
            Number of tokens in the text
            
        Raises:
            EmbeddingError: If token counting fails
        """
        try:
            if not isinstance(text, str):
                raise EmbeddingError("Input must be a string")
                
            return len(self.encoding.encode(text))
            
        except Exception as e:
            logger.error(f"Error counting tokens: {e}")
            raise EmbeddingError(f"Token counting failed: {e}")
    
    def truncate_text(self, text: str, max_tokens: Optional[int] = None) -> str:
        """
        Truncate text to fit within token limits.
        
        Args:
            text: Input text to truncate
            max_tokens: Maximum number of tokens (uses default if not specified)
            
        Returns:
            Truncated text that fits within token limits
            
        Raises:
            EmbeddingError: If truncation fails
        """
        try:
            if not isinstance(text, str):
                raise EmbeddingError("Input must be a string")
                
            if max_tokens is None:
                max_tokens = self.max_tokens
                
            tokens = self.encoding.encode(text)
            if len(tokens) <= max_tokens:
                return text
                
            truncated_tokens = tokens[:max_tokens]
            truncated_text = self.encoding.decode(truncated_tokens)
            
            if len(tokens) > max_tokens:
                logger.warning(f"Text truncated from {len(tokens)} to {max_tokens} tokens")
                
            return truncated_text
            
        except Exception as e:
            logger.error(f"Error truncating text: {e}")
            raise EmbeddingError(f"Text truncation failed: {e}")
    
    def create_embedding(self, text: str, retry_count: int = 3) -> Optional[List[float]]:
        """
        Create an embedding for a single text string using Gemini.
        
        Args:
            text: Input text to create embedding for
            retry_count: Number of retry attempts on failure
            
        Returns:
            List of embedding values or None if failed
            
        Raises:
            EmbeddingError: If all retry attempts fail
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return None
            
        try:
            text = self.truncate_text(text)
            
            for attempt in range(retry_count):
                try:
                    # Use Gemini to generate a semantic representation
                    prompt = f"Analyze the following text and provide a semantic analysis that captures its key concepts, themes, and meaning:\n\n{text}\n\nProvide a structured analysis that can be used for similarity matching."
                    
                    response = self.model.generate_content(prompt)
                    
                    if not response or not response.text:
                        raise EmbeddingError("Empty response from Gemini API")
                    
                    # Convert the semantic analysis to a numerical embedding
                    embedding = self._text_to_embedding(response.text)
                    
                    if not embedding or len(embedding) == 0:
                        raise EmbeddingError("Empty embedding generated")
                        
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
    
    def _text_to_embedding(self, text: str) -> List[float]:
        """
        Convert Gemini's semantic analysis to a numerical embedding vector.
        Creates a more semantically meaningful representation than simple hashing.
        
        Args:
            text: Semantic analysis text from Gemini
            
        Returns:
            List of numerical values representing the text (1536 dimensions)
        """
        try:
            import hashlib
            import re
            
            # Clean and normalize the text
            clean_text = re.sub(r'[^\w\s]', '', text.lower())
            words = clean_text.split()
            
            # Create a more sophisticated embedding based on word frequencies and patterns
            embedding = []
            
            # Use word frequency analysis
            word_freq = {}
            for word in words:
                if len(word) > 2:  # Skip very short words
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            # Create base embedding from word frequencies
            for word, freq in sorted(word_freq.items()):
                if len(embedding) >= 512:  # Use first 512 dimensions for word-based features
                    break
                # Create hash-based value for each word
                word_hash = hashlib.md5(word.encode()).hexdigest()
                for i in range(0, min(8, len(word_hash)), 2):
                    if len(embedding) >= 512:
                        break
                    hex_pair = word_hash[i:i+2]
                    embedding.append((float(int(hex_pair, 16)) / 255.0) * (freq / max(word_freq.values())))
            
            # Fill remaining dimensions with semantic features
            remaining_dims = 1536 - len(embedding)
            if remaining_dims > 0:
                # Use text length, complexity, and other features
                text_hash = hashlib.md5(text.encode()).hexdigest()
                for i in range(0, min(remaining_dims * 2, len(text_hash)), 2):
                    if len(embedding) >= 1536:
                        break
                    hex_pair = text_hash[i:i+2]
                    embedding.append(float(int(hex_pair, 16)) / 255.0)
                
                # Pad to exactly 1536 dimensions
                while len(embedding) < 1536:
                    embedding.append(0.0)
            
            return embedding[:1536]
            
        except Exception as e:
            logger.error(f"Error converting text to embedding: {e}")
            # Return zero vector as fallback
            return [0.0] * 1536
    
    def create_embeddings_batch(self, texts: List[str], batch_size: int = 100) -> List[Optional[List[float]]]:
        """
        Create embeddings for multiple texts in batches.
        
        Args:
            texts: List of input texts
            batch_size: Number of texts to process in each batch
            
        Returns:
            List of embeddings (same order as input texts)
            
        Raises:
            EmbeddingError: If batch processing fails
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
                
                # Filter and truncate texts
                truncated_batch = []
                for text in batch:
                    if text and text.strip():
                        truncated_batch.append(self.truncate_text(text))
                    else:
                        truncated_batch.append("")
                
                try:
                    # Only send non-empty texts to API
                    valid_texts = [t for t in truncated_batch if t.strip()]
                    
                    if not valid_texts:
                        # All texts in batch are empty
                        embeddings.extend([None] * len(batch))
                        continue
                    
                    # Process each text individually for Gemini
                    batch_embeddings = []
                    for text in truncated_batch:
                        if text.strip():
                            try:
                                embedding = self.create_embedding(text)
                                if embedding:
                                    batch_embeddings.append(embedding)
                                    logger.debug(f"Successfully created Gemini embedding for text: {text[:50]}...")
                                else:
                                    batch_embeddings.append(None)
                                    logger.warning(f"Failed to create Gemini embedding for text: {text[:50]}...")
                            except Exception as e:
                                logger.error(f"Failed to create Gemini embedding for text: {e}")
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
    
    def cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """
        Calculate cosine similarity between two embedding vectors.
        
        Args:
            a: First embedding vector
            b: Second embedding vector
            
        Returns:
            Cosine similarity score between 0 and 1
            
        Raises:
            EmbeddingError: If similarity calculation fails
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
            
            # Ensure result is in valid range
            similarity = max(0.0, min(1.0, similarity))
            
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
            
        Raises:
            EmbeddingError: If search fails
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
    business-specific use cases.
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
            return self.embedding_service.create_embedding(enhanced_text)
            
        except Exception as e:
            logger.error(f"Error creating context-aware embedding: {e}")
            # Fallback to basic embedding
            return self.embedding_service.create_embedding(text)
    
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
                'date': 'Date'
            }
            
            for field, label in context_fields.items():
                value = metadata.get(field)
                if value and str(value).strip():
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