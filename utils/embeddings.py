"""
Modern Multi-Provider Embedding Manager for UbuntuAI
Supports OpenAI, Sentence Transformers, Google, Cohere with LangChain integration
"""

import logging
from typing import List, Dict, Any, Optional, Union
from abc import ABC, abstractmethod
import numpy as np
import time
import asyncio

# LangChain Embeddings
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import SentenceTransformerEmbeddings

# Sentence Transformers
try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False

# Cohere
try:
    import cohere
    from langchain_cohere import CohereEmbeddings
    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False

# HuggingFace Transformers
try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

from config.settings import settings

logger = logging.getLogger(__name__)

class BaseEmbeddingProvider(ABC):
    """Abstract base class for embedding providers"""
    
    def __init__(self, provider_name: str):
        self.provider_name = provider_name
        self.is_available = False
        self.model = None
        self.dimensions = 0
        self._initialize()
    
    @abstractmethod
    def _initialize(self):
        """Initialize the embedding provider"""
        pass
    
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        pass
    
    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        pass
    
    def get_langchain_embeddings(self) -> Optional[Embeddings]:
        """Get LangChain-compatible embeddings"""
        return self.model
    
    def get_info(self) -> Dict[str, Any]:
        """Get provider information"""
        return {
            "provider": self.provider_name,
            "is_available": self.is_available,
            "dimensions": self.dimensions,
            "model_name": getattr(self.model, 'model', 'unknown') if self.model else None
        }

class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """OpenAI embedding provider"""
    
    def _initialize(self):
        try:
            if not settings.OPENAI_API_KEY:
                logger.warning("OpenAI API key not configured")
                return
            
            config = settings.get_embedding_config("openai")
            
            self.model = OpenAIEmbeddings(
                api_key=settings.OPENAI_API_KEY,
                model=config["model"],
                dimensions=config["dimensions"]
            )
            
            self.dimensions = config["dimensions"]
            self.is_available = True
            
            logger.info(f"OpenAI embeddings initialized: {config['model']} ({self.dimensions}D)")
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI embeddings: {e}")
            self.is_available = False
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not self.is_available:
            raise ValueError("OpenAI embeddings not available")
        return self.model.embed_documents(texts)
    
    def embed_query(self, text: str) -> List[float]:
        if not self.is_available:
            raise ValueError("OpenAI embeddings not available")
        return self.model.embed_query(text)

class SentenceTransformerProvider(BaseEmbeddingProvider):
    """Sentence Transformers embedding provider"""
    
    def _initialize(self):
        try:
            if not ST_AVAILABLE:
                logger.warning("Sentence Transformers not available")
                return
            
            config = settings.get_embedding_config("sentence-transformers")
            model_name = config["model"]
            
            # Initialize the sentence transformer model directly
            self.sentence_model = SentenceTransformer(
                model_name,
                device=config.get("device", "cpu")
            )
            
            # Create LangChain wrapper
            self.model = SentenceTransformerEmbeddings(
                model_name=model_name,
                model_kwargs={'device': config.get("device", "cpu")},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            self.dimensions = config["dimensions"]
            self.is_available = True
            
            logger.info(f"Sentence Transformers initialized: {model_name} ({self.dimensions}D)")
            
        except Exception as e:
            logger.error(f"Failed to initialize Sentence Transformers: {e}")
            self.is_available = False
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not self.is_available:
            raise ValueError("Sentence Transformers not available")
        return self.model.embed_documents(texts)
    
    def embed_query(self, text: str) -> List[float]:
        if not self.is_available:
            raise ValueError("Sentence Transformers not available")
        return self.model.embed_query(text)
    
    def encode_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Batch encoding with custom batch size"""
        if not self.is_available:
            raise ValueError("Sentence Transformers not available")
        
        embeddings = self.sentence_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100,
            normalize_embeddings=True
        )
        
        return embeddings.tolist()

class GoogleEmbeddingProvider(BaseEmbeddingProvider):
    """Google Gemini embedding provider"""
    
    def _initialize(self):
        try:
            if not settings.GOOGLE_API_KEY:
                logger.warning("Google API key not configured")
                return
            
            config = settings.get_embedding_config("google")
            
            self.model = GoogleGenerativeAIEmbeddings(
                google_api_key=settings.GOOGLE_API_KEY,
                model=config["model"]
            )
            
            self.dimensions = config["dimensions"]
            self.is_available = True
            
            logger.info(f"Google embeddings initialized: {config['model']} ({self.dimensions}D)")
            
        except Exception as e:
            logger.error(f"Failed to initialize Google embeddings: {e}")
            self.is_available = False
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not self.is_available:
            raise ValueError("Google embeddings not available")
        return self.model.embed_documents(texts)
    
    def embed_query(self, text: str) -> List[float]:
        if not self.is_available:
            raise ValueError("Google embeddings not available")
        return self.model.embed_query(text)

class CohereEmbeddingProvider(BaseEmbeddingProvider):
    """Cohere embedding provider"""
    
    def _initialize(self):
        try:
            if not COHERE_AVAILABLE:
                logger.warning("Cohere not available")
                return
            
            if not settings.COHERE_API_KEY:
                logger.warning("Cohere API key not configured")
                return
            
            config = settings.get_embedding_config("cohere")
            
            self.model = CohereEmbeddings(
                cohere_api_key=settings.COHERE_API_KEY,
                model=config["model"]
            )
            
            self.dimensions = config["dimensions"]
            self.is_available = True
            
            logger.info(f"Cohere embeddings initialized: {config['model']} ({self.dimensions}D)")
            
        except Exception as e:
            logger.error(f"Failed to initialize Cohere embeddings: {e}")
            self.is_available = False
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not self.is_available:
            raise ValueError("Cohere embeddings not available")
        return self.model.embed_documents(texts)
    
    def embed_query(self, text: str) -> List[float]:
        if not self.is_available:
            raise ValueError("Cohere embeddings not available")
        return self.model.embed_query(text)

class HuggingFaceEmbeddingProvider(BaseEmbeddingProvider):
    """HuggingFace Transformers embedding provider"""
    
    def _initialize(self):
        try:
            if not HF_AVAILABLE:
                logger.warning("HuggingFace Transformers not available")
                return
            
            # Use BGE model as default
            model_name = "BAAI/bge-large-en-v1.5"
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.hf_model = AutoModel.from_pretrained(model_name)
            
            # Set to evaluation mode
            self.hf_model.eval()
            
            # Create a simple wrapper for LangChain compatibility
            self.model = self
            self.dimensions = 1024  # BGE-large dimensions
            self.is_available = True
            
            logger.info(f"HuggingFace embeddings initialized: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize HuggingFace embeddings: {e}")
            self.is_available = False
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not self.is_available:
            raise ValueError("HuggingFace embeddings not available")
        
        embeddings = []
        for text in texts:
            embedding = self._encode_single(text)
            embeddings.append(embedding)
        
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        if not self.is_available:
            raise ValueError("HuggingFace embeddings not available")
        
        return self._encode_single(text)
    
    def _encode_single(self, text: str) -> List[float]:
        """Encode a single text using HuggingFace model"""
        try:
            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.hf_model(**inputs)
                # Use CLS token embedding
                embedding = outputs.last_hidden_state[:, 0, :].squeeze()
                # Normalize
                embedding = torch.nn.functional.normalize(embedding, p=2, dim=0)
                
            return embedding.numpy().tolist()
            
        except Exception as e:
            logger.error(f"HuggingFace encoding failed: {e}")
            return [0.0] * self.dimensions

class ModernEmbeddingManager:
    """
    Modern embedding manager with multi-provider support and fallback
    """
    
    def __init__(self):
        self.providers = {}
        self.primary_provider = settings.PRIMARY_EMBEDDING_PROVIDER
        self.fallback_order = ["sentence-transformers", "openai", "google", "cohere", "huggingface"]
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize all available embedding providers"""
        
        provider_classes = {
            "openai": OpenAIEmbeddingProvider,
            "sentence-transformers": SentenceTransformerProvider,
            "google": GoogleEmbeddingProvider,
            "cohere": CohereEmbeddingProvider,
            "huggingface": HuggingFaceEmbeddingProvider
        }
        
        for provider_name, provider_class in provider_classes.items():
            try:
                provider = provider_class(provider_name)
                self.providers[provider_name] = provider
                status = "Available" if provider.is_available else "Unavailable"
                logger.info(f"Embedding provider {provider_name}: {status}")
            except Exception as e:
                logger.error(f"Failed to initialize {provider_name} embeddings: {e}")
    
    def get_provider(self, provider_name: str = None) -> Optional[BaseEmbeddingProvider]:
        """Get a specific embedding provider"""
        
        if provider_name and provider_name in self.providers:
            provider = self.providers[provider_name]
            if provider.is_available:
                return provider
            else:
                logger.warning(f"Provider {provider_name} not available")
        
        # Use primary provider
        if self.primary_provider in self.providers and self.providers[self.primary_provider].is_available:
            return self.providers[self.primary_provider]
        
        # Fallback to any available provider
        for provider_name in self.fallback_order:
            if provider_name in self.providers and self.providers[provider_name].is_available:
                logger.warning(f"Falling back to {provider_name} embeddings")
                return self.providers[provider_name]
        
        logger.error("No embedding providers available")
        return None
    
    def get_langchain_embeddings(self, provider: str = None) -> Optional[Embeddings]:
        """Get LangChain-compatible embeddings instance"""
        embedding_provider = self.get_provider(provider)
        if embedding_provider:
            return embedding_provider.get_langchain_embeddings()
        return None
    
    def embed_documents(self, texts: List[str], provider: str = None) -> List[List[float]]:
        """Embed multiple documents"""
        embedding_provider = self.get_provider(provider)
        if not embedding_provider:
            raise ValueError("No embedding provider available")
        
        try:
            return embedding_provider.embed_documents(texts)
        except Exception as e:
            logger.error(f"Document embedding failed: {e}")
            raise
    
    def embed_query(self, text: str, provider: str = None) -> List[float]:
        """Embed a single query"""
        embedding_provider = self.get_provider(provider)
        if not embedding_provider:
            raise ValueError("No embedding provider available")
        
        try:
            return embedding_provider.embed_query(text)
        except Exception as e:
            logger.error(f"Query embedding failed: {e}")
            raise
    
    def embed_documents_batch(self, 
                            texts: List[str], 
                            batch_size: int = 32,
                            provider: str = None) -> List[List[float]]:
        """Embed documents in batches for efficiency"""
        
        embedding_provider = self.get_provider(provider)
        if not embedding_provider:
            raise ValueError("No embedding provider available")
        
        # Use specialized batch method if available
        if hasattr(embedding_provider, 'encode_batch'):
            return embedding_provider.encode_batch(texts, batch_size)
        
        # Fallback to chunked processing
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = embedding_provider.embed_documents(batch)
            all_embeddings.extend(batch_embeddings)
            
            # Small delay to avoid rate limiting
            if i + batch_size < len(texts):
                time.sleep(0.1)
        
        return all_embeddings
    
    def calculate_similarity(self, 
                           embedding1: List[float], 
                           embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings"""
        
        try:
            a = np.array(embedding1, dtype=np.float32)
            b = np.array(embedding2, dtype=np.float32)
            
            # Calculate cosine similarity
            dot_product = np.dot(a, b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            
            if norm_a == 0 or norm_b == 0:
                return 0.0
            
            similarity = dot_product / (norm_a * norm_b)
            return float(np.clip(similarity, -1.0, 1.0))
            
        except Exception as e:
            logger.error(f"Similarity calculation failed: {e}")
            return 0.0
    
    def find_most_similar(self, 
                         query_embedding: List[float],
                         document_embeddings: List[List[float]],
                         top_k: int = 10) -> List[Dict[str, Any]]:
        """Find most similar documents to query"""
        
        similarities = []
        for i, doc_embedding in enumerate(document_embeddings):
            similarity = self.calculate_similarity(query_embedding, doc_embedding)
            similarities.append({
                "index": i,
                "similarity": similarity
            })
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        
        return similarities[:top_k]
    
    def get_available_providers(self) -> List[str]:
        """Get list of available providers"""
        return [name for name, provider in self.providers.items() if provider.is_available]
    
    def get_provider_info(self, provider: str = None) -> Dict[str, Any]:
        """Get detailed provider information"""
        
        if provider:
            if provider in self.providers:
                return self.providers[provider].get_info()
            else:
                return {"error": f"Provider {provider} not found"}
        
        # Return info for all providers
        info = {
            "primary_provider": self.primary_provider,
            "available_providers": self.get_available_providers(),
            "provider_details": {}
        }
        
        for name, provider in self.providers.items():
            info["provider_details"][name] = provider.get_info()
        
        return info
    
    def switch_provider(self, provider_name: str) -> bool:
        """Switch to a different embedding provider"""
        
        if provider_name in self.providers and self.providers[provider_name].is_available:
            self.primary_provider = provider_name
            logger.info(f"Switched to {provider_name} embeddings")
            return True
        else:
            logger.error(f"Cannot switch to {provider_name} - not available")
            return False
    
    def validate_embeddings(self, embeddings: List[List[float]]) -> Dict[str, Any]:
        """Validate embedding quality and consistency"""
        
        if not embeddings:
            return {"valid": False, "error": "No embeddings provided"}
        
        try:
            # Check dimensions consistency
            dimensions = [len(emb) for emb in embeddings]
            if len(set(dimensions)) > 1:
                return {
                    "valid": False,
                    "error": "Inconsistent embedding dimensions",
                    "dimensions": dimensions
                }
            
            # Check for NaN or infinite values
            embeddings_array = np.array(embeddings)
            if np.any(np.isnan(embeddings_array)) or np.any(np.isinf(embeddings_array)):
                return {
                    "valid": False,
                    "error": "Contains NaN or infinite values"
                }
            
            # Calculate statistics
            norms = np.linalg.norm(embeddings_array, axis=1)
            
            return {
                "valid": True,
                "count": len(embeddings),
                "dimensions": dimensions[0],
                "norm_stats": {
                    "mean": float(np.mean(norms)),
                    "std": float(np.std(norms)),
                    "min": float(np.min(norms)),
                    "max": float(np.max(norms))
                }
            }
            
        except Exception as e:
            return {
                "valid": False,
                "error": f"Validation failed: {str(e)}"
            }

# Global embedding manager instance
try:
    embedding_manager = ModernEmbeddingManager()
    logger.info(f"Modern embedding manager initialized with {len(embedding_manager.get_available_providers())} providers")
except Exception as e:
    logger.error(f"Failed to initialize embedding manager: {e}")
    embedding_manager = None

# For backward compatibility
embedding_service = embedding_manager