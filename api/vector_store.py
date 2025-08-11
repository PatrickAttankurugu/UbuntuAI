"""
Modern Multi-Provider Vector Store for UbuntuAI
Supports ChromaDB, Pinecone, FAISS with LangChain integration
"""

import logging
from typing import List, Dict, Any, Optional, Union
from abc import ABC, abstractmethod
import json
import os
from datetime import datetime

# LangChain Vector Store imports
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

# Vector store specific imports
try:
    from langchain_pinecone import PineconeVectorStore
    import pinecone
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False

import chromadb
from chromadb.config import Settings as ChromaSettings

from config.settings import settings
from utils.embeddings import ModernEmbeddingManager

logger = logging.getLogger(__name__)

class BaseVectorStoreProvider(ABC):
    """Abstract base class for vector store providers"""
    
    def __init__(self, embedding_provider: Embeddings):
        self.embedding_provider = embedding_provider
        self.vector_store = None
        self.is_available = False
        self._initialize()
    
    @abstractmethod
    def _initialize(self):
        """Initialize the vector store provider"""
        pass
    
    @abstractmethod
    def add_documents(self, documents: List[Document], **kwargs) -> List[str]:
        """Add documents to vector store"""
        pass
    
    @abstractmethod
    def similarity_search(self, query: str, k: int = 10, **kwargs) -> List[Document]:
        """Perform similarity search"""
        pass
    
    @abstractmethod
    def similarity_search_with_score(self, query: str, k: int = 10, **kwargs) -> List[tuple]:
        """Perform similarity search with scores"""
        pass
    
    def get_langchain_vectorstore(self) -> Optional[VectorStore]:
        """Get LangChain-compatible vector store"""
        return self.vector_store
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        return {
            "provider": self.__class__.__name__,
            "is_available": self.is_available,
            "vector_store_type": type(self.vector_store).__name__ if self.vector_store else None
        }

class ChromaVectorStoreProvider(BaseVectorStoreProvider):
    """ChromaDB vector store provider"""
    
    def _initialize(self):
        try:
            # Setup ChromaDB client
            self.persist_directory = settings.CHROMA_PERSIST_DIRECTORY
            os.makedirs(self.persist_directory, exist_ok=True)
            
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Initialize LangChain Chroma vector store
            self.vector_store = Chroma(
                client=self.client,
                collection_name=settings.COLLECTION_NAME,
                embedding_function=self.embedding_provider,
                persist_directory=self.persist_directory
            )
            
            self.is_available = True
            logger.info(f"ChromaDB initialized at {self.persist_directory}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            self.is_available = False
    
    def add_documents(self, documents: List[Document], **kwargs) -> List[str]:
        """Add documents to ChromaDB"""
        if not self.is_available:
            raise ValueError("ChromaDB not available")
        
        try:
            ids = kwargs.get('ids')
            if not ids:
                ids = [f"doc_{i}_{hash(doc.page_content) % 10000}" for i, doc in enumerate(documents)]
            
            # Add documents using LangChain interface
            added_ids = self.vector_store.add_documents(documents, ids=ids)
            
            logger.info(f"Added {len(documents)} documents to ChromaDB")
            return added_ids
            
        except Exception as e:
            logger.error(f"Failed to add documents to ChromaDB: {e}")
            raise
    
    def similarity_search(self, query: str, k: int = 10, **kwargs) -> List[Document]:
        """Perform similarity search"""
        if not self.is_available:
            return []
        
        try:
            # Use LangChain interface
            filter_dict = kwargs.get('filter')
            results = self.vector_store.similarity_search(
                query=query,
                k=k,
                filter=filter_dict
            )
            
            return results
            
        except Exception as e:
            logger.error(f"ChromaDB similarity search failed: {e}")
            return []
    
    def similarity_search_with_score(self, query: str, k: int = 10, **kwargs) -> List[tuple]:
        """Perform similarity search with scores"""
        if not self.is_available:
            return []
        
        try:
            filter_dict = kwargs.get('filter')
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=k,
                filter=filter_dict
            )
            
            return results
            
        except Exception as e:
            logger.error(f"ChromaDB similarity search with score failed: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get ChromaDB statistics"""
        base_stats = super().get_stats()
        
        if self.is_available:
            try:
                collection = self.client.get_collection(settings.COLLECTION_NAME)
                count = collection.count()
                
                base_stats.update({
                    "total_documents": count,
                    "collection_name": settings.COLLECTION_NAME,
                    "persist_directory": self.persist_directory
                })
            except Exception as e:
                base_stats["error"] = str(e)
        
        return base_stats

class PineconeVectorStoreProvider(BaseVectorStoreProvider):
    """Pinecone vector store provider"""
    
    def _initialize(self):
        if not PINECONE_AVAILABLE:
            logger.warning("Pinecone not available - install pinecone-client")
            return
        
        try:
            if not settings.PINECONE_API_KEY:
                logger.warning("Pinecone API key not configured")
                return
            
            # Initialize Pinecone
            pinecone.init(
                api_key=settings.PINECONE_API_KEY,
                environment=settings.PINECONE_ENV
            )
            
            # Create or get index
            index_name = settings.PINECONE_INDEX_NAME
            
            if index_name not in pinecone.list_indexes():
                # Create index if it doesn't exist
                embedding_config = settings.get_embedding_config()
                pinecone.create_index(
                    name=index_name,
                    dimension=embedding_config["dimensions"],
                    metric="cosine"
                )
                logger.info(f"Created Pinecone index: {index_name}")
            
            # Initialize LangChain Pinecone vector store
            self.vector_store = PineconeVectorStore(
                index_name=index_name,
                embedding=self.embedding_provider,
                namespace=""
            )
            
            self.is_available = True
            logger.info(f"Pinecone initialized with index: {index_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
            self.is_available = False
    
    def add_documents(self, documents: List[Document], **kwargs) -> List[str]:
        """Add documents to Pinecone"""
        if not self.is_available:
            raise ValueError("Pinecone not available")
        
        try:
            ids = kwargs.get('ids')
            if not ids:
                ids = [f"doc_{i}_{hash(doc.page_content) % 10000}" for i, doc in enumerate(documents)]
            
            added_ids = self.vector_store.add_documents(documents, ids=ids)
            
            logger.info(f"Added {len(documents)} documents to Pinecone")
            return added_ids
            
        except Exception as e:
            logger.error(f"Failed to add documents to Pinecone: {e}")
            raise
    
    def similarity_search(self, query: str, k: int = 10, **kwargs) -> List[Document]:
        """Perform similarity search"""
        if not self.is_available:
            return []
        
        try:
            filter_dict = kwargs.get('filter')
            results = self.vector_store.similarity_search(
                query=query,
                k=k,
                filter=filter_dict
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Pinecone similarity search failed: {e}")
            return []
    
    def similarity_search_with_score(self, query: str, k: int = 10, **kwargs) -> List[tuple]:
        """Perform similarity search with scores"""
        if not self.is_available:
            return []
        
        try:
            filter_dict = kwargs.get('filter')
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=k,
                filter=filter_dict
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Pinecone similarity search with score failed: {e}")
            return []

class FAISSVectorStoreProvider(BaseVectorStoreProvider):
    """FAISS vector store provider"""
    
    def _initialize(self):
        try:
            self.index_path = settings.FAISS_INDEX_PATH
            self.index_dir = os.path.dirname(self.index_path)
            os.makedirs(self.index_dir, exist_ok=True)
            
            # Try to load existing index
            if os.path.exists(f"{self.index_path}.faiss"):
                self.vector_store = FAISS.load_local(
                    self.index_path,
                    self.embedding_provider
                )
                logger.info(f"Loaded existing FAISS index from {self.index_path}")
            else:
                # Create empty vector store
                self.vector_store = None
                logger.info("FAISS will be initialized when first documents are added")
            
            self.is_available = True
            
        except Exception as e:
            logger.error(f"Failed to initialize FAISS: {e}")
            self.is_available = False
    
    def add_documents(self, documents: List[Document], **kwargs) -> List[str]:
        """Add documents to FAISS"""
        if not self.is_available:
            raise ValueError("FAISS not available")
        
        try:
            if self.vector_store is None:
                # Create new vector store with first documents
                self.vector_store = FAISS.from_documents(
                    documents,
                    self.embedding_provider
                )
            else:
                # Add to existing vector store
                self.vector_store.add_documents(documents)
            
            # Save the index
            self.vector_store.save_local(self.index_path)
            
            logger.info(f"Added {len(documents)} documents to FAISS")
            return [f"faiss_{i}" for i in range(len(documents))]
            
        except Exception as e:
            logger.error(f"Failed to add documents to FAISS: {e}")
            raise
    
    def similarity_search(self, query: str, k: int = 10, **kwargs) -> List[Document]:
        """Perform similarity search"""
        if not self.is_available or self.vector_store is None:
            return []
        
        try:
            results = self.vector_store.similarity_search(
                query=query,
                k=k
            )
            
            return results
            
        except Exception as e:
            logger.error(f"FAISS similarity search failed: {e}")
            return []
    
    def similarity_search_with_score(self, query: str, k: int = 10, **kwargs) -> List[tuple]:
        """Perform similarity search with scores"""
        if not self.is_available or self.vector_store is None:
            return []
        
        try:
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=k
            )
            
            return results
            
        except Exception as e:
            logger.error(f"FAISS similarity search with score failed: {e}")
            return []

class ModernVectorStore:
    """
    Modern vector store manager with multi-provider support
    """
    
    def __init__(self):
        self.embedding_manager = ModernEmbeddingManager()
        self.providers = {}
        self.current_provider = settings.VECTOR_STORE_TYPE
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize all available vector store providers"""
        
        # Get embedding provider
        embedding_provider = self.embedding_manager.get_langchain_embeddings()
        
        if not embedding_provider:
            logger.error("No embedding provider available")
            return
        
        # Initialize ChromaDB (always available)
        try:
            chroma_provider = ChromaVectorStoreProvider(embedding_provider)
            self.providers["chroma"] = chroma_provider
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB provider: {e}")
        
        # Initialize Pinecone (if available)
        if PINECONE_AVAILABLE and settings.PINECONE_API_KEY:
            try:
                pinecone_provider = PineconeVectorStoreProvider(embedding_provider)
                self.providers["pinecone"] = pinecone_provider
            except Exception as e:
                logger.error(f"Failed to initialize Pinecone provider: {e}")
        
        # Initialize FAISS
        try:
            faiss_provider = FAISSVectorStoreProvider(embedding_provider)
            self.providers["faiss"] = faiss_provider
        except Exception as e:
            logger.error(f"Failed to initialize FAISS provider: {e}")
        
        logger.info(f"Initialized vector store providers: {list(self.providers.keys())}")
    
    def get_provider(self, provider_name: str = None) -> Optional[BaseVectorStoreProvider]:
        """Get vector store provider"""
        provider_name = provider_name or self.current_provider
        
        if provider_name in self.providers:
            provider = self.providers[provider_name]
            if provider.is_available:
                return provider
            else:
                logger.warning(f"Provider {provider_name} not available")
        
        # Fallback to any available provider
        for name, provider in self.providers.items():
            if provider.is_available:
                logger.warning(f"Falling back to {name} vector store")
                return provider
        
        logger.error("No vector store providers available")
        return None
    
    def add_documents(self, 
                     documents: List[Document], 
                     provider: str = None,
                     **kwargs) -> List[str]:
        """Add documents to vector store"""
        vector_provider = self.get_provider(provider)
        if not vector_provider:
            raise ValueError("No vector store provider available")
        
        return vector_provider.add_documents(documents, **kwargs)
    
    def search(self, 
              query: str, 
              n_results: int = 10,
              filters: Dict[str, Any] = None,
              provider: str = None) -> List[Dict[str, Any]]:
        """Search vector store and return structured results"""
        vector_provider = self.get_provider(provider)
        if not vector_provider:
            return []
        
        try:
            # Perform similarity search with scores
            results = vector_provider.similarity_search_with_score(
                query=query,
                k=n_results,
                filter=filters
            )
            
            # Format results
            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "similarity": float(1 - score) if score <= 1 else float(score),  # Normalize score
                    "distance": float(score)
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    def similarity_search_with_score(self, 
                                   query: str,
                                   k: int = 10,
                                   filter: Dict[str, Any] = None,
                                   provider: str = None) -> List[tuple]:
        """LangChain-compatible similarity search with score"""
        vector_provider = self.get_provider(provider)
        if not vector_provider:
            return []
        
        return vector_provider.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter
        )
    
    def get_langchain_vectorstore(self, provider: str = None) -> Optional[VectorStore]:
        """Get LangChain-compatible vector store"""
        vector_provider = self.get_provider(provider)
        if vector_provider:
            return vector_provider.get_langchain_vectorstore()
        return None
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        stats = {
            "current_provider": self.current_provider,
            "available_providers": [
                name for name, provider in self.providers.items() 
                if provider.is_available
            ],
            "embedding_provider": self.embedding_manager.get_provider_info(),
            "provider_details": {}
        }
        
        for name, provider in self.providers.items():
            stats["provider_details"][name] = provider.get_stats()
        
        # Get current provider stats
        current_provider = self.get_provider()
        if current_provider:
            current_stats = current_provider.get_stats()
            stats.update(current_stats)
        
        return stats
    
    def switch_provider(self, provider_name: str) -> bool:
        """Switch to a different vector store provider"""
        if provider_name in self.providers and self.providers[provider_name].is_available:
            self.current_provider = provider_name
            logger.info(f"Switched to {provider_name} vector store")
            return True
        else:
            logger.error(f"Cannot switch to {provider_name} - not available")
            return False
    
    def backup_data(self, backup_path: str, provider: str = None) -> bool:
        """Backup vector store data"""
        vector_provider = self.get_provider(provider)
        if not vector_provider:
            return False
        
        try:
            # For ChromaDB, backup the entire directory
            if isinstance(vector_provider, ChromaVectorStoreProvider):
                import shutil
                shutil.copytree(
                    vector_provider.persist_directory,
                    backup_path,
                    dirs_exist_ok=True
                )
                logger.info(f"ChromaDB backed up to {backup_path}")
                return True
            
            # For FAISS, copy the index files
            elif isinstance(vector_provider, FAISSVectorStoreProvider):
                import shutil
                if os.path.exists(f"{vector_provider.index_path}.faiss"):
                    shutil.copy(f"{vector_provider.index_path}.faiss", f"{backup_path}.faiss")
                    shutil.copy(f"{vector_provider.index_path}.pkl", f"{backup_path}.pkl")
                    logger.info(f"FAISS index backed up to {backup_path}")
                    return True
            
            logger.warning(f"Backup not implemented for {type(vector_provider).__name__}")
            return False
            
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            return False

# Global vector store instance
try:
    vector_store = ModernVectorStore()
    logger.info("Modern vector store initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize vector store: {e}")
    vector_store = None