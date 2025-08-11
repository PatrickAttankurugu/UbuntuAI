"""
Advanced Retrieval System for UbuntuAI
Implements hybrid search, re-ranking, and multi-stage retrieval
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from abc import ABC, abstractmethod
import numpy as np
from dataclasses import dataclass
import asyncio

# LangChain imports
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_community.retrievers import PineconeHybridSearchRetriever

# Re-ranking
try:
    from flashrank import Ranker, RerankRequest
    FLASHRANK_AVAILABLE = True
except ImportError:
    try:
        from sentence_transformers import CrossEncoder
        CROSSENCODER_AVAILABLE = True
        FLASHRANK_AVAILABLE = False
    except ImportError:
        CROSSENCODER_AVAILABLE = False
        FLASHRANK_AVAILABLE = False

# BM25 for keyword search
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("BM25 not available - install rank-bm25 for keyword search")

from config.settings import settings

logger = logging.getLogger(__name__)

@dataclass
class RetrievalResult:
    """Structured retrieval result with metadata"""
    content: str
    metadata: Dict[str, Any]
    score: float
    retrieval_method: str
    rerank_score: Optional[float] = None

class BaseRetriever(ABC):
    """Abstract base class for retrievers"""
    
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """Retrieve relevant documents"""
        pass
    
    @abstractmethod
    def add_documents(self, documents: List[Document]):
        """Add documents to the retriever"""
        pass

class SemanticRetriever(BaseRetriever):
    """Semantic retrieval using vector similarity"""
    
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.retrieval_method = "semantic"
    
    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """Retrieve using semantic similarity"""
        try:
            results = self.vector_store.search(
                query=query,
                n_results=top_k,
                filters=None
            )
            
            retrieval_results = []
            for result in results:
                retrieval_results.append(RetrievalResult(
                    content=result["content"],
                    metadata=result.get("metadata", {}),
                    score=result.get("similarity", 0.0),
                    retrieval_method=self.retrieval_method
                ))
            
            return retrieval_results
            
        except Exception as e:
            logger.error(f"Semantic retrieval failed: {e}")
            return []
    
    def add_documents(self, documents: List[Document]):
        """Add documents to vector store"""
        docs = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        self.vector_store.add_documents(docs, metadatas)

class KeywordRetriever(BaseRetriever):
    """Keyword-based retrieval using BM25"""
    
    def __init__(self):
        self.bm25 = None
        self.documents = []
        self.retrieval_method = "keyword"
        self.is_available = BM25_AVAILABLE
    
    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """Retrieve using BM25 keyword matching"""
        if not self.is_available or not self.bm25:
            return []
        
        try:
            query_tokens = query.lower().split()
            scores = self.bm25.get_scores(query_tokens)
            
            # Get top-k results
            top_indices = np.argsort(scores)[::-1][:top_k]
            
            retrieval_results = []
            for idx in top_indices:
                if scores[idx] > 0:  # Only include relevant results
                    doc = self.documents[idx]
                    retrieval_results.append(RetrievalResult(
                        content=doc.page_content,
                        metadata=doc.metadata,
                        score=float(scores[idx]),
                        retrieval_method=self.retrieval_method
                    ))
            
            return retrieval_results
            
        except Exception as e:
            logger.error(f"Keyword retrieval failed: {e}")
            return []
    
    def add_documents(self, documents: List[Document]):
        """Add documents and rebuild BM25 index"""
        if not self.is_available:
            return
        
        self.documents.extend(documents)
        
        # Tokenize documents for BM25
        tokenized_docs = []
        for doc in self.documents:
            tokens = doc.page_content.lower().split()
            tokenized_docs.append(tokens)
        
        # Build BM25 index
        self.bm25 = BM25Okapi(tokenized_docs)
        logger.info(f"BM25 index updated with {len(self.documents)} documents")

class HybridRetriever(BaseRetriever):
    """Hybrid retrieval combining semantic and keyword search"""
    
    def __init__(self, vector_store):
        self.semantic_retriever = SemanticRetriever(vector_store)
        self.keyword_retriever = KeywordRetriever()
        self.semantic_weight = settings.SEMANTIC_WEIGHT
        self.keyword_weight = settings.KEYWORD_WEIGHT
        self.retrieval_method = "hybrid"
    
    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """Retrieve using hybrid approach"""
        
        # Get results from both retrievers
        retrieval_size = min(top_k * 2, 50)  # Retrieve more for fusion
        
        semantic_results = self.semantic_retriever.retrieve(query, retrieval_size)
        keyword_results = self.keyword_retriever.retrieve(query, retrieval_size)
        
        # Combine and score fusion
        fused_results = self._fusion_scoring(
            semantic_results, 
            keyword_results, 
            query
        )
        
        # Return top-k
        return fused_results[:top_k]
    
    def _fusion_scoring(self, 
                       semantic_results: List[RetrievalResult],
                       keyword_results: List[RetrievalResult],
                       query: str) -> List[RetrievalResult]:
        """Perform score fusion of semantic and keyword results"""
        
        # Create content-based mapping
        result_map = {}
        
        # Add semantic results
        for result in semantic_results:
            content_hash = hash(result.content)
            if content_hash not in result_map:
                result_map[content_hash] = result
                result.score = result.score * self.semantic_weight
            else:
                # Update with higher semantic score
                existing = result_map[content_hash]
                if result.score > existing.score / self.semantic_weight:
                    existing.score = result.score * self.semantic_weight
        
        # Add keyword results
        for result in keyword_results:
            content_hash = hash(result.content)
            if content_hash in result_map:
                # Combine scores
                result_map[content_hash].score += result.score * self.keyword_weight
                result_map[content_hash].retrieval_method = "hybrid"
            else:
                # New result from keyword search
                result.score = result.score * self.keyword_weight
                result.retrieval_method = "keyword_only"
                result_map[content_hash] = result
        
        # Sort by combined score
        combined_results = list(result_map.values())
        combined_results.sort(key=lambda x: x.score, reverse=True)
        
        return combined_results
    
    def add_documents(self, documents: List[Document]):
        """Add documents to both retrievers"""
        self.semantic_retriever.add_documents(documents)
        self.keyword_retriever.add_documents(documents)

class RerankerComponent:
    """Re-ranking component for improving retrieval quality"""
    
    def __init__(self):
        self.ranker = None
        self.method = "none"
        self._initialize_ranker()
    
    def _initialize_ranker(self):
        """Initialize the best available re-ranker"""
        
        if FLASHRANK_AVAILABLE:
            try:
                self.ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")
                self.method = "flashrank"
                logger.info("FlashRank re-ranker initialized")
                return
            except Exception as e:
                logger.warning(f"FlashRank initialization failed: {e}")
        
        if CROSSENCODER_AVAILABLE:
            try:
                self.ranker = CrossEncoder(settings.RERANKER_MODEL)
                self.method = "crossencoder"
                logger.info(f"CrossEncoder re-ranker initialized: {settings.RERANKER_MODEL}")
                return
            except Exception as e:
                logger.warning(f"CrossEncoder initialization failed: {e}")
        
        logger.warning("No re-ranking model available")
    
    def rerank(self, 
              query: str, 
              results: List[RetrievalResult],
              top_k: int = None) -> List[RetrievalResult]:
        """Re-rank retrieval results"""
        
        if not self.ranker or not results:
            return results
        
        top_k = top_k or settings.RERANKED_TOP_K
        
        try:
            if self.method == "flashrank":
                return self._rerank_flashrank(query, results, top_k)
            elif self.method == "crossencoder":
                return self._rerank_crossencoder(query, results, top_k)
            else:
                return results[:top_k]
                
        except Exception as e:
            logger.error(f"Re-ranking failed: {e}")
            return results[:top_k]
    
    def _rerank_flashrank(self, 
                         query: str, 
                         results: List[RetrievalResult],
                         top_k: int) -> List[RetrievalResult]:
        """Re-rank using FlashRank"""
        
        # Prepare passages for FlashRank
        passages = []
        for i, result in enumerate(results):
            passages.append({
                "id": i,
                "text": result.content,
                "meta": result.metadata
            })
        
        # Create rerank request
        rerank_request = RerankRequest(query=query, passages=passages)
        
        # Get reranked results
        reranked = self.ranker.rerank(rerank_request)
        
        # Map back to RetrievalResult objects
        reranked_results = []
        for item in reranked[:top_k]:
            original_idx = item["id"]
            original_result = results[original_idx]
            original_result.rerank_score = item["score"]
            reranked_results.append(original_result)
        
        return reranked_results
    
    def _rerank_crossencoder(self, 
                           query: str, 
                           results: List[RetrievalResult],
                           top_k: int) -> List[RetrievalResult]:
        """Re-rank using CrossEncoder"""
        
        # Prepare query-document pairs
        pairs = [(query, result.content) for result in results]
        
        # Get rerank scores
        scores = self.ranker.predict(pairs)
        
        # Combine scores with results
        scored_results = []
        for result, score in zip(results, scores):
            result.rerank_score = float(score)
            scored_results.append(result)
        
        # Sort by rerank score
        scored_results.sort(key=lambda x: x.rerank_score, reverse=True)
        
        return scored_results[:top_k]

class AdvancedRetriever:
    """Advanced retrieval system with multiple strategies"""
    
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.retrievers = {
            "semantic": SemanticRetriever(vector_store),
            "keyword": KeywordRetriever(),
            "hybrid": HybridRetriever(vector_store)
        }
        self.reranker = RerankerComponent()
        self.current_strategy = settings.RETRIEVAL_STRATEGY
        
        logger.info(f"Advanced retriever initialized with strategy: {self.current_strategy}")
    
    def retrieve(self, 
                query: str, 
                strategy: str = None,
                top_k: int = None,
                use_reranking: bool = None) -> List[RetrievalResult]:
        """Retrieve with specified strategy"""
        
        strategy = strategy or self.current_strategy
        top_k = top_k or settings.MAX_RETRIEVED_CHUNKS
        use_reranking = use_reranking if use_reranking is not None else settings.USE_RERANKING
        
        # Get retriever
        retriever = self.retrievers.get(strategy, self.retrievers["semantic"])
        
        # Initial retrieval
        retrieval_size = top_k * 2 if use_reranking else top_k
        results = retriever.retrieve(query, retrieval_size)
        
        # Apply re-ranking if enabled
        if use_reranking and self.reranker.method != "none":
            results = self.reranker.rerank(query, results, top_k)
        else:
            results = results[:top_k]
        
        # Filter by similarity threshold
        filtered_results = [
            result for result in results 
            if result.score >= settings.SIMILARITY_THRESHOLD
        ]
        
        logger.debug(f"Retrieved {len(results)} documents, {len(filtered_results)} after filtering")
        
        return filtered_results
    
    def add_documents(self, documents: List[Document]):
        """Add documents to all retrievers"""
        for retriever in self.retrievers.values():
            try:
                retriever.add_documents(documents)
            except Exception as e:
                logger.error(f"Failed to add documents to retriever: {e}")
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get retrieval system statistics"""
        stats = {
            "current_strategy": self.current_strategy,
            "available_strategies": list(self.retrievers.keys()),
            "reranker_method": self.reranker.method,
            "settings": {
                "similarity_threshold": settings.SIMILARITY_THRESHOLD,
                "max_retrieved_chunks": settings.MAX_RETRIEVED_CHUNKS,
                "semantic_weight": settings.SEMANTIC_WEIGHT,
                "keyword_weight": settings.KEYWORD_WEIGHT,
                "use_reranking": settings.USE_RERANKING
            }
        }
        
        # Add retriever-specific stats
        if hasattr(self.retrievers["keyword"], "bm25") and self.retrievers["keyword"].bm25:
            stats["keyword_retriever_docs"] = len(self.retrievers["keyword"].documents)
        
        return stats
    
    def set_strategy(self, strategy: str):
        """Change retrieval strategy"""
        if strategy in self.retrievers:
            self.current_strategy = strategy
            logger.info(f"Retrieval strategy changed to: {strategy}")
        else:
            logger.warning(f"Unknown retrieval strategy: {strategy}")

class ContextualRetriever:
    """Retriever with contextual awareness and self-reflection"""
    
    def __init__(self, base_retriever: AdvancedRetriever):
        self.base_retriever = base_retriever
        self.context_history = []
        self.max_context_history = 5
    
    def retrieve_with_context(self, 
                            query: str,
                            conversation_history: List[Dict[str, str]] = None,
                            user_context: Dict[str, Any] = None) -> List[RetrievalResult]:
        """Retrieve with conversation and user context"""
        
        # Enhance query with context
        enhanced_query = self._enhance_query_with_context(
            query, conversation_history, user_context
        )
        
        # Perform retrieval
        results = self.base_retriever.retrieve(enhanced_query)
        
        # Apply contextual filtering
        filtered_results = self._apply_contextual_filters(
            results, user_context
        )
        
        # Update context history
        self._update_context_history(query, results)
        
        return filtered_results
    
    def _enhance_query_with_context(self, 
                                  query: str,
                                  conversation_history: List[Dict[str, str]],
                                  user_context: Dict[str, Any]) -> str:
        """Enhance query with contextual information"""
        
        enhanced_parts = [query]
        
        # Add user context
        if user_context:
            context_parts = []
            if user_context.get('country'):
                context_parts.append(f"country:{user_context['country']}")
            if user_context.get('sector'):
                context_parts.append(f"sector:{user_context['sector']}")
            if user_context.get('business_stage'):
                context_parts.append(f"stage:{user_context['business_stage']}")
            
            if context_parts:
                enhanced_parts.append(" ".join(context_parts))
        
        # Add conversation context
        if conversation_history:
            recent_context = []
            for msg in conversation_history[-2:]:  # Last 2 exchanges
                if msg.get('role') == 'user' and msg.get('content'):
                    recent_context.append(msg['content'][:100])
            
            if recent_context:
                enhanced_parts.append(" ".join(recent_context))
        
        return " ".join(enhanced_parts)
    
    def _apply_contextual_filters(self, 
                                results: List[RetrievalResult],
                                user_context: Dict[str, Any]) -> List[RetrievalResult]:
        """Apply contextual filtering to results"""
        
        if not user_context:
            return results
        
        filtered_results = []
        
        for result in results:
            relevance_score = self._calculate_contextual_relevance(
                result, user_context
            )
            
            # Boost score based on contextual relevance
            result.score = result.score * (1 + relevance_score * 0.2)
            
            # Only include results above threshold
            if result.score >= settings.SIMILARITY_THRESHOLD:
                filtered_results.append(result)
        
        # Re-sort by boosted scores
        filtered_results.sort(key=lambda x: x.score, reverse=True)
        
        return filtered_results
    
    def _calculate_contextual_relevance(self, 
                                      result: RetrievalResult,
                                      user_context: Dict[str, Any]) -> float:
        """Calculate contextual relevance score"""
        
        relevance = 0.0
        metadata = result.metadata
        
        # Country relevance
        if user_context.get('country') and metadata.get('country'):
            if user_context['country'].lower() in metadata['country'].lower():
                relevance += 0.3
        
        # Sector relevance
        if user_context.get('sector') and metadata.get('sector'):
            if user_context['sector'].lower() in metadata['sector'].lower():
                relevance += 0.3
        
        # Stage relevance
        if user_context.get('business_stage') and metadata.get('funding_stage'):
            if user_context['business_stage'].lower() in metadata['funding_stage'].lower():
                relevance += 0.2
        
        return min(relevance, 1.0)
    
    def _update_context_history(self, query: str, results: List[RetrievalResult]):
        """Update context history for future retrievals"""
        
        self.context_history.append({
            "query": query,
            "result_count": len(results),
            "avg_score": sum(r.score for r in results) / len(results) if results else 0,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only recent history
        if len(self.context_history) > self.max_context_history:
            self.context_history.pop(0)

# Create global retriever instance (will be initialized with vector store)
advanced_retriever = None
contextual_retriever = None

def initialize_retrieval_system(vector_store):
    """Initialize the global retrieval system"""
    global advanced_retriever, contextual_retriever
    
    try:
        advanced_retriever = AdvancedRetriever(vector_store)
        contextual_retriever = ContextualRetriever(advanced_retriever)
        logger.info("Advanced retrieval system initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize retrieval system: {e}")

def get_retriever() -> Optional[AdvancedRetriever]:
    """Get the global retriever instance"""
    return advanced_retriever

def get_contextual_retriever() -> Optional[ContextualRetriever]:
    """Get the global contextual retriever instance"""
    return contextual_retriever