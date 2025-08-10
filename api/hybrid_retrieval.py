import openai
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import json
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import re
from collections import defaultdict
from api.vector_store import vector_store
from utils.embeddings import embedding_service
from config.settings import settings

@dataclass
class RetrievalResult:
    content: str
    metadata: Dict[str, Any]
    relevance_score: float
    retrieval_method: str
    context_match_score: float
    business_relevance_score: float
    recency_score: float
    authority_score: float
    combined_score: float

@dataclass
class SearchQuery:
    text: str
    context: Dict[str, Any]
    filters: Dict[str, Any]
    user_profile: Dict[str, Any]
    search_intent: str
    priority_weights: Dict[str, float]

class HybridRetriever:
    """
    Advanced hybrid retrieval system combining multiple retrieval methods
    Optimized for business context awareness and African market focus
    Implements semantic search, keyword matching, and business rules
    """
    
    def __init__(self):
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        
        # Initialize retrieval components
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 3),
            max_df=0.8,
            min_df=2
        )
        
        self.scaler = StandardScaler()
        
        # Business context weights
        self.business_context_weights = {
            "sector_match": 0.25,
            "location_match": 0.20,
            "stage_match": 0.15,
            "funding_match": 0.15,
            "size_match": 0.10,
            "model_match": 0.15
        }
        
        # Retrieval method weights (adaptive)
        self.method_weights = {
            "semantic": 0.4,
            "keyword": 0.3,
            "business_rules": 0.2,
            "collaborative": 0.1
        }
        
        # African market context boosters
        self.african_context_boosters = {
            "mobile_first": 1.2,
            "rural_focus": 1.15,
            "women_led": 1.1,
            "agriculture": 1.25,
            "fintech": 1.2,
            "local_language": 1.1
        }
        
        # Initialize components
        self._initialize_components()

    def search(self, 
              query: Union[str, SearchQuery],
              context: Dict[str, Any] = None,
              limit: int = 10,
              filters: Dict[str, Any] = None) -> List[RetrievalResult]:
        """
        Main hybrid search function combining multiple retrieval methods
        """
        
        # Normalize query input
        if isinstance(query, str):
            search_query = SearchQuery(
                text=query,
                context=context or {},
                filters=filters or {},
                user_profile={},
                search_intent=self._detect_search_intent(query),
                priority_weights=self._calculate_priority_weights(query, context)
            )
        else:
            search_query = query
        
        # Multi-method retrieval
        retrieval_results = {}
        
        # 1. Semantic retrieval
        semantic_results = self._semantic_retrieval(search_query, limit * 2)
        retrieval_results["semantic"] = semantic_results
        
        # 2. Keyword-based retrieval
        keyword_results = self._keyword_retrieval(search_query, limit * 2)
        retrieval_results["keyword"] = keyword_results
        
        # 3. Business rule-based retrieval
        business_results = self._business_rule_retrieval(search_query, limit * 2)
        retrieval_results["business_rules"] = business_results
        
        # 4. Collaborative filtering (based on similar users/contexts)
        collaborative_results = self._collaborative_retrieval(search_query, limit)
        retrieval_results["collaborative"] = collaborative_results
        
        # 5. Hybrid fusion and re-ranking
        fused_results = self._fuse_and_rerank(retrieval_results, search_query)
        
        # 6. Apply business context boosting
        boosted_results = self._apply_business_boosting(fused_results, search_query)
        
        # 7. Diversify results
        diversified_results = self._diversify_results(boosted_results, search_query)
        
        return diversified_results[:limit]

    def retrieve_similar_business_models(self, 
                                       context: Dict[str, Any],
                                       limit: int = 5) -> List[Dict[str, Any]]:
        """Retrieve similar business models for comparison and learning"""
        
        # Create search query for business models
        search_query = SearchQuery(
            text="business model framework strategy",
            context=context,
            filters={"type": "business_model", "category": "strategy"},
            user_profile={},
            search_intent="business_model_search",
            priority_weights={"business_relevance": 0.6, "recency": 0.2, "authority": 0.2}
        )
        
        # Enhanced business model search
        business_model_results = self._specialized_business_model_search(search_query)
        
        # Format for business model copilot
        formatted_results = []
        for result in business_model_results[:limit]:
            formatted_result = {
                "description": result.content,
                "metadata": result.metadata,
                "relevance": result.business_relevance_score,
                "success_indicators": self._extract_success_indicators(result),
                "implementation_notes": self._extract_implementation_notes(result)
            }
            formatted_results.append(formatted_result)
        
        return formatted_results

    def _semantic_retrieval(self, 
                           search_query: SearchQuery,
                           limit: int) -> List[RetrievalResult]:
        """Semantic retrieval using embeddings and vector similarity"""
        
        try:
            # Enhanced query with context
            enhanced_query = self._enhance_query_with_context(search_query)
            
            # Vector search
            vector_results = self.vector_store.search(
                query=enhanced_query,
                n_results=limit,
                filters=self._convert_filters(search_query.filters)
            )
            
            results = []
            for result in vector_results:
                retrieval_result = RetrievalResult(
                    content=result["content"],
                    metadata=result.get("metadata", {}),
                    relevance_score=result.get("similarity", 0.0),
                    retrieval_method="semantic",
                    context_match_score=self._calculate_context_match(result, search_query),
                    business_relevance_score=self._calculate_business_relevance(result, search_query),
                    recency_score=self._calculate_recency_score(result),
                    authority_score=self._calculate_authority_score(result),
                    combined_score=0.0  # Will be calculated in fusion
                )
                results.append(retrieval_result)
            
            return results
            
        except Exception as e:
            print(f"Semantic retrieval error: {e}")
            return []

    def _keyword_retrieval(self, 
                          search_query: SearchQuery,
                          limit: int) -> List[RetrievalResult]:
        """Keyword-based retrieval using TF-IDF and exact matching"""
        
        try:
            # Extract keywords from query
            keywords = self._extract_keywords(search_query.text)
            
            # Add business context keywords
            business_keywords = self._extract_business_keywords(search_query.context)
            all_keywords = keywords + business_keywords
            
            # Search using multiple keyword strategies
            exact_matches = self._exact_keyword_search(all_keywords, limit // 2)
            fuzzy_matches = self._fuzzy_keyword_search(all_keywords, limit // 2)
            
            # Combine and score keyword results
            keyword_results = exact_matches + fuzzy_matches
            
            # Convert to RetrievalResult format
            results = []
            for result in keyword_results:
                retrieval_result = RetrievalResult(
                    content=result["content"],
                    metadata=result.get("metadata", {}),
                    relevance_score=result.get("keyword_score", 0.0),
                    retrieval_method="keyword",
                    context_match_score=self._calculate_context_match(result, search_query),
                    business_relevance_score=self._calculate_business_relevance(result, search_query),
                    recency_score=self._calculate_recency_score(result),
                    authority_score=self._calculate_authority_score(result),
                    combined_score=0.0
                )
                results.append(retrieval_result)
            
            return results
            
        except Exception as e:
            print(f"Keyword retrieval error: {e}")
            return []

    def _business_rule_retrieval(self, 
                               search_query: SearchQuery,
                               limit: int) -> List[RetrievalResult]:
        """Business rule-based retrieval using domain knowledge"""
        
        try:
            # Apply business rules based on query intent and context
            rule_filters = self._apply_business_rules(search_query)
            
            # Search with rule-based filters
            rule_results = self.vector_store.search(
                query=search_query.text,
                n_results=limit,
                filters=rule_filters
            )
            
            # Score based on business rule matching
            results = []
            for result in rule_results:
                rule_match_score = self._calculate_rule_match_score(result, search_query)
                
                retrieval_result = RetrievalResult(
                    content=result["content"],
                    metadata=result.get("metadata", {}),
                    relevance_score=rule_match_score,
                    retrieval_method="business_rules",
                    context_match_score=self._calculate_context_match(result, search_query),
                    business_relevance_score=self._calculate_business_relevance(result, search_query),
                    recency_score=self._calculate_recency_score(result),
                    authority_score=self._calculate_authority_score(result),
                    combined_score=0.0
                )
                results.append(retrieval_result)
            
            return results
            
        except Exception as e:
            print(f"Business rule retrieval error: {e}")
            return []

    def _collaborative_retrieval(self, 
                               search_query: SearchQuery,
                               limit: int) -> List[RetrievalResult]:
        """Collaborative filtering based on similar users/contexts"""
        
        try:
            # Find similar user profiles or contexts
            similar_contexts = self._find_similar_contexts(search_query)
            
            # Retrieve content that worked well for similar contexts
            collaborative_results = []
            
            for similar_context in similar_contexts[:3]:  # Top 3 similar contexts
                context_results = self._get_successful_content_for_context(similar_context)
                collaborative_results.extend(context_results)
            
            # Remove duplicates and score
            unique_results = self._deduplicate_results(collaborative_results)
            
            # Convert to RetrievalResult format
            results = []
            for result in unique_results[:limit]:
                retrieval_result = RetrievalResult(
                    content=result["content"],
                    metadata=result.get("metadata", {}),
                    relevance_score=result.get("collaborative_score", 0.0),
                    retrieval_method="collaborative",
                    context_match_score=self._calculate_context_match(result, search_query),
                    business_relevance_score=self._calculate_business_relevance(result, search_query),
                    recency_score=self._calculate_recency_score(result),
                    authority_score=self._calculate_authority_score(result),
                    combined_score=0.0
                )
                results.append(retrieval_result)
            
            return results
            
        except Exception as e:
            print(f"Collaborative retrieval error: {e}")
            return []

    def _fuse_and_rerank(self, 
                        retrieval_results: Dict[str, List[RetrievalResult]],
                        search_query: SearchQuery) -> List[RetrievalResult]:
        """Fuse results from multiple retrieval methods and re-rank"""
        
        # Combine all results
        all_results = []
        for method, results in retrieval_results.items():
            for result in results:
                all_results.append(result)
        
        # Remove duplicates based on content similarity
        unique_results = self._deduplicate_retrieval_results(all_results)
        
        # Calculate combined scores
        for result in unique_results:
            combined_score = self._calculate_combined_score(result, search_query)
            result.combined_score = combined_score
        
        # Re-rank by combined score
        unique_results.sort(key=lambda x: x.combined_score, reverse=True)
        
        return unique_results

    def _calculate_combined_score(self, 
                                result: RetrievalResult,
                                search_query: SearchQuery) -> float:
        """Calculate combined relevance score using multiple factors"""
        
        # Base relevance score
        base_score = result.relevance_score
        
        # Context matching boost
        context_boost = result.context_match_score * 0.3
        
        # Business relevance boost
        business_boost = result.business_relevance_score * 0.4
        
        # Recency boost (for time-sensitive queries)
        recency_boost = result.recency_score * 0.1
        
        # Authority boost
        authority_boost = result.authority_score * 0.2
        
        # Method-specific weighting
        method_weight = self.method_weights.get(result.retrieval_method, 1.0)
        
        # Query-specific priority weighting
        priority_weights = search_query.priority_weights
        
        combined_score = (
            base_score * method_weight +
            context_boost * priority_weights.get("context", 1.0) +
            business_boost * priority_weights.get("business_relevance", 1.0) +
            recency_boost * priority_weights.get("recency", 1.0) +
            authority_boost * priority_weights.get("authority", 1.0)
        )
        
        return min(combined_score, 1.0)  # Cap at 1.0

    def _apply_business_boosting(self, 
                               results: List[RetrievalResult],
                               search_query: SearchQuery) -> List[RetrievalResult]:
        """Apply African market and business-specific boosting"""
        
        for result in results:
            boost_factor = 1.0
            
            # African context boosting
            for context_key, boost_value in self.african_context_boosters.items():
                if self._has_african_context(result, context_key):
                    boost_factor *= boost_value
            
            # User context matching
            user_context = search_query.context
            if self._matches_user_context(result, user_context):
                boost_factor *= 1.15
            
            # Apply boost to combined score
            result.combined_score = min(result.combined_score * boost_factor, 1.0)
        
        # Re-sort after boosting
        results.sort(key=lambda x: x.combined_score, reverse=True)
        
        return results

    def _diversify_results(self, 
                         results: List[RetrievalResult],
                         search_query: SearchQuery) -> List[RetrievalResult]:
        """Diversify results to avoid redundancy"""
        
        if len(results) <= 5:
            return results
        
        diversified = []
        used_categories = set()
        content_embeddings = []
        
        for result in results:
            # Category diversification
            category = result.metadata.get("category", "general")
            
            # Content similarity diversification
            if diversified:
                content_similarity = self._calculate_content_similarity(
                    result.content, [r.content for r in diversified]
                )
                if content_similarity > 0.8:  # Too similar
                    continue
            
            # Add if it adds diversity
            if category not in used_categories or len(diversified) < 3:
                diversified.append(result)
                used_categories.add(category)
                
                if len(diversified) >= len(results) * 0.8:  # Keep top 80%
                    break
        
        return diversified

    def _enhance_query_with_context(self, search_query: SearchQuery) -> str:
        """Enhance query with business context for better retrieval"""
        
        enhanced_parts = [search_query.text]
        
        context = search_query.context
        
        # Add business context
        if context.get("sector"):
            enhanced_parts.append(f"sector:{context['sector']}")
        
        if context.get("location"):
            enhanced_parts.append(f"location:{context['location']}")
        
        if context.get("business_stage"):
            enhanced_parts.append(f"stage:{context['business_stage']}")
        
        if context.get("funding_stage"):
            enhanced_parts.append(f"funding:{context['funding_stage']}")
        
        # Add African market context
        enhanced_parts.append("African business ecosystem")
        
        return " ".join(enhanced_parts)

    def _detect_search_intent(self, query: str) -> str:
        """Detect search intent from query text"""
        
        query_lower = query.lower()
        
        # Assessment intent
        if any(word in query_lower for word in ['assess', 'evaluate', 'score', 'rate']):
            return "assessment"
        
        # Funding intent
        if any(word in query_lower for word in ['fund', 'investment', 'capital', 'money']):
            return "funding"
        
        # Guidance intent
        if any(word in query_lower for word in ['how', 'guide', 'help', 'advice']):
            return "guidance"
        
        # Research intent
        if any(word in query_lower for word in ['research', 'analyze', 'study', 'market']):
            return "research"
        
        # Comparison intent
        if any(word in query_lower for word in ['compare', 'versus', 'vs', 'difference']):
            return "comparison"
        
        return "general"

    def _calculate_priority_weights(self, 
                                  query: str,
                                  context: Dict[str, Any] = None) -> Dict[str, float]:
        """Calculate priority weights based on query and context"""
        
        intent = self._detect_search_intent(query)
        
        # Default weights
        weights = {
            "business_relevance": 0.4,
            "context": 0.3,
            "recency": 0.15,
            "authority": 0.15
        }
        
        # Adjust based on intent
        if intent == "funding":
            weights["recency"] = 0.3  # Recent funding info is important
            weights["authority"] = 0.25
        elif intent == "assessment":
            weights["business_relevance"] = 0.5
            weights["context"] = 0.35
        elif intent == "research":
            weights["authority"] = 0.3
            weights["recency"] = 0.25
        
        return weights

    def _calculate_context_match(self, 
                               result: Dict[str, Any],
                               search_query: SearchQuery) -> float:
        """Calculate how well result matches search context"""
        
        match_score = 0.0
        total_weight = 0.0
        
        result_metadata = result.get("metadata", {})
        query_context = search_query.context
        
        # Sector matching
        if query_context.get("sector") and result_metadata.get("sector"):
            if query_context["sector"].lower() == result_metadata["sector"].lower():
                match_score += self.business_context_weights["sector_match"]
            total_weight += self.business_context_weights["sector_match"]
        
        # Location matching
        if query_context.get("location") and result_metadata.get("country"):
            if query_context["location"].lower() in result_metadata["country"].lower():
                match_score += self.business_context_weights["location_match"]
            total_weight += self.business_context_weights["location_match"]
        
        # Stage matching
        if query_context.get("business_stage") and result_metadata.get("stage"):
            if query_context["business_stage"].lower() in result_metadata["stage"].lower():
                match_score += self.business_context_weights["stage_match"]
            total_weight += self.business_context_weights["stage_match"]
        
        return match_score / total_weight if total_weight > 0 else 0.0

    def _calculate_business_relevance(self, 
                                    result: Dict[str, Any],
                                    search_query: SearchQuery) -> float:
        """Calculate business relevance score"""
        
        relevance_score = 0.0
        
        content = result.get("content", "").lower()
        metadata = result.get("metadata", {})
        
        # African business keywords
        african_keywords = ['africa', 'african', 'ghana', 'nigeria', 'kenya', 'startup', 'entrepreneur']
        african_matches = sum(1 for keyword in african_keywords if keyword in content)
        relevance_score += min(african_matches / 3, 1.0) * 0.3
        
        # Business model keywords
        business_keywords = ['business model', 'revenue', 'funding', 'growth', 'market', 'customer']
        business_matches = sum(1 for keyword in business_keywords if keyword in content)
        relevance_score += min(business_matches / 3, 1.0) * 0.4
        
        # Sector relevance
        if metadata.get("sector") in search_query.context.get("sector", ""):
            relevance_score += 0.3
        
        return min(relevance_score, 1.0)

    def _calculate_recency_score(self, result: Dict[str, Any]) -> float:
        """Calculate recency score based on content date"""
        
        metadata = result.get("metadata", {})
        date_str = metadata.get("date") or metadata.get("created_at")
        
        if not date_str:
            return 0.5  # Default middle score
        
        try:
            if isinstance(date_str, str):
                # Try to parse different date formats
                try:
                    content_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                except:
                    content_date = datetime.strptime(date_str[:10], '%Y-%m-%d')
            else:
                content_date = date_str
            
            # Calculate age in days
            age_days = (datetime.now() - content_date).days
            
            # Score based on age (fresher content scores higher)
            if age_days <= 30:
                return 1.0
            elif age_days <= 90:
                return 0.8
            elif age_days <= 365:
                return 0.6
            elif age_days <= 730:
                return 0.4
            else:
                return 0.2
                
        except Exception:
            return 0.5

    def _calculate_authority_score(self, result: Dict[str, Any]) -> float:
        """Calculate authority score based on source credibility"""
        
        metadata = result.get("metadata", {})
        source = metadata.get("source", "").lower()
        
        # Authority scoring based on source type
        if any(term in source for term in ['government', 'official', 'ministry']):
            return 1.0
        elif any(term in source for term in ['world bank', 'afdb', 'african development']):
            return 0.95
        elif any(term in source for term in ['university', 'research', 'academic']):
            return 0.9
        elif any(term in source for term in ['seedstars', 'vc', 'investment']):
            return 0.85
        elif any(term in source for term in ['startup', 'entrepreneur', 'business']):
            return 0.7
        else:
            return 0.5

    def _initialize_components(self):
        """Initialize retrieval components"""
        
        try:
            # Initialize TF-IDF if we have documents
            # In production, this would be pre-computed
            pass
            
        except Exception as e:
            print(f"Component initialization error: {e}")

# Factory function
def create_hybrid_retriever():
    return HybridRetriever()