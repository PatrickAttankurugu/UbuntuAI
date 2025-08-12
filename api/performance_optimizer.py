"""
Performance Optimization System for UbuntuAI RAG
Implements industry-standard caching, batching, and optimization techniques
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import json
import time
from functools import wraps
import threading
from collections import OrderedDict

# Caching and optimization libraries (optional)
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    from cachetools import TTLCache, LRUCache
    CACHETOOLS_AVAILABLE = True
except ImportError:
    CACHETOOLS_AVAILABLE = False

# NumPy for calculations (optional)
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

from config.settings import settings

logger = logging.getLogger(__name__)

@dataclass
class CacheMetrics:
    """Cache performance metrics"""
    hits: int = 0
    misses: int = 0
    total_requests: int = 0
    cache_size: int = 0
    hit_rate: float = 0.0
    avg_response_time: float = 0.0

class AdvancedCacheManager:
    """Advanced caching system with multiple backends and strategies"""
    
    def __init__(self):
        self.caches = {}
        self.metrics = {}
        self._initialize_caches()
    
    def _initialize_caches(self):
        """Initialize different cache backends"""
        
        # In-memory TTL cache for fast access
        if CACHETOOLS_AVAILABLE:
            try:
                self.caches["memory_ttl"] = TTLCache(
                    maxsize=1000,
                    ttl=getattr(settings, 'CACHE_TTL', 3600)
                )
                self.metrics["memory_ttl"] = CacheMetrics()
                logger.info("Memory TTL cache initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize memory TTL cache: {e}")
        
        # In-memory LRU cache for frequently accessed items
        if CACHETOOLS_AVAILABLE:
            try:
                self.caches["memory_lru"] = LRUCache(maxsize=500)
                self.metrics["memory_lru"] = CacheMetrics()
                logger.info("Memory LRU cache initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize memory LRU cache: {e}")
        
        # Redis cache for distributed systems
        if REDIS_AVAILABLE and hasattr(settings, 'REDIS_URL'):
            try:
                self.caches["redis"] = redis.Redis.from_url(settings.REDIS_URL)
                self.metrics["redis"] = CacheMetrics()
                logger.info("Redis cache initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Redis cache: {e}")
        
        # Semantic cache for similar queries
        if getattr(settings, 'USE_SEMANTIC_CACHE', False):
            self.caches["semantic"] = self._initialize_semantic_cache()
            self.metrics["semantic"] = CacheMetrics()
            logger.info("Semantic cache initialized")
        
        # Fallback simple cache if no advanced caches available
        if not self.caches:
            self.caches["simple"] = {}
            self.metrics["simple"] = CacheMetrics()
            logger.info("Simple fallback cache initialized")
    
    def _initialize_semantic_cache(self):
        """Initialize semantic cache for similar queries"""
        return {
            "embeddings": {},
            "similarity_threshold": 0.85,
            "max_cache_size": 1000
        }
    
    def get(self, key: str, cache_type: str = "auto") -> Optional[Any]:
        """Get value from cache"""
        
        if cache_type == "auto":
            cache_type = self._select_optimal_cache(key)
        
        if cache_type not in self.caches:
            return None
        
        start_time = time.time()
        
        try:
            if cache_type == "redis":
                value = self.caches[cache_type].get(key)
                if value:
                    value = json.loads(value)
            else:
                value = self.caches[cache_type].get(key)
            
            # Update metrics
            if value is not None:
                self._update_metrics(cache_type, "hit", time.time() - start_time)
                return value
            else:
                self._update_metrics(cache_type, "miss", time.time() - start_time)
                return None
                
        except Exception as e:
            logger.error(f"Cache get error for {cache_type}: {e}")
            return None
    
    def set(self, key: str, value: Any, cache_type: str = "auto", ttl: int = None):
        """Set value in cache"""
        
        if cache_type == "auto":
            cache_type = self._select_optimal_cache(key)
        
        if cache_type not in self.caches:
            return False
        
        try:
            if cache_type == "redis":
                serialized_value = json.dumps(value)
                if ttl:
                    self.caches[cache_type].setex(key, ttl, serialized_value)
                else:
                    self.caches[cache_type].set(key, serialized_value)
            else:
                self.caches[cache_type][key] = value
            
            # Update cache size metric
            if hasattr(self.caches[cache_type], '__len__'):
                self.metrics[cache_type].cache_size = len(self.caches[cache_type])
            
            return True
            
        except Exception as e:
            logger.error(f"Cache set error for {cache_type}: {e}")
            return False
    
    def _select_optimal_cache(self, key: str) -> str:
        """Select optimal cache based on key characteristics"""
        
        # Use Redis for large objects or distributed access
        if len(key) > 100:
            return "redis" if "redis" in self.caches else "memory_ttl"
        
        # Use semantic cache for query-like keys
        if "query" in key.lower() or "question" in key.lower():
            return "semantic" if "semantic" in self.caches else "memory_ttl"
        
        # Default to TTL cache or simple cache
        if "memory_ttl" in self.caches:
            return "memory_ttl"
        elif "simple" in self.caches:
            return "simple"
        else:
            return list(self.caches.keys())[0] if self.caches else "simple"
    
    def _update_metrics(self, cache_type: str, operation: str, response_time: float):
        """Update cache performance metrics"""
        
        if cache_type not in self.metrics:
            return
        
        metrics = self.metrics[cache_type]
        metrics.total_requests += 1
        
        if operation == "hit":
            metrics.hits += 1
        else:
            metrics.misses += 1
        
        # Update hit rate
        metrics.hit_rate = metrics.hits / metrics.total_requests if metrics.total_requests > 0 else 0
        
        # Update average response time
        if metrics.hits > 0:
            metrics.avg_response_time = (
                (metrics.avg_response_time * (metrics.hits - 1) + response_time) / metrics.hits
            )
    
    def get_metrics(self, cache_type: str = None) -> Dict[str, Any]:
        """Get cache performance metrics"""
        
        if cache_type:
            if cache_type in self.metrics:
                return {
                    cache_type: self.metrics[cache_type].__dict__
                }
            return {}
        
        return {
            cache_type: metrics.__dict__ 
            for cache_type, metrics in self.metrics.items()
        }
    
    def clear_cache(self, cache_type: str = None):
        """Clear cache"""
        
        if cache_type:
            if cache_type in self.caches:
                if hasattr(self.caches[cache_type], 'clear'):
                    self.caches[cache_type].clear()
                else:
                    self.caches[cache_type] = {}
                self.metrics[cache_type] = CacheMetrics()
        else:
            for cache_type in list(self.caches.keys()):
                if hasattr(self.caches[cache_type], 'clear'):
                    self.caches[cache_type].clear()
                else:
                    self.caches[cache_type] = {}
                self.metrics[cache_type] = CacheMetrics()

class BatchProcessor:
    """Batch processing for improved performance"""
    
    def __init__(self, batch_size: int = 32, max_wait_time: float = 1.0):
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.pending_items = []
        self.processing_lock = threading.Lock()
        self.processing_thread = None
        self.stop_event = threading.Event()
    
    def add_item(self, item: Any, callback: Callable):
        """Add item to batch processing queue"""
        
        with self.processing_lock:
            self.pending_items.append((item, callback))
            
            # Start processing if batch is full or processing thread is not running
            if (len(self.pending_items) >= self.batch_size or 
                (self.processing_thread is None or not self.processing_thread.is_alive())):
                self._start_processing()
    
    def _start_processing(self):
        """Start batch processing thread"""
        
        if self.processing_thread and self.processing_thread.is_alive():
            return
        
        self.processing_thread = threading.Thread(target=self._process_batch)
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def _process_batch(self):
        """Process items in batches"""
        
        while not self.stop_event.is_set():
            with self.processing_lock:
                if len(self.pending_items) == 0:
                    break
                
                # Take up to batch_size items
                batch = self.pending_items[:self.batch_size]
                self.pending_items = self.pending_items[self.batch_size:]
            
            if batch:
                try:
                    # Process batch
                    items, callbacks = zip(*batch)
                    results = self._process_batch_items(items)
                    
                    # Call callbacks with results
                    for callback, result in zip(callbacks, results):
                        try:
                            callback(result)
                        except Exception as e:
                            logger.error(f"Callback error: {e}")
                
                except Exception as e:
                    logger.error(f"Batch processing error: {e}")
            
            # Wait for more items or timeout
            time.sleep(self.max_wait_time)
    
    def _process_batch_items(self, items: List[Any]) -> List[Any]:
        """Process a batch of items - override in subclasses"""
        return items
    
    def stop(self):
        """Stop batch processing"""
        self.stop_event.set()
        if self.processing_thread:
            self.processing_thread.join()

class QueryOptimizer:
    """Query optimization for improved retrieval performance"""
    
    def __init__(self):
        self.query_cache = {}
        self.optimization_rules = self._load_optimization_rules()
    
    def _load_optimization_rules(self) -> Dict[str, Any]:
        """Load query optimization rules"""
        
        # Get Ghana regions from settings if available
        ghana_regions = getattr(settings, 'GHANA_REGIONS', [
            "Greater Accra", "Ashanti", "Western", "Central", "Eastern"
        ])
        
        return {
            "ghana_keywords": [
                "ghana", "ghanian", "accra", "kumasi", "tamale", "tema",
                "bank of ghana", "gipc", "ghana enterprise agency", "mofa"
            ],
            "sector_keywords": {
                "fintech": ["fintech", "financial", "banking", "mobile money"],
                "agritech": ["agritech", "agriculture", "farming", "crop"],
                "healthtech": ["healthtech", "healthcare", "medical"]
            },
            "boosting_rules": {
                "exact_match": 2.0,
                "partial_match": 1.5,
                "sector_match": 1.3,
                "regional_match": 1.2
            },
            "ghana_regions": ghana_regions
        }
    
    def optimize_query(self, query: str, user_context: Dict[str, Any] = None) -> str:
        """Optimize query for better retrieval"""
        
        # Check cache first
        cache_key = self._create_cache_key(query, user_context)
        if cache_key in self.query_cache:
            return self.query_cache[cache_key]
        
        optimized_query = query
        
        # Apply optimization rules
        optimized_query = self._apply_ghana_boosting(optimized_query)
        optimized_query = self._apply_sector_boosting(optimized_query, user_context)
        optimized_query = self._apply_regional_boosting(optimized_query, user_context)
        optimized_query = self._apply_synonym_expansion(optimized_query)
        
        # Cache optimized query
        self.query_cache[cache_key] = optimized_query
        
        return optimized_query
    
    def _create_cache_key(self, query: str, user_context: Dict[str, Any]) -> str:
        """Create cache key for query optimization"""
        
        context_str = json.dumps(user_context, sort_keys=True) if user_context else ""
        return hashlib.md5(f"{query}:{context_str}".encode()).hexdigest()
    
    def _apply_ghana_boosting(self, query: str) -> str:
        """Apply Ghana-specific boosting"""
        
        # Add Ghana keywords if not present
        ghana_keywords = self.optimization_rules["ghana_keywords"]
        query_lower = query.lower()
        
        missing_keywords = []
        for keyword in ghana_keywords[:3]:  # Use top 3 keywords
            if keyword not in query_lower:
                missing_keywords.append(keyword)
        
        if missing_keywords:
            query += f" {' '.join(missing_keywords)}"
        
        return query
    
    def _apply_sector_boosting(self, query: str, user_context: Dict[str, Any]) -> str:
        """Apply sector-specific boosting"""
        
        if not user_context or 'sector' not in user_context:
            return query
        
        user_sector = user_context['sector'].lower()
        sector_keywords = self.optimization_rules["sector_keywords"].get(user_sector, [])
        
        if sector_keywords:
            # Add sector keywords if not present
            query_lower = query.lower()
            missing_sector_keywords = [
                keyword for keyword in sector_keywords 
                if keyword not in query_lower
            ]
            
            if missing_sector_keywords:
                query += f" {' '.join(missing_sector_keywords[:2])}"
        
        return query
    
    def _apply_regional_boosting(self, query: str, user_context: Dict[str, Any]) -> str:
        """Apply regional boosting"""
        
        if not user_context or 'region' not in user_context:
            return query
        
        user_region = user_context['region'].lower()
        if user_region not in query.lower():
            query += f" {user_region}"
        
        return query
    
    def _apply_synonym_expansion(self, query: str) -> str:
        """Apply synonym expansion"""
        
        # Simple synonym mapping for Ghanaian business terms
        synonyms = {
            "startup": ["startup", "business", "company", "enterprise"],
            "funding": ["funding", "investment", "capital", "money", "finance"],
            "regulation": ["regulation", "compliance", "legal", "law", "requirement"],
            "ghana": ["ghana", "ghanian", "accra", "kumasi"]
        }
        
        expanded_terms = []
        for term, synonym_list in synonyms.items():
            if term in query.lower():
                # Add 1-2 relevant synonyms
                relevant_synonyms = [
                    syn for syn in synonym_list 
                    if syn not in query.lower()
                ][:2]
                expanded_terms.extend(relevant_synonyms)
        
        if expanded_terms:
            query += f" {' '.join(expanded_terms)}"
        
        return query

class PerformanceProfiler:
    """Performance profiling and monitoring"""
    
    def __init__(self):
        self.profiles = {}
        self.active_profiles = {}
    
    def start_profile(self, profile_name: str) -> str:
        """Start a performance profile"""
        
        profile_id = f"{profile_name}_{int(time.time() * 1000)}"
        self.active_profiles[profile_id] = {
            "name": profile_name,
            "start_time": time.time(),
            "checkpoints": [],
            "metadata": {}
        }
        
        return profile_id
    
    def add_checkpoint(self, profile_id: str, checkpoint_name: str, metadata: Dict[str, Any] = None):
        """Add a checkpoint to a profile"""
        
        if profile_id not in self.active_profiles:
            return
        
        profile = self.active_profiles[profile_id]
        checkpoint = {
            "name": checkpoint_name,
            "timestamp": time.time(),
            "elapsed": time.time() - profile["start_time"],
            "metadata": metadata or {}
        }
        
        profile["checkpoints"].append(checkpoint)
    
    def end_profile(self, profile_id: str) -> Dict[str, Any]:
        """End a performance profile and return results"""
        
        if profile_id not in self.active_profiles:
            return {}
        
        profile = self.active_profiles[profile_id]
        end_time = time.time()
        
        # Calculate final metrics
        total_time = end_time - profile["start_time"]
        
        # Calculate time between checkpoints
        checkpoint_times = []
        for i, checkpoint in enumerate(profile["checkpoints"]):
            if i > 0:
                prev_checkpoint = profile["checkpoints"][i-1]
                checkpoint_times.append({
                    "from": prev_checkpoint["name"],
                    "to": checkpoint["name"],
                    "duration": checkpoint["timestamp"] - prev_checkpoint["timestamp"]
                })
        
        # Create final profile
        final_profile = {
            "profile_id": profile_id,
            "name": profile["name"],
            "total_time": total_time,
            "checkpoints": profile["checkpoints"],
            "checkpoint_times": checkpoint_times,
            "metadata": profile["metadata"]
        }
        
        # Store in profiles history
        self.profiles[profile_id] = final_profile
        
        # Remove from active profiles
        del self.active_profiles[profile_id]
        
        return final_profile
    
    def get_profile(self, profile_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific profile"""
        return self.profiles.get(profile_id)
    
    def get_profiles_summary(self) -> Dict[str, Any]:
        """Get summary of all profiles"""
        
        if not self.profiles:
            return {"total_profiles": 0}
        
        total_profiles = len(self.profiles)
        total_time = sum(profile["total_time"] for profile in self.profiles.values())
        avg_time = total_time / total_profiles
        
        # Group by profile name
        profile_groups = {}
        for profile in self.profiles.values():
            name = profile["name"]
            if name not in profile_groups:
                profile_groups[name] = []
            profile_groups[name].append(profile["total_time"])
        
        # Calculate statistics for each group
        group_stats = {}
        for name, times in profile_groups.items():
            if NUMPY_AVAILABLE:
                group_stats[name] = {
                    "count": len(times),
                    "avg_time": np.mean(times),
                    "min_time": np.min(times),
                    "max_time": np.max(times),
                    "std_time": np.std(times)
                }
            else:
                # Fallback without numpy
                group_stats[name] = {
                    "count": len(times),
                    "avg_time": sum(times) / len(times),
                    "min_time": min(times),
                    "max_time": max(times),
                    "std_time": 0.0  # Can't calculate without numpy
                }
        
        return {
            "total_profiles": total_profiles,
            "total_time": total_time,
            "avg_time": avg_time,
            "profile_groups": group_stats
        }

# Performance optimization decorators
def cache_result(cache_type: str = "auto", ttl: int = None):
    """Decorator to cache function results"""
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            cache_key = f"{func.__name__}:{hash(str(args) + str(sorted(kwargs.items()))}"
            
            # Try to get from cache
            cache_manager = AdvancedCacheManager()
            cached_result = cache_manager.get(cache_key, cache_type)
            
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache_manager.set(cache_key, result, cache_type, ttl)
            
            return result
        
        return wrapper
    return decorator

def profile_performance(profile_name: str = None):
    """Decorator to profile function performance"""
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create profile name
            name = profile_name or func.__name__
            
            # Start profiling
            profiler = PerformanceProfiler()
            profile_id = profiler.start_profile(name)
            
            try:
                # Execute function
                start_time = time.time()
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                # Add execution checkpoint
                profiler.add_checkpoint(profile_id, "execution", {
                    "execution_time": execution_time,
                    "args_count": len(args),
                    "kwargs_count": len(kwargs)
                })
                
                return result
                
            finally:
                # End profiling
                profiler.end_profile(profile_id)
        
        return wrapper
    return decorator

def batch_process(batch_size: int = 32, max_wait_time: float = 1.0):
    """Decorator to enable batch processing"""
    
    def decorator(func):
        # Create batch processor for this function
        processor = BatchProcessor(batch_size, max_wait_time)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # For now, just call the function directly
            # In a real implementation, you'd add items to the batch processor
            return func(*args, **kwargs)
        
        # Store processor for potential future use
        wrapper.batch_processor = processor
        return wrapper
    
    return decorator

# Global instances
cache_manager = AdvancedCacheManager()
query_optimizer = QueryOptimizer()
performance_profiler = PerformanceProfiler()

# Convenience functions
def optimize_query(query: str, user_context: Dict[str, Any] = None) -> str:
    """Optimize query for better performance"""
    return query_optimizer.optimize_query(query, user_context)

def start_performance_profile(name: str) -> str:
    """Start a performance profile"""
    return performance_profiler.start_profile(name)

def add_profile_checkpoint(profile_id: str, checkpoint_name: str, metadata: Dict[str, Any] = None):
    """Add a checkpoint to a performance profile"""
    performance_profiler.add_checkpoint(profile_id, checkpoint_name, metadata)

def end_performance_profile(profile_id: str) -> Dict[str, Any]:
    """End a performance profile"""
    return performance_profiler.end_profile(profile_id) 