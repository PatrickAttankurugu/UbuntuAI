import openai
import numpy as np
from typing import List, Dict, Any, Optional
import tiktoken
import time
from config.settings import settings

class EmbeddingService:
    def __init__(self):
        openai.api_key = settings.OPENAI_API_KEY
        self.model = settings.EMBEDDING_MODEL
        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        self.max_tokens = 8191
        
    def count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))
    
    def truncate_text(self, text: str, max_tokens: int = None) -> str:
        if max_tokens is None:
            max_tokens = self.max_tokens
            
        tokens = self.encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
            
        truncated_tokens = tokens[:max_tokens]
        return self.encoding.decode(truncated_tokens)
    
    def create_embedding(self, text: str, retry_count: int = 3) -> Optional[List[float]]:
        text = self.truncate_text(text)
        
        for attempt in range(retry_count):
            try:
                response = openai.Embedding.create(
                    model=self.model,
                    input=text
                )
                return response['data'][0]['embedding']
                
            except openai.error.RateLimitError:
                if attempt < retry_count - 1:
                    time.sleep(2 ** attempt)
                    continue
                else:
                    raise
                    
            except Exception as e:
                if attempt < retry_count - 1:
                    time.sleep(1)
                    continue
                else:
                    print(f"Error creating embedding: {e}")
                    return None
                    
        return None
    
    def create_embeddings_batch(self, texts: List[str], batch_size: int = 100) -> List[Optional[List[float]]]:
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            truncated_batch = [self.truncate_text(text) for text in batch]
            
            try:
                response = openai.Embedding.create(
                    model=self.model,
                    input=truncated_batch
                )
                
                batch_embeddings = [item['embedding'] for item in response['data']]
                embeddings.extend(batch_embeddings)
                
                if i + batch_size < len(texts):
                    time.sleep(0.1)
                    
            except Exception as e:
                print(f"Error in batch embedding: {e}")
                embeddings.extend([None] * len(batch))
                
        return embeddings
    
    def cosine_similarity(self, a: List[float], b: List[float]) -> float:
        a_np = np.array(a)
        b_np = np.array(b)
        
        dot_product = np.dot(a_np, b_np)
        norm_a = np.linalg.norm(a_np)
        norm_b = np.linalg.norm(b_np)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
            
        return dot_product / (norm_a * norm_b)
    
    def semantic_search(self, query_embedding: List[float], 
                       document_embeddings: List[List[float]], 
                       top_k: int = 10) -> List[Dict[str, Any]]:
        similarities = []
        
        for idx, doc_embedding in enumerate(document_embeddings):
            if doc_embedding is not None:
                similarity = self.cosine_similarity(query_embedding, doc_embedding)
                similarities.append({
                    'index': idx,
                    'similarity': similarity
                })
        
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:top_k]
    
    def enhance_query_embedding(self, query: str, context: Dict[str, Any] = None) -> str:
        enhanced_query = query
        
        if context:
            if 'country' in context:
                enhanced_query += f" {context['country']} African business"
            if 'sector' in context:
                enhanced_query += f" {context['sector']} industry"
            if 'stage' in context:
                enhanced_query += f" {context['stage']} startup"
        
        return enhanced_query
    
    def get_embedding_stats(self, embeddings: List[List[float]]) -> Dict[str, Any]:
        valid_embeddings = [e for e in embeddings if e is not None]
        
        if not valid_embeddings:
            return {"count": 0, "valid": 0, "invalid": len(embeddings)}
        
        embeddings_array = np.array(valid_embeddings)
        
        return {
            "count": len(embeddings),
            "valid": len(valid_embeddings),
            "invalid": len(embeddings) - len(valid_embeddings),
            "dimensions": embeddings_array.shape[1] if len(embeddings_array.shape) > 1 else 0,
            "mean_magnitude": np.mean(np.linalg.norm(embeddings_array, axis=1)),
            "std_magnitude": np.std(np.linalg.norm(embeddings_array, axis=1))
        }

class BusinessContextEmbeddings:
    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service
        
    def create_context_aware_embedding(self, text: str, metadata: Dict[str, Any]) -> List[float]:
        enhanced_text = self._enhance_text_with_context(text, metadata)
        return self.embedding_service.create_embedding(enhanced_text)
    
    def _enhance_text_with_context(self, text: str, metadata: Dict[str, Any]) -> str:
        enhancements = []
        
        if 'country' in metadata and metadata['country']:
            enhancements.append(f"Country: {metadata['country']}")
        
        if 'sector' in metadata and metadata['sector']:
            enhancements.append(f"Business Sector: {metadata['sector']}")
        
        if 'funding_stage' in metadata and metadata['funding_stage']:
            enhancements.append(f"Funding Stage: {metadata['funding_stage']}")
        
        if 'source_type' in metadata and metadata['source_type']:
            enhancements.append(f"Source Type: {metadata['source_type']}")
        
        if 'date' in metadata and metadata['date']:
            enhancements.append(f"Date: {metadata['date']}")
        
        if enhancements:
            context_string = " | ".join(enhancements)
            return f"{text}\n\nContext: {context_string}"
        
        return text

embedding_service = EmbeddingService()
business_context_embeddings = BusinessContextEmbeddings(embedding_service)