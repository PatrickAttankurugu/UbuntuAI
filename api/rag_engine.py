#!/usr/bin/env python3
"""
RAG Engine for UbuntuAI
Retrieval-Augmented Generation using Google's Gemini model
"""

import os
import logging
from typing import List, Dict, Any, Optional
import google.generativeai as genai
from datetime import datetime

from config.settings import settings
from api.vector_store import vector_store
from utils.embeddings import embedding_service

logger = logging.getLogger(__name__)

class RAGEngine:
    """Retrieval-Augmented Generation engine"""
    
    def __init__(self, vector_store_instance=None):
        """Initialize the RAG engine"""
        self.vector_store = vector_store_instance or vector_store
        self.model_name = settings.LLM_MODEL
        self.max_retrieval_results = settings.MAX_RETRIEVAL_RESULTS
        
        # Initialize Google Gemini
        if not settings.GOOGLE_API_KEY:
            logger.error("Google API key not configured")
            self._initialized = False
            return
        
        try:
            genai.configure(api_key=settings.GOOGLE_API_KEY)
            self.model = genai.GenerativeModel(self.model_name)
            logger.info(f"RAG engine initialized with model: {self.model_name}")
            self._initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize RAG engine: {e}")
            self._initialized = False
    
    def is_initialized(self) -> bool:
        """Check if the RAG engine is properly initialized"""
        return self._initialized
    
    def query(self, question: str, context: str = None, max_results: int = None) -> Dict[str, Any]:
        """
        Process a query using RAG
        
        Args:
            question: User's question
            context: Additional context (optional)
            max_results: Maximum number of retrieval results
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        if not self._initialized:
            return {
                'answer': 'RAG engine not initialized. Please check your configuration.',
                'sources': [],
                'confidence': 0.0,
                'error': 'Engine not initialized'
            }
        
        if not question or not question.strip():
            return {
                'answer': 'Please provide a valid question.',
                'sources': [],
                'confidence': 0.0,
                'error': 'Empty question'
            }
        
        try:
            # Step 1: Retrieve relevant documents
            retrieved_docs = self._retrieve_documents(question, max_results)
            
            if not retrieved_docs:
                logger.warning(f"No relevant documents found for query: {question}")
                return self._generate_answer_without_context(question)
            
            # Step 2: Generate answer using retrieved context
            answer = self._generate_answer(question, retrieved_docs, context)
            
            # Step 3: Calculate confidence based on retrieval quality
            confidence = self._calculate_confidence(retrieved_docs)
            
            # Step 4: Format response
            response = {
                'answer': answer,
                'sources': self._format_sources(retrieved_docs),
                'confidence': confidence,
                'retrieved_documents': len(retrieved_docs),
                'query': question,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"RAG query processed successfully: {len(retrieved_docs)} documents retrieved")
            return response
            
        except Exception as e:
            logger.error(f"Error processing RAG query: {e}")
            return {
                'answer': f'I encountered an error while processing your question: {str(e)}',
                'sources': [],
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _retrieve_documents(self, query: str, max_results: int = None) -> List[Dict[str, Any]]:
        """Retrieve relevant documents from vector store"""
        if not self.vector_store:
            logger.error("Vector store not available")
            return []
        
        try:
            max_results = max_results or self.max_retrieval_results
            
            # Search for relevant documents
            results = self.vector_store.search(query, max_results)
            
            # Filter out low-quality results
            filtered_results = []
            for result in results:
                # Basic quality check - ensure document has content
                if result.get('document') and len(result['document'].strip()) > 50:
                    filtered_results.append(result)
            
            logger.info(f"Retrieved {len(filtered_results)} relevant documents")
            return filtered_results
            
        except Exception as e:
            logger.error(f"Document retrieval failed: {e}")
            return []
    
    def _generate_answer(self, question: str, retrieved_docs: List[Dict[str, Any]], context: str = None) -> str:
        """Generate answer using retrieved documents and LLM"""
        try:
            # Prepare context from retrieved documents
            context_text = self._prepare_context(retrieved_docs)
            
            # Add additional context if provided
            if context:
                context_text = f"{context}\n\n{context_text}"
            
            # Create prompt for the LLM
            prompt = self._create_prompt(question, context_text)
            
            # Generate response using Gemini
            response = self.model.generate_content(prompt)
            
            if response and response.text:
                answer = response.text.strip()
                logger.info(f"Generated answer of length {len(answer)}")
                return answer
            else:
                logger.warning("LLM returned empty response")
                return "I couldn't generate a response based on the available information."
                
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return f"I encountered an error while generating an answer: {str(e)}"
    
    def _prepare_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """Prepare context text from retrieved documents"""
        if not retrieved_docs:
            return ""
        
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            doc_text = doc.get('document', '')
            metadata = doc.get('metadata', {})
            
            # Add document source information
            source = metadata.get('source', 'Unknown source')
            doc_type = metadata.get('document_type', 'Document')
            
            context_parts.append(f"Document {i} ({doc_type} from {source}):\n{doc_text}\n")
        
        return "\n".join(context_parts)
    
    def _create_prompt(self, question: str, context: str) -> str:
        """Create a prompt for the LLM"""
        prompt = f"""You are UbuntuAI, an AI assistant specialized in African business intelligence, particularly focused on Ghana's startup ecosystem including fintech, agritech, and healthtech sectors.

Context Information:
{context}

Question: {question}

Instructions:
1. Answer the question based on the context information provided above
2. If the context doesn't contain enough information to answer the question, say so clearly
3. Focus on practical, actionable insights relevant to African entrepreneurs
4. Be specific about Ghanaian business opportunities, regulations, and ecosystem
5. Use a professional but approachable tone
6. If relevant, mention specific regions, sectors, or organizations from Ghana

Answer:"""
        
        return prompt
    
    def _generate_answer_without_context(self, question: str) -> str:
        """Generate answer when no relevant documents are found"""
        try:
            prompt = f"""You are UbuntuAI, an AI assistant specialized in African business intelligence, particularly focused on Ghana's startup ecosystem.

Question: {question}

Note: I don't have specific documents to reference for this question, but I can provide general guidance based on my knowledge of African business and Ghana's startup ecosystem.

Please provide a helpful response focusing on general business principles and opportunities in Ghana's fintech, agritech, and healthtech sectors."""

            response = self.model.generate_content(prompt)
            
            if response and response.text:
                return response.text.strip()
            else:
                return "I don't have enough specific information to answer your question accurately. Please try rephrasing or ask about a different topic related to Ghanaian business opportunities."
                
        except Exception as e:
            logger.error(f"Fallback answer generation failed: {e}")
            return "I'm unable to provide a comprehensive answer at the moment. Please try again later."
    
    def _calculate_confidence(self, retrieved_docs: List[Dict[str, Any]]) -> float:
        """Calculate confidence score based on retrieval quality"""
        if not retrieved_docs:
            return 0.0
        
        try:
            # Calculate average similarity score
            total_similarity = 0.0
            valid_scores = 0
            
            for doc in retrieved_docs:
                distance = doc.get('distance', 0.0)
                if distance is not None and distance >= 0:
                    # Convert distance to similarity (lower distance = higher similarity)
                    similarity = max(0.0, 1.0 - distance)
                    total_similarity += similarity
                    valid_scores += 1
            
            if valid_scores > 0:
                avg_similarity = total_similarity / valid_scores
                # Boost confidence if we have multiple relevant documents
                document_boost = min(0.2, len(retrieved_docs) * 0.05)
                confidence = min(1.0, avg_similarity + document_boost)
                return round(confidence, 2)
            else:
                return 0.5  # Default confidence
                
        except Exception as e:
            logger.error(f"Confidence calculation failed: {e}")
            return 0.5
    
    def _format_sources(self, retrieved_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format source information for the response"""
        sources = []
        
        for doc in retrieved_docs:
            metadata = doc.get('metadata', {})
            source_info = {
                'source': metadata.get('source', 'Unknown'),
                'document_type': metadata.get('document_type', 'Document'),
                'added_at': metadata.get('added_at', 'Unknown'),
                'relevance_score': round(1.0 - doc.get('distance', 0.0), 3)
            }
            sources.append(source_info)
        
        return sources
    
    def batch_query(self, questions: List[str]) -> List[Dict[str, Any]]:
        """Process multiple queries in batch"""
        if not questions:
            return []
        
        results = []
        for question in questions:
            result = self.query(question)
            results.append(result)
        
        return results
    
    def get_engine_info(self) -> Dict[str, Any]:
        """Get information about the RAG engine"""
        return {
            'model_name': self.model_name,
            'vector_store_available': self.vector_store is not None,
            'embedding_service_available': embedding_service is not None,
            'max_retrieval_results': self.max_retrieval_results,
            'initialized': self._initialized,
            'timestamp': datetime.now().isoformat()
        }

# Global RAG engine instance
try:
    rag_engine = RAGEngine()
    if not rag_engine.is_initialized():
        logger.warning("RAG engine failed to initialize - some features may not work")
        rag_engine = None
except Exception as e:
    logger.error(f"Failed to create RAG engine: {e}")
    rag_engine = None

# Convenience functions
def initialize_rag_engine(vector_store_instance=None):
    """Initialize and return the RAG engine"""
    global rag_engine
    if rag_engine is None:
        try:
            rag_engine = RAGEngine(vector_store_instance)
            if rag_engine.is_initialized():
                logger.info("RAG engine initialized successfully")
            else:
                logger.warning("RAG engine failed to initialize")
                return None
        except Exception as e:
            logger.error(f"Failed to initialize RAG engine: {e}")
            return None
    return rag_engine

def query_rag(question: str, context: str = None) -> Dict[str, Any]:
    """Query the RAG engine"""
    if rag_engine:
        return rag_engine.query(question, context)
    else:
        logger.error("RAG engine not available")
        return {
            'answer': 'RAG engine not available. Please check your configuration.',
            'sources': [],
            'confidence': 0.0,
            'error': 'Engine not available'
        }