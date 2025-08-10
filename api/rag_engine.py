import google.generativeai as genai
from typing import List, Dict, Any, Optional
from api.vector_store import vector_store
from config.settings import settings
from config.prompts import prompt_templates
from utils.context_enhancer import context_enhancer
import json
import time

class RAGEngine:
    def __init__(self):
        genai.configure(api_key=settings.GOOGLE_API_KEY)
        self.model = genai.GenerativeModel('gemini-1.5-pro')
        self.vector_store = vector_store
        self.similarity_threshold = settings.SIMILARITY_THRESHOLD
        self.max_chunks = settings.MAX_RETRIEVED_CHUNKS
        self.context_window = settings.CONTEXT_WINDOW
    
    def query(self, 
             question: str, 
             conversation_history: List[Dict[str, str]] = None,
             user_context: Dict[str, Any] = None) -> Dict[str, Any]:
        
        try:
            # Step 1: Enhance query with context
            enhanced_query = self._enhance_query(question, conversation_history, user_context)
            
            # Step 2: Classify query and extract entities
            query_classification = self._classify_query(question)
            
            # Step 3: Retrieve relevant documents
            retrieved_docs = self._retrieve_documents(enhanced_query, query_classification)
            
            # Step 4: Generate response
            response = self._generate_response(
                question, 
                retrieved_docs, 
                conversation_history,
                query_classification
            )
            
            # Step 5: Generate follow-up questions
            follow_ups = self._generate_follow_up_questions(question, response)
            
            return {
                "answer": response,
                "sources": self._format_sources(retrieved_docs),
                "follow_up_questions": follow_ups,
                "query_classification": query_classification,
                "confidence": self._calculate_confidence(retrieved_docs),
                "enhanced_query": enhanced_query
            }
            
        except Exception as e:
            return {
                "answer": f"I apologize, but I encountered an error processing your question: {str(e)}",
                "sources": [],
                "follow_up_questions": [],
                "query_classification": {},
                "confidence": 0.0,
                "enhanced_query": question
            }
    
    def _enhance_query(self, 
                      query: str, 
                      history: List[Dict[str, str]] = None,
                      user_context: Dict[str, Any] = None) -> str:
        
        # Use context enhancer to improve query
        enhancements = context_enhancer.create_search_enhancements(query, user_context)
        
        # Add conversation context if available
        if history:
            recent_context = " ".join([
                f"{msg.get('role', '')}: {msg.get('content', '')}" 
                for msg in history[-3:] if msg.get('content')
            ])
            
            context_prompt = prompt_templates.CONTEXT_ENHANCEMENT_PROMPT.format(
                query=query,
                history=recent_context
            )
            
            try:
                response = self.model.generate_content(context_prompt)
                return response.text.strip()
            except Exception:
                pass
        
        return enhancements.get('expanded_query', query)
    
    def _classify_query(self, query: str) -> Dict[str, Any]:
        classification_prompt = prompt_templates.QUERY_CLASSIFICATION_PROMPT.format(query=query)
        
        try:
            response = self.model.generate_content(classification_prompt)
            result = response.text.strip()
            return json.loads(result)
            
        except Exception as e:
            # Fallback classification
            return self._fallback_classification(query)
    
    def _fallback_classification(self, query: str) -> Dict[str, Any]:
        query_lower = query.lower()
        
        categories = []
        confidence_scores = {}
        
        # Simple keyword-based classification
        if any(word in query_lower for word in ['funding', 'investment', 'vc', 'capital', 'grant']):
            categories.append('FUNDING')
            confidence_scores['FUNDING'] = 0.8
        
        if any(word in query_lower for word in ['regulation', 'legal', 'compliance', 'registration']):
            categories.append('REGULATORY')
            confidence_scores['REGULATORY'] = 0.8
        
        if any(word in query_lower for word in ['market', 'industry', 'competition', 'trends']):
            categories.append('MARKET')
            confidence_scores['MARKET'] = 0.8
        
        if any(word in query_lower for word in ['success', 'case study', 'company', 'founder']):
            categories.append('SUCCESS_STORIES')
            confidence_scores['SUCCESS_STORIES'] = 0.8
        
        if not categories:
            categories = ['GENERAL']
            confidence_scores['GENERAL'] = 0.5
        
        return {
            "categories": categories,
            "confidence_scores": confidence_scores
        }
    
    def _retrieve_documents(self, query: str, classification: Dict[str, Any]) -> List[Dict[str, Any]]:
        # Create filters based on classification
        filters = {}
        
        categories = classification.get('categories', [])
        if categories and 'GENERAL' not in categories:
            # Add category-based filtering if needed
            pass
        
        # Perform vector search
        results = self.vector_store.search(
            query=query,
            n_results=self.max_chunks,
            filters=filters
        )
        
        # Filter by similarity threshold
        filtered_results = [
            doc for doc in results 
            if doc.get('similarity', 0) >= self.similarity_threshold
        ]
        
        return filtered_results[:self.max_chunks]
    
    def _generate_response(self, 
                          question: str, 
                          documents: List[Dict[str, Any]], 
                          history: List[Dict[str, str]] = None,
                          classification: Dict[str, Any] = None) -> str:
        
        # Prepare context from retrieved documents
        context_parts = []
        for i, doc in enumerate(documents):
            metadata = doc.get('metadata', {})
            source_info = ""
            
            if metadata.get('source'):
                source_info = f"Source: {metadata['source']}"
            if metadata.get('country'):
                source_info += f" | Country: {metadata['country']}"
            if metadata.get('sector'):
                source_info += f" | Sector: {metadata['sector']}"
            
            context_parts.append(f"[{i+1}] {source_info}\n{doc['content']}\n")
        
        context = "\n".join(context_parts)
        
        # Determine primary category for context-aware prompting
        primary_category = None
        if classification and 'categories' in classification:
            categories = classification['categories']
            if categories:
                primary_category = categories[0].lower()
        
        # Format the prompt
        rag_prompt = prompt_templates.format_rag_prompt(
            context=context,
            question=question,
            category=primary_category
        )
        
        # Prepare messages
        messages = [
            {"role": "system", "content": prompt_templates.SYSTEM_PROMPT}
        ]
        
        # Add conversation history if available
        if history:
            for msg in history[-5:]:  # Last 5 exchanges
                if msg.get('role') and msg.get('content'):
                    messages.append({
                        "role": msg['role'],
                        "content": msg['content']
                    })
        
        messages.append({"role": "user", "content": rag_prompt})
        
        # Generate response
        try:
            # For Gemini, we'll use a simplified approach since it doesn't support chat history like OpenAI
            full_prompt = f"{prompt_templates.SYSTEM_PROMPT}\n\n{rag_prompt}"
            response = self.model.generate_content(full_prompt)
            
            return response.text.strip()
            
        except Exception as e:
            return f"I apologize, but I'm having trouble generating a response right now. Please try again. Error: {str(e)}"
    
    def _generate_follow_up_questions(self, question: str, answer: str) -> List[str]:
        try:
            prompt = prompt_templates.FOLLOW_UP_QUESTIONS_PROMPT.format(
                question=question,
                answer=answer[:1000]  # Truncate for token limits
            )
            
            response = self.model.generate_content(prompt)
            result = response.text.strip()
            
            # Parse the response into a list
            follow_ups = []
            for line in result.split('\n'):
                line = line.strip()
                if line and (line.startswith('-') or line.startswith('•') or line[0].isdigit()):
                    # Clean up the line
                    cleaned = line.lstrip('-•0123456789. ').strip()
                    if cleaned and '?' in cleaned:
                        follow_ups.append(cleaned)
            
            return follow_ups[:5]  # Return max 5 follow-ups
            
        except Exception:
            # Fallback follow-up questions
            return [
                "Can you tell me more about this topic?",
                "What are the next steps I should consider?",
                "Are there any related opportunities or challenges?"
            ]
    
    def _format_sources(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        sources = []
        
        for doc in documents:
            metadata = doc.get('metadata', {})
            
            source = {
                "content_preview": doc['content'][:200] + "..." if len(doc['content']) > 200 else doc['content'],
                "similarity": round(doc.get('similarity', 0), 3),
                "metadata": {}
            }
            
            # Add relevant metadata
            for key in ['source', 'country', 'sector', 'date', 'title']:
                if metadata.get(key):
                    source['metadata'][key] = metadata[key]
            
            sources.append(source)
        
        return sources
    
    def _calculate_confidence(self, documents: List[Dict[str, Any]]) -> float:
        if not documents:
            return 0.0
        
        # Calculate confidence based on similarity scores and number of relevant documents
        similarities = [doc.get('similarity', 0) for doc in documents]
        avg_similarity = sum(similarities) / len(similarities)
        
        # Factor in the number of high-quality results
        high_quality_count = sum(1 for sim in similarities if sim > 0.8)
        quality_factor = min(high_quality_count / 3, 1.0)  # Normalize to max 1.0
        
        confidence = (avg_similarity * 0.7) + (quality_factor * 0.3)
        return round(confidence, 3)

rag_engine = RAGEngine()