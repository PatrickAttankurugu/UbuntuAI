import google.generativeai as genai
from typing import List, Dict, Any, Optional
from api.vector_store import vector_store
from config.settings import settings
from config.prompts import prompt_templates
from utils.context_enhancer import context_enhancer
import json
import time
import logging

logger = logging.getLogger(__name__)

class RAGEngine:
    def __init__(self):
        if not settings.GOOGLE_API_KEY:
            raise ValueError("Google API key not configured for RAG engine")
            
        genai.configure(api_key=settings.GOOGLE_API_KEY)
        
        # Initialize Gemini model for text generation with fixed configuration
        self.model = genai.GenerativeModel(
            model_name=settings.LLM_MODEL,
            generation_config=settings.get_gemini_config()  # This now excludes the model field
        )
        
        self.vector_store = vector_store
        self.similarity_threshold = settings.SIMILARITY_THRESHOLD
        self.max_chunks = settings.MAX_RETRIEVED_CHUNKS
        self.context_window = settings.CONTEXT_WINDOW
        
        logger.info(f"RAG Engine initialized with model: {settings.LLM_MODEL}")
    
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
                query_classification,
                user_context
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
            logger.error(f"RAG Engine error: {e}")
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
                # Create a separate model instance for context enhancement
                enhancement_model = genai.GenerativeModel(
                    model_name="gemini-1.5-flash",  # Use faster model for enhancement
                    generation_config={"temperature": 0.3, "max_output_tokens": 200}
                )
                response = enhancement_model.generate_content(context_prompt)
                if response and response.text:
                    return response.text.strip()
            except Exception as e:
                logger.warning(f"Context enhancement failed: {e}")
        
        return enhancements.get('expanded_query', query)
    
    def _classify_query(self, query: str) -> Dict[str, Any]:
        classification_prompt = prompt_templates.QUERY_CLASSIFICATION_PROMPT.format(query=query)
        
        try:
            # Create a separate model instance for classification
            classification_model = genai.GenerativeModel(
                model_name="gemini-1.5-flash",
                generation_config={"temperature": 0.1, "max_output_tokens": 300}
            )
            response = classification_model.generate_content(classification_prompt)
            if response and response.text:
                # Try to parse JSON response
                result = response.text.strip()
                # Remove markdown code blocks if present
                if result.startswith('```'):
                    result = result.split('\n', 1)[1].rsplit('\n', 1)[0]
                return json.loads(result)
            else:
                return self._fallback_classification(query)
                
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Query classification failed: {e}")
            return self._fallback_classification(query)
    
    def _fallback_classification(self, query: str) -> Dict[str, Any]:
        query_lower = query.lower()
        
        categories = []
        confidence_scores = {}
        
        # Simple keyword-based classification
        classification_keywords = {
            'FUNDING': ['funding', 'investment', 'vc', 'capital', 'grant', 'investor', 'money'],
            'REGULATORY': ['regulation', 'legal', 'compliance', 'registration', 'license', 'permit'],
            'MARKET': ['market', 'industry', 'competition', 'trends', 'analysis', 'research'],
            'SUCCESS_STORIES': ['success', 'case study', 'company', 'founder', 'unicorn'],
            'SECTOR': ['fintech', 'agritech', 'healthtech', 'edtech', 'technology'],
            'COUNTRY': ['nigeria', 'kenya', 'ghana', 'south africa', 'egypt', 'morocco']
        }
        
        for category, keywords in classification_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in query_lower)
            if matches > 0:
                categories.append(category)
                confidence_scores[category] = min(matches / len(keywords) * 2, 1.0)
        
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
            if 'COUNTRY' in categories:
                # Extract country names from query for filtering
                for country in settings.AFRICAN_COUNTRIES:
                    if country.lower() in query.lower():
                        filters['country'] = country
                        break
        
        # Perform vector search
        try:
            results = self.vector_store.search(
                query=query,
                n_results=self.max_chunks,
                filters=filters if filters else None
            )
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
        
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
                          classification: Dict[str, Any] = None,
                          user_context: Dict[str, Any] = None) -> str:
        
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
        
        # Add user context information
        user_info = ""
        if user_context:
            user_details = []
            if user_context.get('country'):
                user_details.append(f"User location: {user_context['country']}")
            if user_context.get('sector'):
                user_details.append(f"Business sector: {user_context['sector']}")
            if user_context.get('business_stage'):
                user_details.append(f"Business stage: {user_context['business_stage']}")
            
            if user_details:
                user_info = f"\nUser Context: {' | '.join(user_details)}"
        
        # Format the prompt
        rag_prompt = prompt_templates.format_rag_prompt(
            context=context + user_info,
            question=question,
            category=primary_category
        )
        
        # Build conversation context
        conversation_context = ""
        if history:
            recent_messages = []
            for msg in history[-3:]:  # Last 3 exchanges
                if msg.get('role') and msg.get('content'):
                    role = "Human" if msg['role'] == 'user' else "Assistant"
                    recent_messages.append(f"{role}: {msg['content'][:200]}")
            
            if recent_messages:
                conversation_context = f"\nRecent conversation:\n" + "\n".join(recent_messages) + "\n"
        
        # Complete prompt with system instructions and conversation context
        full_prompt = f"""{prompt_templates.SYSTEM_PROMPT}

{conversation_context}

{rag_prompt}"""
        
        # Generate response using Gemini
        try:
            response = self.model.generate_content(full_prompt)
            
            if response and response.text:
                return response.text.strip()
            else:
                return "I apologize, but I'm having trouble generating a response right now. Please try rephrasing your question."
                
        except Exception as e:
            logger.error(f"Gemini generation error: {e}")
            return f"I encountered an error while generating a response. Please try again. Error: {str(e)}"
    
    def _generate_follow_up_questions(self, question: str, answer: str) -> List[str]:
        try:
            prompt = prompt_templates.FOLLOW_UP_QUESTIONS_PROMPT.format(
                question=question,
                answer=answer[:1000]  # Truncate for token limits
            )
            
            # Create a separate model instance for follow-up generation
            followup_model = genai.GenerativeModel(
                model_name="gemini-1.5-flash",
                generation_config={"temperature": 0.5, "max_output_tokens": 300}
            )
            response = followup_model.generate_content(prompt)
            
            if not response or not response.text:
                return self._get_default_follow_ups()
            
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
            
            return follow_ups[:5] if follow_ups else self._get_default_follow_ups()
            
        except Exception as e:
            logger.warning(f"Follow-up generation failed: {e}")
            return self._get_default_follow_ups()
    
    def _get_default_follow_ups(self) -> List[str]:
        """Get default follow-up questions"""
        return [
            "Can you tell me more about this topic?",
            "What are the next steps I should consider?",
            "Are there any related opportunities or challenges?",
            "How does this apply to my specific situation?",
            "What resources can help me learn more?"
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
            for key in ['source', 'country', 'sector', 'date', 'title', 'type']:
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
        
        # Factor in total number of results
        quantity_factor = min(len(documents) / 5, 1.0)  # Normalize to max 1.0
        
        confidence = (avg_similarity * 0.5) + (quality_factor * 0.3) + (quantity_factor * 0.2)
        return round(confidence, 3)

# Global RAG engine instance
try:
    rag_engine = RAGEngine()
    logger.info("RAG Engine initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize RAG Engine: {e}")
    rag_engine = None