import openai
from typing import List, Dict, Any, Optional
from api.vector_store import vector_store
from config.settings import settings
from config.prompts import prompt_templates
from utils.context_enhancer import context_enhancer
import json
import time

class RAGEngine:
    def __init__(self):
        openai.api_key = settings.OPENAI_API_KEY
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
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": context_prompt}],
                    temperature=0.3,
                    max_tokens=200
                )
                return response.choices[0].message.content.strip()
            except Exception:
                pass
        
        return enhancements.get('expanded_query', query)
    
    def _classify_query(self, query: str) -> Dict[str, Any]:\n        classification_prompt = prompt_templates.QUERY_CLASSIFICATION_PROMPT.format(query=query)\n        \n        try:\n            response = openai.ChatCompletion.create(\n                model=\"gpt-3.5-turbo\",\n                messages=[{\"role\": \"user\", \"content\": classification_prompt}],\n                temperature=0.1,\n                max_tokens=300\n            )\n            \n            result = response.choices[0].message.content.strip()\n            return json.loads(result)\n            \n        except Exception as e:\n            # Fallback classification\n            return self._fallback_classification(query)\n    \n    def _fallback_classification(self, query: str) -> Dict[str, Any]:\n        query_lower = query.lower()\n        \n        categories = []\n        confidence_scores = {}\n        \n        # Simple keyword-based classification\n        if any(word in query_lower for word in ['funding', 'investment', 'vc', 'capital', 'grant']):\n            categories.append('FUNDING')\n            confidence_scores['FUNDING'] = 0.8\n        \n        if any(word in query_lower for word in ['regulation', 'legal', 'compliance', 'registration']):\n            categories.append('REGULATORY')\n            confidence_scores['REGULATORY'] = 0.8\n        \n        if any(word in query_lower for word in ['market', 'industry', 'competition', 'trends']):\n            categories.append('MARKET')\n            confidence_scores['MARKET'] = 0.8\n        \n        if any(word in query_lower for word in ['success', 'case study', 'company', 'founder']):\n            categories.append('SUCCESS_STORIES')\n            confidence_scores['SUCCESS_STORIES'] = 0.8\n        \n        if not categories:\n            categories = ['GENERAL']\n            confidence_scores['GENERAL'] = 0.5\n        \n        return {\n            \"categories\": categories,\n            \"confidence_scores\": confidence_scores\n        }\n    \n    def _retrieve_documents(self, query: str, classification: Dict[str, Any]) -> List[Dict[str, Any]]:\n        # Create filters based on classification\n        filters = {}\n        \n        categories = classification.get('categories', [])\n        if categories and 'GENERAL' not in categories:\n            # Add category-based filtering if needed\n            pass\n        \n        # Perform vector search\n        results = self.vector_store.search(\n            query=query,\n            n_results=self.max_chunks,\n            filters=filters\n        )\n        \n        # Filter by similarity threshold\n        filtered_results = [\n            doc for doc in results \n            if doc.get('similarity', 0) >= self.similarity_threshold\n        ]\n        \n        return filtered_results[:self.max_chunks]\n    \n    def _generate_response(self, \n                          question: str, \n                          documents: List[Dict[str, Any]], \n                          history: List[Dict[str, str]] = None,\n                          classification: Dict[str, Any] = None) -> str:\n        \n        # Prepare context from retrieved documents\n        context_parts = []\n        for i, doc in enumerate(documents):\n            metadata = doc.get('metadata', {})\n            source_info = \"\"\n            \n            if metadata.get('source'):\n                source_info = f\"Source: {metadata['source']}\"\n            if metadata.get('country'):\n                source_info += f\" | Country: {metadata['country']}\"\n            if metadata.get('sector'):\n                source_info += f\" | Sector: {metadata['sector']}\"\n            \n            context_parts.append(f\"[{i+1}] {source_info}\\n{doc['content']}\\n\")\n        \n        context = \"\\n\".join(context_parts)\n        \n        # Determine primary category for context-aware prompting\n        primary_category = None\n        if classification and 'categories' in classification:\n            categories = classification['categories']\n            if categories:\n                primary_category = categories[0].lower()\n        \n        # Format the prompt\n        rag_prompt = prompt_templates.format_rag_prompt(\n            context=context,\n            question=question,\n            category=primary_category\n        )\n        \n        # Prepare messages\n        messages = [\n            {\"role\": \"system\", \"content\": prompt_templates.SYSTEM_PROMPT}\n        ]\n        \n        # Add conversation history if available\n        if history:\n            for msg in history[-5:]:  # Last 5 exchanges\n                if msg.get('role') and msg.get('content'):\n                    messages.append({\n                        \"role\": msg['role'],\n                        \"content\": msg['content']\n                    })\n        \n        messages.append({\"role\": \"user\", \"content\": rag_prompt})\n        \n        # Generate response\n        try:\n            response = openai.ChatCompletion.create(\n                model=\"gpt-4\" if len(context) > 4000 else \"gpt-3.5-turbo\",\n                messages=messages,\n                **settings.get_model_config()\n            )\n            \n            return response.choices[0].message.content.strip()\n            \n        except Exception as e:\n            return f\"I apologize, but I'm having trouble generating a response right now. Please try again. Error: {str(e)}\"\n    \n    def _generate_follow_up_questions(self, question: str, answer: str) -> List[str]:\n        try:\n            prompt = prompt_templates.FOLLOW_UP_QUESTIONS_PROMPT.format(\n                question=question,\n                answer=answer[:1000]  # Truncate for token limits\n            )\n            \n            response = openai.ChatCompletion.create(\n                model=\"gpt-3.5-turbo\",\n                messages=[{\"role\": \"user\", \"content\": prompt}],\n                temperature=0.7,\n                max_tokens=300\n            )\n            \n            result = response.choices[0].message.content.strip()\n            \n            # Parse the response into a list\n            follow_ups = []\n            for line in result.split('\\n'):\n                line = line.strip()\n                if line and (line.startswith('-') or line.startswith('•') or line[0].isdigit()):\n                    # Clean up the line\n                    cleaned = line.lstrip('-•0123456789. ').strip()\n                    if cleaned and '?' in cleaned:\n                        follow_ups.append(cleaned)\n            \n            return follow_ups[:5]  # Return max 5 follow-ups\n            \n        except Exception:\n            # Fallback follow-up questions\n            return [\n                \"Can you tell me more about this topic?\",\n                \"What are the next steps I should consider?\",\n                \"Are there any related opportunities or challenges?\"\n            ]\n    \n    def _format_sources(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:\n        sources = []\n        \n        for doc in documents:\n            metadata = doc.get('metadata', {})\n            \n            source = {\n                \"content_preview\": doc['content'][:200] + \"...\" if len(doc['content']) > 200 else doc['content'],\n                \"similarity\": round(doc.get('similarity', 0), 3),\n                \"metadata\": {}\n            }\n            \n            # Add relevant metadata\n            for key in ['source', 'country', 'sector', 'date', 'title']:\n                if metadata.get(key):\n                    source['metadata'][key] = metadata[key]\n            \n            sources.append(source)\n        \n        return sources\n    \n    def _calculate_confidence(self, documents: List[Dict[str, Any]]) -> float:\n        if not documents:\n            return 0.0\n        \n        # Calculate confidence based on similarity scores and number of relevant documents\n        similarities = [doc.get('similarity', 0) for doc in documents]\n        avg_similarity = sum(similarities) / len(similarities)\n        \n        # Factor in the number of high-quality results\n        high_quality_count = sum(1 for sim in similarities if sim > 0.8)\n        quality_factor = min(high_quality_count / 3, 1.0)  # Normalize to max 1.0\n        \n        confidence = (avg_similarity * 0.7) + (quality_factor * 0.3)\n        return round(confidence, 3)\n\nrag_engine = RAGEngine()"}