"""
Modern RAG Engine for UbuntuAI using LangChain
Implements self-reflective, corrective RAG with multi-provider support
Focused on Ghanaian startup ecosystem (fintech, agritech, healthtech)
"""

import logging
from typing import Dict, List, Any, Optional, AsyncGenerator
import json
from datetime import datetime
import asyncio

# LangChain imports
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain.schema.runnable import RunnableConfig

# LangSmith integration
try:
    from langsmith import trace
    from langfuse import Langfuse
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False
    def trace(name):
        def decorator(func):
            return func
        return decorator

from config.settings import settings
from config.prompts import prompt_templates
from api.llm_providers import llm_manager
from api.retrieval import get_contextual_retriever, initialize_retrieval_system
from api.evaluation import ragas_evaluator, reflection_evaluator, continuous_evaluator
from utils.context_enhancer import context_enhancer

logger = logging.getLogger(__name__)

class ModernRAGEngine:
    """
    Modern RAG Engine with self-reflection and corrective capabilities
    Focused on Ghanaian startup ecosystem
    """
    
    def __init__(self, vector_store):
        """Initialize the modern RAG engine"""
        self.vector_store = vector_store
        
        # Initialize retrieval system
        initialize_retrieval_system(vector_store)
        self.retriever = get_contextual_retriever()
        
        # Initialize LLM manager
        self.llm_manager = llm_manager
        
        # Initialize monitoring
        self.monitoring = self._setup_monitoring()
        
        # RAG chains
        self.rag_chain = None
        self.reflection_chain = None
        self.correction_chain = None
        
        self._setup_chains()
        
        logger.info("Modern RAG Engine initialized successfully for Ghanaian startup ecosystem")
    
    def _setup_monitoring(self) -> Optional[Any]:
        """Setup monitoring and observability"""
        if not MONITORING_AVAILABLE or not settings.USE_LANGFUSE:
            return None
        
        try:
            langfuse = Langfuse(
                secret_key=settings.LANGFUSE_SECRET_KEY,
                public_key=settings.LANGFUSE_PUBLIC_KEY,
                host=settings.LANGFUSE_HOST
            )
            logger.info("LangFuse monitoring initialized")
            return langfuse
        except Exception as e:
            logger.warning(f"Failed to initialize monitoring: {e}")
            return None
    
    def _setup_chains(self):
        """Setup LangChain LCEL chains"""
        
        # Main RAG chain
        self.rag_chain = self._create_rag_chain()
        
        # Self-reflection chain
        if settings.USE_SELF_REFLECTION:
            self.reflection_chain = self._create_reflection_chain()
        
        # Correction chain for CRAG
        self.correction_chain = self._create_correction_chain()
    
    def _create_rag_chain(self):
        """Create the main RAG chain using LCEL"""
        
        # Context formatting function
        def format_context(docs):
            if not docs:
                return "No relevant documents found in the Ghanaian startup ecosystem knowledge base."
            
            context_parts = []
            for i, doc in enumerate(docs[:settings.MAX_CONTEXT_CHUNKS]):
                content = doc.page_content.strip()
                metadata = doc.metadata
                
                # Format source information
                source_info = f"Source {i+1}: "
                if metadata.get('source'):
                    source_info += f"{metadata.get('source', 'Unknown')}"
                if metadata.get('sector_relevance'):
                    source_info += f" (Relevant to: {', '.join(metadata.get('sector_relevance', []))})"
                
                context_parts.append(f"{content}\n\n{source_info}")
            
            return "\n\n---\n\n".join(context_parts)
        
        # User context formatting
        def format_user_context(context):
            if not context:
                return ""
            
            context_parts = []
            if context.get('sector'):
                context_parts.append(f"User's primary sector: {context['sector']}")
            if context.get('region'):
                context_parts.append(f"User's primary region: {context['region']}")
            if context.get('experience_level'):
                context_parts.append(f"User's experience level: {context['experience_level']}")
            
            if context_parts:
                return f"User Context: {'; '.join(context_parts)}"
            return ""
        
        # Create the chain
        chain = (
            {
                "context": self.retriever | format_context,
                "user_context": RunnablePassthrough() | format_user_context,
                "question": RunnablePassthrough()
            }
            | prompt_templates.RAG_PROMPT_TEMPLATE
            | self.llm_manager.get_langchain_llm()
            | StrOutputParser()
        )
        
        return chain
    
    def _create_reflection_chain(self):
        """Create the self-reflection chain"""
        
        reflection_prompt = ChatPromptTemplate.from_template("""
        You are an AI assistant evaluating the quality of your own response about the Ghanaian startup ecosystem.
        
        Question: {question}
        Your Answer: {answer}
        Context Used: {context}
        
        Please evaluate your response on these criteria:
        1. **Accuracy**: Is the information correct and up-to-date for Ghana?
        2. **Relevance**: Does it directly address the question about Ghanaian startups?
        3. **Completeness**: Does it provide sufficient information for the user's needs?
        4. **Ghanaian Focus**: Does it maintain focus on Ghana and the specified sectors (fintech, agritech, healthtech)?
        5. **Actionability**: Does it provide practical next steps for Ghanaian entrepreneurs?
        
        Provide a JSON response with:
        - overall_score (0-100)
        - strengths (list of what was done well)
        - areas_for_improvement (list of what could be better)
        - confidence_in_evaluation (0-100)
        """)
        
        chain = (
            {
                "question": RunnablePassthrough(),
                "answer": RunnablePassthrough(),
                "context": RunnablePassthrough()
            }
            | reflection_prompt
            | self.llm_manager.get_langchain_llm()
            | JsonOutputParser()
        )
        
        return chain
    
    def _create_correction_chain(self):
        """Create the correction chain for CRAG"""
        
        correction_prompt = ChatPromptTemplate.from_template("""
        Based on the reflection results, improve your previous answer about the Ghanaian startup ecosystem.
        
        Original Question: {question}
        Original Answer: {answer}
        Reflection Results: {reflection}
        Context: {context}
        
        Please provide an improved answer that addresses the identified areas for improvement.
        Focus on:
        - Ghanaian specificity and relevance
        - Accuracy and completeness
        - Practical guidance for Ghanaian entrepreneurs
        - Sector-specific insights (fintech, agritech, healthtech)
        
        Improved Answer:
        """)
        
        chain = (
            {
                "question": RunnablePassthrough(),
                "answer": RunnablePassthrough(),
                "reflection": RunnablePassthrough(),
                "context": RunnablePassthrough()
            }
            | correction_prompt
            | self.llm_manager.get_langchain_llm()
            | StrOutputParser()
        )
        
        return chain
    
    @trace("rag_query")
    def query(self, 
             question: str,
             conversation_history: List[Dict[str, str]] = None,
             user_context: Dict[str, Any] = None,
             provider: str = None) -> Dict[str, Any]:
        """
        Process a query using the RAG engine
        Focused on Ghanaian startup ecosystem
        """
        
        start_time = datetime.now()
        
        try:
            # Validate input
            if not question or not question.strip():
                return self._create_error_response("", "Question cannot be empty")
            
            # Prepare context
            conversation_history = conversation_history or []
            user_context = user_context or {}
            
            # Retrieve relevant documents
            documents = self._retrieve_with_context(question, conversation_history, user_context)
            
            if not documents:
                logger.warning("No relevant documents found for query")
                return self._create_fallback_response(question)
            
            # Generate initial response
            initial_answer = self._generate_response(question, documents, user_context, provider)
            
            # Perform self-reflection if enabled
            reflection_result = None
            if settings.USE_SELF_REFLECTION and self.reflection_chain:
                reflection_result = self._perform_self_reflection(question, initial_answer, documents)
                
                # Perform correction if reflection indicates issues
                if reflection_result and reflection_result.get('overall_score', 100) < 70:
                    corrected_answer = self._perform_correction(question, initial_answer, reflection_result, documents)
                    if corrected_answer:
                        initial_answer = corrected_answer
            
            # Generate follow-up questions
            follow_up_questions = self._generate_follow_ups(question, initial_answer)
            
            # Calculate confidence
            confidence = self._calculate_confidence(documents, reflection_result)
            
            # Format sources
            sources = self._format_sources(documents)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            response = {
                "answer": initial_answer,
                "sources": sources,
                "follow_up_questions": follow_up_questions,
                "confidence": confidence,
                "processing_time": f"{processing_time:.2f}s",
                "chunks_retrieved": len(documents),
                "provider": provider or settings.PRIMARY_LLM_PROVIDER,
                "ghana_focus": True,
                "sectors_covered": self._identify_sectors_covered(documents)
            }
            
            logger.info(f"Query processed successfully in {processing_time:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return self._create_error_response(question, str(e))
    
    def _retrieve_with_context(self, 
                             question: str,
                             conversation_history: List[Dict[str, str]],
                             user_context: Dict[str, Any]) -> List[Document]:
        """Retrieve relevant documents with context enhancement"""
        
        try:
            # Enhance query with context
            enhanced_query = question
            
            # Add user context to query
            if user_context.get('sector'):
                enhanced_query += f" Focus on {user_context['sector']} sector in Ghana."
            if user_context.get('region'):
                enhanced_query += f" Consider {user_context['region']} region context."
            
            # Add conversation history context
            if conversation_history:
                recent_context = " ".join([msg.get('content', '') for msg in conversation_history[-3:]])
                if recent_context:
                    enhanced_query += f" Previous context: {recent_context}"
            
            # Retrieve documents
            documents = self.retriever.get_relevant_documents(enhanced_query)
            
            # Filter for Ghanaian relevance
            ghana_relevant_docs = []
            for doc in documents:
                if self._is_ghana_relevant(doc, user_context):
                    ghana_relevant_docs.append(doc)
            
            # If no Ghana-specific docs, use all docs but mark as general
            if not ghana_relevant_docs and documents:
                ghana_relevant_docs = documents
            
            logger.info(f"Retrieved {len(ghana_relevant_docs)} relevant documents")
            return ghana_relevant_docs
            
        except Exception as e:
            logger.error(f"Error in document retrieval: {e}")
            return []
    
    def _is_ghana_relevant(self, doc: Document, user_context: Dict[str, Any]) -> bool:
        """Check if document is relevant to Ghanaian startup ecosystem"""
        
        content = doc.page_content.lower()
        metadata = doc.metadata
        
        # Check for Ghana-specific keywords
        ghana_keywords = [
            "ghana", "ghanian", "accra", "kumasi", "tamale", "tema", "koforidua",
            "sunyani", "ho", "bolgatanga", "wa", "damongo", "goaso", "techiman"
        ]
        
        # Check for sector-specific keywords
        sector_keywords = {
            "fintech": ["fintech", "financial", "banking", "mobile money", "digital payments"],
            "agritech": ["agritech", "agriculture", "farming", "crop", "livestock"],
            "healthtech": ["healthtech", "healthcare", "medical", "pharmaceutical"]
        }
        
        # Check Ghana relevance
        is_ghana_relevant = any(keyword in content for keyword in ghana_keywords)
        
        # Check sector relevance
        user_sector = user_context.get('sector', 'general')
        is_sector_relevant = False
        
        if user_sector in sector_keywords:
            is_sector_relevant = any(keyword in content for keyword in sector_keywords[user_sector])
        
        # Document is relevant if it's Ghana-related or sector-related
        return is_ghana_relevant or is_sector_relevant
    
    def _generate_response(self, 
                         question: str,
                         documents: List[Document],
                         user_context: Dict[str, Any],
                         provider: str = None) -> str:
        """Generate response using the RAG chain"""
        
        try:
            # Prepare context for the chain
            chain_input = {
                "question": question,
                "user_context": user_context
            }
            
            # Execute the chain
            response = self.rag_chain.invoke(chain_input)
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"I apologize, but I encountered an error while processing your question about the Ghanaian startup ecosystem. Please try rephrasing your question or ask about a different aspect of fintech, agritech, or healthtech in Ghana."
    
    def _perform_self_reflection(self, 
                               question: str, 
                               answer: str, 
                               documents: List[Document]) -> Dict[str, Any]:
        """Perform self-reflection on the generated answer"""
        
        try:
            if not self.reflection_chain:
                return None
            
            # Prepare context for reflection
            context_summary = "\n".join([doc.page_content[:200] + "..." for doc in documents[:3]])
            
            reflection_input = {
                "question": question,
                "answer": answer,
                "context": context_summary
            }
            
            # Execute reflection
            reflection_result = self.reflection_chain.invoke(reflection_input)
            
            logger.info(f"Self-reflection completed with score: {reflection_result.get('overall_score', 'N/A')}")
            return reflection_result
            
        except Exception as e:
            logger.error(f"Error in self-reflection: {e}")
            return None
    
    def _perform_correction(self, 
                          question: str,
                          original_answer: str,
                          reflection_result: Dict[str, Any],
                          documents: List[Document]) -> str:
        """Perform correction based on reflection results"""
        
        try:
            if not self.correction_chain:
                return original_answer
            
            # Prepare context for correction
            context_summary = "\n".join([doc.page_content[:200] + "..." for doc in documents[:3]])
            
            correction_input = {
                "question": question,
                "answer": original_answer,
                "reflection": json.dumps(reflection_result),
                "context": context_summary
            }
            
            # Execute correction
            corrected_answer = self.correction_chain.invoke(correction_input)
            
            logger.info("Answer corrected based on self-reflection")
            return corrected_answer
            
        except Exception as e:
            logger.error(f"Error in correction: {e}")
            return original_answer
    
    def _generate_follow_ups(self, question: str, answer: str) -> List[str]:
        """Generate relevant follow-up questions"""
        
        try:
            # Use the prompt template for follow-up questions
            follow_up_prompt = prompt_templates.FOLLOW_UP_QUESTIONS_PROMPT.format(
                question=question,
                answer=answer
            )
            
            # Generate follow-ups using LLM
            follow_ups = self.llm_manager.generate(
                follow_up_prompt,
                provider=settings.PRIMARY_LLM_PROVIDER
            )
            
            # Parse and clean follow-ups
            if follow_ups:
                # Simple parsing - split by newlines and clean
                questions = [q.strip() for q in follow_ups.split('\n') if q.strip()]
                # Limit to 3 questions
                return questions[:3]
            
            return self._get_default_follow_ups()
            
        except Exception as e:
            logger.error(f"Error generating follow-ups: {e}")
            return self._get_default_follow_ups()
    
    def _get_default_follow_ups(self) -> List[str]:
        """Get default follow-up questions for Ghanaian startup ecosystem"""
        
        return [
            "What are the key regulatory requirements for this sector in Ghana?",
            "Are there any government programs or incentives available?",
            "What are the main challenges entrepreneurs face in this area?"
        ]
    
    def _format_sources(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """Format source documents for display"""
        
        sources = []
        
        for doc in documents[:5]:  # Limit to top 5 sources
            source = {
                "title": doc.metadata.get('source', 'Unknown Source'),
                "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                "relevance": "High" if doc.metadata.get('sector_relevance') else "Medium",
                "metadata": doc.metadata
            }
            sources.append(source)
        
        return sources
    
    def _calculate_confidence(self, 
                            documents: List[Document],
                            reflection_result: Dict[str, Any] = None) -> float:
        """Calculate confidence score for the response"""
        
        try:
            # Base confidence from document relevance
            base_confidence = min(len(documents) / 10.0, 1.0)  # More docs = higher confidence
            
            # Adjust based on reflection if available
            if reflection_result and reflection_result.get('overall_score'):
                reflection_score = reflection_result['overall_score'] / 100.0
                # Weight: 70% base confidence, 30% reflection
                final_confidence = (0.7 * base_confidence) + (0.3 * reflection_score)
            else:
                final_confidence = base_confidence
            
            return min(final_confidence, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5
    
    def _identify_sectors_covered(self, documents: List[Document]) -> List[str]:
        """Identify which Ghanaian sectors are covered in the documents"""
        
        sectors_covered = set()
        
        for doc in documents:
            sector_relevance = doc.metadata.get('sector_relevance', [])
            sectors_covered.update(sector_relevance)
        
        return list(sectors_covered) if sectors_covered else ["general"]
    
    def _create_fallback_response(self, question: str) -> Dict[str, Any]:
        """Create a fallback response when no documents are found"""
        
        return {
            "answer": f"I apologize, but I don't have enough specific information about '{question}' in my Ghanaian startup ecosystem knowledge base. This could be because:\n\n1. The topic is outside my current knowledge scope\n2. The information needs to be updated\n3. The question is too specific for my current data\n\nPlease try:\n- Rephrasing your question\n- Asking about broader aspects of fintech, agritech, or healthtech in Ghana\n- Contacting relevant Ghanaian business organizations for the most current information",
            "sources": [],
            "follow_up_questions": [
                "What are the main challenges in the Ghanaian startup ecosystem?",
                "How can I find funding opportunities in Ghana?",
                "What are the key regulations for startups in Ghana?"
            ],
            "confidence": 0.1,
            "processing_time": "0.00s",
            "chunks_retrieved": 0,
            "provider": settings.PRIMARY_LLM_PROVIDER,
            "ghana_focus": True,
            "sectors_covered": ["general"]
        }
    
    def _create_error_response(self, question: str, error: str) -> Dict[str, Any]:
        """Create an error response"""
        
        return {
            "answer": f"I encountered an error while processing your question about the Ghanaian startup ecosystem: {error}. Please try again or rephrase your question about fintech, agritech, or healthtech in Ghana.",
            "sources": [],
            "follow_up_questions": [],
            "confidence": 0.0,
            "processing_time": "0.00s",
            "chunks_retrieved": 0,
            "provider": settings.PRIMARY_LLM_PROVIDER,
            "ghana_focus": True,
            "sectors_covered": ["general"],
            "error": error
        }
    
    async def query_stream(self, 
                          question: str,
                          conversation_history: List[Dict[str, str]] = None,
                          user_context: Dict[str, Any] = None,
                          provider: str = None) -> AsyncGenerator[Dict[str, Any], None]:
        """Streaming version of query processing"""
        
        start_time = datetime.now()
        
        try:
            # Validate input
            if not question or not question.strip():
                yield self._create_error_response("", "Question cannot be empty")
                return
            
            # Prepare context
            conversation_history = conversation_history or []
            user_context = user_context or {}
            
            # Retrieve relevant documents
            documents = self._retrieve_with_context(question, conversation_history, user_context)
            
            if not documents:
                yield self._create_fallback_response(question)
                return
            
            # Generate streaming response
            streaming_prompt = self._create_streaming_prompt(question, documents, user_context)
            
            # Stream the response
            async for chunk in self.llm_manager.generate_stream(
                streaming_prompt,
                provider=provider
            ):
                yield {
                    "type": "chunk",
                    "content": chunk,
                    "timestamp": datetime.now().isoformat()
                }
            
            # Final metadata
            processing_time = (datetime.now() - start_time).total_seconds()
            
            yield {
                "type": "complete",
                "metadata": {
                    "processing_time": f"{processing_time:.2f}s",
                    "chunks_retrieved": len(documents),
                    "provider": provider or settings.PRIMARY_LLM_PROVIDER,
                    "ghana_focus": True
                }
            }
            
        except Exception as e:
            logger.error(f"Error in streaming query: {e}")
            yield self._create_error_response(question, str(e))
    
    def _create_streaming_prompt(self, 
                               question: str,
                               documents: List[Document],
                               user_context: Dict[str, Any]) -> str:
        """Create a prompt optimized for streaming responses"""
        
        # Format context
        context_parts = []
        for i, doc in enumerate(documents[:settings.MAX_CONTEXT_CHUNKS]):
            content = doc.page_content.strip()
            source = doc.metadata.get('source', 'Unknown')
            context_parts.append(f"Source {i+1} ({source}): {content}")
        
        context = "\n\n".join(context_parts)
        
        # Create streaming prompt
        prompt = f"""Based on the following context about the Ghanaian startup ecosystem, answer the user's question in a clear, helpful way. Focus on providing practical information for Ghanaian entrepreneurs.

Context:
{context}

Question: {question}

User Context: {user_context.get('sector', 'General')} sector, {user_context.get('region', 'Ghana')} region

Answer:"""
        
        return prompt

def initialize_rag_engine(vector_store):
    """Initialize the RAG engine with the vector store"""
    global rag_engine
    try:
        rag_engine = ModernRAGEngine(vector_store)
        logger.info("RAG engine initialized successfully")
        return rag_engine
    except Exception as e:
        logger.error(f"Failed to initialize RAG engine: {e}")
        return None

def get_rag_engine() -> Optional[ModernRAGEngine]:
    """Get the global RAG engine instance"""
    return rag_engine

# Global instance
rag_engine = None

class RAGEngineWrapper:
    """Wrapper class for RAG engine to handle initialization"""
    
    def __init__(self):
        self.engine = None
    
    def set_engine(self, engine):
        self.engine = engine
    
    def query(self, question: str, **kwargs) -> Dict[str, Any]:
        if self.engine:
            return self.engine.query(question, **kwargs)
        else:
            return {"error": "RAG engine not initialized"}