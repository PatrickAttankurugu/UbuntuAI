"""
Modern RAG Engine for UbuntuAI using LangChain
Implements self-reflective, corrective RAG with multi-provider support
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
        
        logger.info("Modern RAG Engine initialized successfully")
    
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
        
        # System prompt template
        system_template = prompt_templates.SYSTEM_PROMPT
        
        # RAG prompt template
        rag_template = """Based on the following context, please answer the question.

Context:
{context}

User Context:
{user_context}

Question: {question}

Provide a comprehensive answer that:
1. Directly addresses the question
2. Uses the provided context effectively
3. Considers the user's background and needs
4. Offers practical insights and next steps

Answer:"""
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("human", rag_template)
        ])
        
        # Create the chain
        def format_context(docs):
            return "\n\n".join([
                f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}"
                for doc in docs
            ])
        
        def format_user_context(context):
            if not context:
                return "No specific user context provided."
            
            parts = []
            for key, value in context.items():
                if key in ['country', 'sector', 'business_stage', 'name']:
                    parts.append(f"{key.title()}: {value}")
            
            return " | ".join(parts) if parts else "No specific user context provided."
        
        # Create runnable chain
        chain = (
            {
                "context": lambda x: format_context(x["documents"]),
                "user_context": lambda x: format_user_context(x.get("user_context")),
                "question": lambda x: x["question"]
            }
            | prompt
            | self.llm_manager.get_langchain_llm()
            | StrOutputParser()
        )
        
        return chain
    
    def _create_reflection_chain(self):
        """Create self-reflection chain"""
        
        reflection_template = """Please evaluate the quality of this answer to the given question:

Question: {question}
Answer: {answer}
Context Used: {context}

Evaluate the answer on these criteria:
1. Relevance: Does it address the question?
2. Accuracy: Is the information correct based on context?
3. Completeness: Does it fully answer the question?
4. Clarity: Is it well-structured and understandable?

For each criterion, provide:
- Score (1-10)
- Brief explanation

Also suggest any improvements if the answer could be enhanced.

Provide your evaluation in JSON format:
{{
    "relevance": {{"score": X, "explanation": "..."}},
    "accuracy": {{"score": X, "explanation": "..."}},
    "completeness": {{"score": X, "explanation": "..."}},
    "clarity": {{"score": X, "explanation": "..."}},
    "overall_score": X,
    "improvements": ["..."],
    "requires_correction": true/false
}}"""
        
        prompt = PromptTemplate(
            template=reflection_template,
            input_variables=["question", "answer", "context"]
        )
        
        chain = (
            prompt
            | self.llm_manager.get_langchain_llm()
            | StrOutputParser()
        )
        
        return chain
    
    def _create_correction_chain(self):
        """Create corrective RAG chain"""
        
        correction_template = """The following answer has been identified as needing improvement:

Original Question: {question}
Original Answer: {answer}
Issues Identified: {issues}
Additional Context: {additional_context}

Please provide an improved answer that addresses the identified issues while maintaining accuracy and relevance.

Improved Answer:"""
        
        prompt = PromptTemplate(
            template=correction_template,
            input_variables=["question", "answer", "issues", "additional_context"]
        )
        
        chain = (
            prompt
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
        Process query through modern RAG pipeline
        
        Args:
            question: User's question
            conversation_history: Previous conversation context
            user_context: User profile and preferences
            provider: Specific LLM provider to use
            
        Returns:
            Complete RAG response with metadata
        """
        
        try:
            # Step 1: Enhanced retrieval
            retrieved_docs = self._retrieve_with_context(
                question, conversation_history, user_context
            )
            
            if not retrieved_docs:
                return self._create_fallback_response(question)
            
            # Step 2: Generate initial response
            initial_response = self._generate_response(
                question, retrieved_docs, user_context, provider
            )
            
            # Step 3: Self-reflection (if enabled)
            reflection_result = None
            if settings.USE_SELF_REFLECTION and self.reflection_chain:
                reflection_result = self._perform_self_reflection(
                    question, initial_response, retrieved_docs
                )
            
            # Step 4: Correction (if needed)
            final_response = initial_response
            if reflection_result and reflection_result.get("requires_correction"):
                final_response = self._perform_correction(
                    question, initial_response, reflection_result, retrieved_docs
                )
            
            # Step 5: Generate follow-up questions
            follow_ups = self._generate_follow_ups(question, final_response)
            
            # Step 6: Create response object
            response = {
                "answer": final_response,
                "sources": self._format_sources(retrieved_docs),
                "follow_up_questions": follow_ups,
                "metadata": {
                    "retrieval_method": getattr(retrieved_docs[0], 'retrieval_method', 'unknown') if retrieved_docs else 'none',
                    "num_sources": len(retrieved_docs),
                    "llm_provider": provider or settings.PRIMARY_LLM_PROVIDER,
                    "self_reflection": reflection_result,
                    "was_corrected": reflection_result.get("requires_correction", False) if reflection_result else False,
                    "timestamp": datetime.now().isoformat()
                },
                "confidence": self._calculate_confidence(retrieved_docs, reflection_result)
            }
            
            # Step 7: Add to evaluation queue
            if continuous_evaluator:
                continuous_evaluator.add_to_evaluation_queue(
                    query=question,
                    answer=final_response,
                    contexts=[doc.page_content for doc in retrieved_docs[:3]],
                    metadata={"user_context": user_context}
                )
            
            return response
            
        except Exception as e:
            logger.error(f"RAG query failed: {e}")
            return self._create_error_response(question, str(e))
    
    def _retrieve_with_context(self, 
                             question: str,
                             conversation_history: List[Dict[str, str]],
                             user_context: Dict[str, Any]) -> List[Document]:
        """Retrieve documents with contextual awareness"""
        
        if not self.retriever:
            logger.error("Retriever not initialized")
            return []
        
        try:
            # Use contextual retriever
            retrieval_results = self.retriever.retrieve_with_context(
                query=question,
                conversation_history=conversation_history,
                user_context=user_context
            )
            
            # Convert to LangChain Documents
            documents = []
            for result in retrieval_results:
                doc = Document(
                    page_content=result.content,
                    metadata={
                        **result.metadata,
                        "similarity_score": result.score,
                        "retrieval_method": result.retrieval_method,
                        "rerank_score": result.rerank_score
                    }
                )
                documents.append(doc)
            
            logger.debug(f"Retrieved {len(documents)} documents for query: {question[:100]}")
            return documents
            
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return []
    
    def _generate_response(self, 
                         question: str,
                         documents: List[Document],
                         user_context: Dict[str, Any],
                         provider: str = None) -> str:
        """Generate response using RAG chain"""
        
        try:
            # Prepare input for chain
            chain_input = {
                "question": question,
                "documents": documents,
                "user_context": user_context or {}
            }
            
            # Use specific provider if requested
            if provider and provider in self.llm_manager.get_available_providers():
                # Create temporary chain with specific provider
                temp_llm = self.llm_manager.get_langchain_llm(provider)
                temp_chain = self.rag_chain.with_config(
                    configurable={"llm": temp_llm}
                )
                response = temp_chain.invoke(chain_input)
            else:
                response = self.rag_chain.invoke(chain_input)
            
            return response
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return f"I apologize, but I encountered an error generating a response: {str(e)}"
    
    def _perform_self_reflection(self, 
                               question: str,
                               answer: str,
                               documents: List[Document]) -> Dict[str, Any]:
        """Perform self-reflection on the generated answer"""
        
        try:
            context_summary = "\n".join([
                doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                for doc in documents[:3]
            ])
            
            reflection_input = {
                "question": question,
                "answer": answer,
                "context": context_summary
            }
            
            reflection_response = self.reflection_chain.invoke(reflection_input)
            
            # Parse JSON response
            try:
                reflection_data = json.loads(reflection_response)
                return reflection_data
            except json.JSONDecodeError:
                # Fallback parsing
                return {
                    "overall_score": 7.0,
                    "requires_correction": False,
                    "raw_response": reflection_response
                }
                
        except Exception as e:
            logger.error(f"Self-reflection failed: {e}")
            return {"overall_score": 7.0, "requires_correction": False}
    
    def _perform_correction(self, 
                          question: str,
                          original_answer: str,
                          reflection_result: Dict[str, Any],
                          documents: List[Document]) -> str:
        """Perform corrective RAG if needed"""
        
        try:
            # Extract issues from reflection
            issues = []
            for criterion in ["relevance", "accuracy", "completeness", "clarity"]:
                if criterion in reflection_result:
                    criterion_data = reflection_result[criterion]
                    if isinstance(criterion_data, dict) and criterion_data.get("score", 10) < 7:
                        issues.append(f"{criterion}: {criterion_data.get('explanation', 'Needs improvement')}")
            
            if reflection_result.get("improvements"):
                issues.extend(reflection_result["improvements"])
            
            # Additional context from documents
            additional_context = "\n".join([
                doc.page_content for doc in documents[:2]
            ])
            
            correction_input = {
                "question": question,
                "answer": original_answer,
                "issues": "; ".join(issues),
                "additional_context": additional_context
            }
            
            corrected_answer = self.correction_chain.invoke(correction_input)
            
            logger.info("Answer corrected based on self-reflection")
            return corrected_answer
            
        except Exception as e:
            logger.error(f"Correction failed: {e}")
            return original_answer
    
    def _generate_follow_ups(self, question: str, answer: str) -> List[str]:
        """Generate relevant follow-up questions"""
        
        try:
            follow_up_prompt = f"""Based on this Q&A about African business, suggest 3-4 relevant follow-up questions:

Question: {question}
Answer: {answer[:500]}...

Generate follow-up questions that would help the user:
1. Dive deeper into the topic
2. Explore related opportunities
3. Understand practical next steps
4. Learn about regional variations

Return only the questions, one per line."""
            
            response = self.llm_manager.generate(follow_up_prompt)
            
            # Parse follow-up questions
            questions = []
            for line in response.split('\n'):
                line = line.strip()
                if line and ('?' in line or line.endswith('.')):
                    # Clean up the question
                    clean_line = line.lstrip('1234567890.-• ').strip()
                    if clean_line:
                        questions.append(clean_line)
            
            return questions[:4]
            
        except Exception as e:
            logger.error(f"Follow-up generation failed: {e}")
            return self._get_default_follow_ups()
    
    def _get_default_follow_ups(self) -> List[str]:
        """Get default follow-up questions"""
        return [
            "Can you provide more specific examples?",
            "What are the key challenges in this area?",
            "How does this vary across different African countries?",
            "What are the next steps I should consider?"
        ]
    
    def _format_sources(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """Format source documents for response"""
        
        sources = []
        for doc in documents:
            metadata = doc.metadata
            
            source = {
                "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "similarity_score": metadata.get("similarity_score", 0.0),
                "metadata": {
                    key: value for key, value in metadata.items()
                    if key not in ["similarity_score", "rerank_score"]
                }
            }
            
            if metadata.get("rerank_score"):
                source["rerank_score"] = metadata["rerank_score"]
            
            sources.append(source)
        
        return sources
    
    def _calculate_confidence(self, 
                            documents: List[Document],
                            reflection_result: Dict[str, Any] = None) -> float:
        """Calculate confidence score for the response"""
        
        if not documents:
            return 0.0
        
        # Base confidence from retrieval scores
        scores = [doc.metadata.get("similarity_score", 0.0) for doc in documents]
        base_confidence = sum(scores) / len(scores) if scores else 0.0
        
        # Factor in number of high-quality sources
        high_quality_count = sum(1 for score in scores if score > 0.8)
        quality_factor = min(high_quality_count / 3, 1.0)
        
        # Factor in self-reflection if available
        reflection_factor = 1.0
        if reflection_result:
            overall_score = reflection_result.get("overall_score", 7.0)
            reflection_factor = min(overall_score / 10.0, 1.0)
        
        # Combined confidence
        confidence = (base_confidence * 0.5) + (quality_factor * 0.3) + (reflection_factor * 0.2)
        
        return round(min(confidence, 1.0), 3)
    
    def _create_fallback_response(self, question: str) -> Dict[str, Any]:
        """Create fallback response when no documents found"""
        
        return {
            "answer": "I apologize, but I couldn't find relevant information to answer your question. Could you please rephrase or provide more context?",
            "sources": [],
            "follow_up_questions": [
                "Could you provide more specific details about what you're looking for?",
                "Are you interested in a particular African country or region?",
                "What specific aspect of this topic would you like to explore?"
            ],
            "metadata": {
                "retrieval_method": "none",
                "num_sources": 0,
                "was_fallback": True
            },
            "confidence": 0.0
        }
    
    def _create_error_response(self, question: str, error: str) -> Dict[str, Any]:
        """Create error response"""
        
        return {
            "answer": f"I encountered an error processing your question: {error}. Please try again.",
            "sources": [],
            "follow_up_questions": [],
            "metadata": {
                "error": error,
                "timestamp": datetime.now().isoformat()
            },
            "confidence": 0.0
        }
    
    async def query_stream(self, 
                          question: str,
                          conversation_history: List[Dict[str, str]] = None,
                          user_context: Dict[str, Any] = None,
                          provider: str = None) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream RAG response for real-time user experience"""
        
        try:
            # Retrieve documents first
            documents = self._retrieve_with_context(
                question, conversation_history, user_context
            )
            
            if not documents:
                yield {"type": "answer", "content": "No relevant information found."}
                return
            
            # Stream the response generation
            async for chunk in self.llm_manager.generate_stream(
                prompt=self._create_streaming_prompt(question, documents, user_context),
                provider=provider
            ):
                yield {"type": "answer_chunk", "content": chunk}
            
            # Send sources and metadata
            yield {
                "type": "sources",
                "content": self._format_sources(documents)
            }
            
            yield {
                "type": "metadata",
                "content": {
                    "num_sources": len(documents),
                    "confidence": self._calculate_confidence(documents)
                }
            }
            
        except Exception as e:
            logger.error(f"Streaming query failed: {e}")
            yield {"type": "error", "content": str(e)}
    
    def _create_streaming_prompt(self, 
                               question: str,
                               documents: List[Document],
                               user_context: Dict[str, Any]) -> str:
        """Create prompt for streaming response"""
        
        context = "\n\n".join([
            f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}"
            for doc in documents[:5]
        ])
        
        user_info = ""
        if user_context:
            info_parts = []
            for key in ['country', 'sector', 'business_stage']:
                if user_context.get(key):
                    info_parts.append(f"{key}: {user_context[key]}")
            if info_parts:
                user_info = f"\nUser Context: {' | '.join(info_parts)}"
        
        return f"""{prompt_templates.SYSTEM_PROMPT}

Context:
{context}
{user_info}

Question: {question}

Please provide a comprehensive answer based on the context above."""

# Initialize global RAG engine (will be set up with vector store)
modern_rag_engine = None

def initialize_rag_engine(vector_store):
    """Initialize the global RAG engine"""
    global modern_rag_engine
    
    try:
        modern_rag_engine = ModernRAGEngine(vector_store)
        logger.info("Modern RAG Engine initialized successfully")
        return modern_rag_engine
    except Exception as e:
        logger.error(f"Failed to initialize RAG engine: {e}")
        return None

def get_rag_engine() -> Optional[ModernRAGEngine]:
    """Get the global RAG engine instance"""
    return modern_rag_engine

# For backward compatibility, maintain the rag_engine interface
class RAGEngineWrapper:
    """Wrapper to maintain backward compatibility"""
    
    def __init__(self):
        self.modern_engine = None
    
    def set_engine(self, engine):
        self.modern_engine = engine
    
    def query(self, question: str, **kwargs) -> Dict[str, Any]:
        if self.modern_engine:
            return self.modern_engine.query(question, **kwargs)
        else:
            return {
                "answer": "RAG engine not initialized",
                "sources": [],
                "follow_up_questions": [],
                "confidence": 0.0
            }

# Global wrapper for backward compatibility
rag_engine = RAGEngineWrapper()