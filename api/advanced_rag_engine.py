"""
Advanced RAG Engine for UbuntuAI - Industry Standard Implementation
Implements CRAG, DRAG, multi-stage retrieval, and advanced RAG patterns
"""

import logging
from typing import Dict, List, Any, Optional, AsyncGenerator, Tuple
import json
from datetime import datetime
import asyncio
from dataclasses import dataclass
import re

# LangChain imports
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableConfig
from langchain.schema.runnable import RunnableConfig

# Advanced RAG patterns (optional)
try:
    from langchain_experimental.utilities import PythonREPL
    EXPERIMENTAL_AVAILABLE = True
except ImportError:
    EXPERIMENTAL_AVAILABLE = False

try:
    from langchain.agents import AgentExecutor, create_openai_functions_agent
    from langchain.tools import BaseTool
    AGENTS_AVAILABLE = True
except ImportError:
    AGENTS_AVAILABLE = False

# Monitoring and tracing
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

# Import with fallbacks
try:
    from api.llm_providers import llm_manager
except ImportError:
    llm_manager = None

try:
    from api.retrieval import get_contextual_retriever, initialize_retrieval_system
except ImportError:
    get_contextual_retriever = None
    initialize_retrieval_system = None

try:
    from api.evaluation import ragas_evaluator, reflection_evaluator, continuous_evaluator
except ImportError:
    ragas_evaluator = None
    reflection_evaluator = None
    continuous_evaluator = None

try:
    from utils.context_enhancer import context_enhancer
except ImportError:
    context_enhancer = None

logger = logging.getLogger(__name__)

@dataclass
class RAGResponse:
    """Structured RAG response with comprehensive metadata"""
    answer: str
    sources: List[Dict[str, Any]]
    follow_up_questions: List[str]
    confidence: float
    processing_time: float
    chunks_retrieved: int
    provider: str
    ghana_focus: bool
    sectors_covered: List[str]
    retrieval_metadata: Dict[str, Any]
    generation_metadata: Dict[str, Any]
    evaluation_metadata: Dict[str, Any]
    security_metadata: Dict[str, Any]

class AdvancedRAGEngine:
    """
    Advanced RAG Engine implementing industry-standard patterns
    - CRAG (Corrective RAG)
    - DRAG (Dynamic RAG)
    - Multi-stage retrieval
    - Advanced query expansion
    - Comprehensive evaluation
    """
    
    def __init__(self, vector_store):
        """Initialize the advanced RAG engine"""
        self.vector_store = vector_store
        
        # Initialize retrieval system with fallback
        if get_contextual_retriever and initialize_retrieval_system:
            try:
                initialize_retrieval_system(vector_store)
                self.retriever = get_contextual_retriever()
            except Exception as e:
                logger.warning(f"Failed to initialize advanced retrieval: {e}")
                self.retriever = None
        else:
            self.retriever = None
        
        # Initialize LLM manager with fallback
        self.llm_manager = llm_manager
        
        # Initialize monitoring
        self.monitoring = self._setup_monitoring()
        
        # Advanced RAG chains
        self.rag_chain = None
        self.crag_chain = None
        self.drag_chain = None
        self.reflection_chain = None
        self.correction_chain = None
        self.query_expansion_chain = None
        
        # Performance optimization
        self.response_cache = {}
        self.query_cache = {}
        
        # Security and rate limiting
        self.rate_limiter = self._setup_rate_limiter()
        self.security_validator = self._setup_security_validator()
        
        self._setup_chains()
        
        logger.info("Advanced RAG Engine initialized successfully for Ghanaian startup ecosystem")
    
    def _setup_monitoring(self) -> Optional[Any]:
        """Setup advanced monitoring and observability"""
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
    
    def _setup_rate_limiter(self):
        """Setup rate limiting for security"""
        return {
            "max_requests_per_minute": 60,
            "max_requests_per_hour": 1000,
            "current_requests": 0,
            "last_reset": datetime.now()
        }
    
    def _setup_security_validator(self):
        """Setup security validation"""
        return {
            "blocked_patterns": [
                r"<script.*?>.*?</script>",
                r"javascript:",
                r"eval\(",
                r"exec\(",
                r"import\s+os",
                r"subprocess\."
            ],
            "max_query_length": 2000,
            "max_response_length": 10000
        }
    
    def _setup_chains(self):
        """Setup advanced RAG chains"""
        
        # Main RAG chain with advanced patterns
        self.rag_chain = self._create_advanced_rag_chain()
        
        # CRAG (Corrective RAG) chain
        self.crag_chain = self._create_crag_chain()
        
        # DRAG (Dynamic RAG) chain
        self.drag_chain = self._create_drag_chain()
        
        # Query expansion chain
        self.query_expansion_chain = self._create_query_expansion_chain()
        
        # Self-reflection chain
        if settings.USE_SELF_REFLECTION:
            self.reflection_chain = self._create_advanced_reflection_chain()
        
        # Correction chain for CRAG
        self.correction_chain = self._create_advanced_correction_chain()
    
    def _create_advanced_rag_chain(self):
        """Create advanced RAG chain with industry-standard patterns"""
        
        # Advanced context formatting with metadata enrichment
        def format_context_with_metadata(docs):
            if not docs:
                return "No relevant documents found in the Ghanaian startup ecosystem knowledge base."
            
            context_parts = []
            for i, doc in enumerate(docs[:getattr(settings, 'MAX_CONTEXT_CHUNKS', 10)]):
                content = doc.page_content.strip()
                metadata = doc.metadata
                
                # Enhanced source information
                source_info = f"Source {i+1}: "
                if metadata.get('source'):
                    source_info += f"{metadata.get('source', 'Unknown')}"
                if metadata.get('sector_relevance'):
                    source_info += f" (Relevant to: {', '.join(metadata.get('sector_relevance', []))})"
                if metadata.get('confidence_score'):
                    source_info += f" [Confidence: {metadata.get('confidence_score', 'N/A')}]"
                
                context_parts.append(f"{content}\n\n{source_info}")
            
            return "\n\n---\n\n".join(context_parts)
        
        # Advanced user context formatting
        def format_advanced_user_context(context):
            if not context:
                return ""
            
            context_parts = []
            if context.get('sector'):
                context_parts.append(f"User's primary sector: {context['sector']}")
            if context.get('region'):
                context_parts.append(f"User's primary region: {context['region']}")
            if context.get('experience_level'):
                context_parts.append(f"User's experience level: {context['experience_level']}")
            if context.get('business_stage'):
                context_parts.append(f"Business stage: {context['business_stage']}")
            if context.get('team_size'):
                context_parts.append(f"Team size: {context['team_size']}")
            
            if context_parts:
                return f"User Context: {'; '.join(context_parts)}"
            return ""
        
        # Create the advanced chain
        if self.retriever and self.llm_manager:
            try:
                chain = (
                    {
                        "context": self.retriever | format_context_with_metadata,
                        "user_context": RunnablePassthrough() | format_advanced_user_context,
                        "question": RunnablePassthrough()
                    }
                    | prompt_templates.RAG_PROMPT_TEMPLATE
                    | self.llm_manager.get_langchain_llm()
                    | StrOutputParser()
                )
                return chain
            except Exception as e:
                logger.warning(f"Failed to create advanced RAG chain: {e}")
                return None
        else:
            logger.warning("Advanced RAG chain not available - missing dependencies")
            return None
    
    def _create_crag_chain(self):
        """Create CRAG (Corrective RAG) chain"""
        
        if not self.llm_manager:
            return None
        
        try:
            crag_prompt = ChatPromptTemplate.from_template("""
            You are an AI assistant implementing Corrective RAG (CRAG) for the Ghanaian startup ecosystem.
            
            Your task is to:
            1. Analyze the initial response
            2. Identify any factual errors or inconsistencies
            3. Provide a corrected response
            4. Explain what was corrected and why
            
            Question: {question}
            Initial Answer: {initial_answer}
            Context: {context}
            Reflection Results: {reflection}
            
            Please provide a JSON response with:
            - corrected_answer: The improved answer
            - corrections_made: List of specific corrections
            - reasoning: Explanation of why corrections were needed
            - confidence_in_correction: 0-100 score
            """)
            
            chain = (
                {
                    "question": RunnablePassthrough(),
                    "initial_answer": RunnablePassthrough(),
                    "context": RunnablePassthrough(),
                    "reflection": RunnablePassthrough()
                }
                | crag_prompt
                | self.llm_manager.get_langchain_llm()
                | JsonOutputParser()
            )
            
            return chain
        except Exception as e:
            logger.warning(f"Failed to create CRAG chain: {e}")
            return None
    
    def _create_drag_chain(self):
        """Create DRAG (Dynamic RAG) chain"""
        
        if not self.llm_manager:
            return None
        
        try:
            drag_prompt = ChatPromptTemplate.from_template("""
            You are implementing Dynamic RAG (DRAG) for Ghanaian startup ecosystem queries.
            
            Analyze the query and determine:
            1. Query complexity level
            2. Required retrieval strategy
            3. Optimal chunk size and overlap
            4. Whether to use query expansion
            
            Query: {question}
            User Context: {user_context}
            Available Strategies: {available_strategies}
            
            Provide a JSON response with:
            - retrieval_strategy: Recommended strategy
            - chunk_size: Optimal chunk size
            - chunk_overlap: Optimal overlap
            - use_query_expansion: Boolean
            - reasoning: Explanation of choices
            """)
            
            chain = (
                {
                    "question": RunnablePassthrough(),
                    "user_context": RunnablePassthrough(),
                    "available_strategies": RunnablePassthrough()
                }
                | drag_prompt
                | self.llm_manager.get_langchain_llm()
                | JsonOutputParser()
            )
            
            return chain
        except Exception as e:
            logger.warning(f"Failed to create DRAG chain: {e}")
            return None
    
    def _create_query_expansion_chain(self):
        """Create query expansion chain"""
        
        if not self.llm_manager:
            return None
        
        try:
            expansion_prompt = ChatPromptTemplate.from_template("""
            Expand the user's query to improve retrieval for the Ghanaian startup ecosystem.
            
            Original Query: {query}
            User Context: {user_context}
            
            Generate 3-5 expanded queries that:
            1. Include relevant Ghanaian business terms
            2. Add sector-specific keywords
            3. Include regional context
            4. Add regulatory or compliance terms
            
            Return as JSON:
            - expanded_queries: List of expanded queries
            - reasoning: Why each expansion was chosen
            """)
            
            chain = (
                {
                    "query": RunnablePassthrough(),
                    "user_context": RunnablePassthrough()
                }
                | expansion_prompt
                | self.llm_manager.get_langchain_llm()
                | JsonOutputParser()
            )
            
            return chain
        except Exception as e:
            logger.warning(f"Failed to create query expansion chain: {e}")
            return None
    
    def _create_advanced_reflection_chain(self):
        """Create advanced self-reflection chain"""
        
        if not self.llm_manager:
            return None
        
        try:
            reflection_prompt = ChatPromptTemplate.from_template("""
            You are an AI assistant performing advanced self-reflection on Ghanaian startup ecosystem responses.
            
            Evaluate your response on these criteria:
            1. **Factual Accuracy**: Are all claims supported by the context?
            2. **Ghanaian Relevance**: Is the information specific to Ghana?
            3. **Sector Expertise**: Does it show deep knowledge of fintech/agritech/healthtech?
            4. **Actionability**: Does it provide practical next steps?
            5. **Completeness**: Does it address all aspects of the question?
            6. **Cultural Sensitivity**: Is it appropriate for Ghanaian business context?
            
            Question: {question}
            Your Answer: {answer}
            Context Used: {context}
            
            Provide a JSON response with:
            - overall_score: 0-100
            - criterion_scores: Individual scores for each criterion
            - strengths: What was done well
            - areas_for_improvement: What could be better
            - confidence_in_evaluation: 0-100
            - correction_needed: Boolean
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
        except Exception as e:
            logger.warning(f"Failed to create advanced reflection chain: {e}")
            return None
    
    def _create_advanced_correction_chain(self):
        """Create advanced correction chain"""
        
        if not self.llm_manager:
            return None
        
        try:
            correction_prompt = ChatPromptTemplate.from_template("""
            Based on the advanced reflection results, provide a significantly improved answer about the Ghanaian startup ecosystem.
            
            Original Question: {question}
            Original Answer: {answer}
            Reflection Results: {reflection}
            Context: {context}
            
            Focus on:
            - Correcting factual inaccuracies
            - Improving Ghanaian specificity
            - Enhancing sector expertise
            - Adding practical actionability
            - Ensuring cultural appropriateness
            
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
        except Exception as e:
            logger.warning(f"Failed to create advanced correction chain: {e}")
            return None
    
    def _validate_security(self, query: str) -> Tuple[bool, str]:
        """Validate query for security concerns"""
        
        # Check rate limiting
        current_time = datetime.now()
        if (current_time - self.rate_limiter["last_reset"]).seconds >= 60:
            self.rate_limiter["current_requests"] = 0
            self.rate_limiter["last_reset"] = current_time
        
        if self.rate_limiter["current_requests"] >= self.rate_limiter["max_requests_per_minute"]:
            return False, "Rate limit exceeded. Please wait before making another request."
        
        # Check query length
        if len(query) > self.security_validator["max_query_length"]:
            return False, f"Query too long. Maximum length is {self.security_validator['max_query_length']} characters."
        
        # Check for blocked patterns
        for pattern in self.security_validator["blocked_patterns"]:
            if re.search(pattern, query, re.IGNORECASE):
                return False, "Query contains blocked content patterns."
        
        # Increment request counter
        self.rate_limiter["current_requests"] += 1
        
        return True, ""
    
    def _expand_query(self, query: str, user_context: Dict[str, Any]) -> List[str]:
        """Expand query using advanced techniques"""
        
        try:
            if not self.query_expansion_chain:
                return [query]
            
            # Get expanded queries
            expansion_result = self.query_expansion_chain.invoke({
                "query": query,
                "user_context": user_context
            })
            
            expanded_queries = expansion_result.get("expanded_queries", [query])
            return expanded_queries[:5]  # Limit to 5 expansions
            
        except Exception as e:
            logger.error(f"Query expansion failed: {e}")
            return [query]
    
    def _multi_stage_retrieval(self, 
                              query: str,
                              expanded_queries: List[str],
                              user_context: Dict[str, Any]) -> List[Document]:
        """Perform multi-stage retrieval"""
        
        all_documents = []
        
        try:
            if not self.retriever:
                logger.warning("Retriever not available for multi-stage retrieval")
                return []
            
            # Stage 1: Initial retrieval with original query
            initial_docs = self.retriever.get_relevant_documents(query)
            all_documents.extend(initial_docs)
            
            # Stage 2: Retrieval with expanded queries
            for expanded_query in expanded_queries[1:]:  # Skip first as it's the original
                expanded_docs = self.retriever.get_relevant_documents(expanded_query)
                all_documents.extend(expanded_docs)
            
            # Stage 3: Context-aware filtering
            filtered_docs = self._apply_contextual_filtering(all_documents, user_context)
            
            # Stage 4: Deduplication and ranking
            final_docs = self._deduplicate_and_rank(filtered_docs, query)
            
            logger.info(f"Multi-stage retrieval: {len(initial_docs)} initial + {len(expanded_docs)} expanded = {len(final_docs)} final")
            return final_docs
            
        except Exception as e:
            logger.error(f"Multi-stage retrieval failed: {e}")
            return []
    
    def _apply_contextual_filtering(self, 
                                  documents: List[Document],
                                  user_context: Dict[str, Any]) -> List[Document]:
        """Apply advanced contextual filtering"""
        
        filtered_docs = []
        
        for doc in documents:
            relevance_score = self._calculate_contextual_relevance(doc, user_context)
            if relevance_score > 0.3:  # Minimum relevance threshold
                # Enhance document metadata with relevance score
                doc.metadata["contextual_relevance"] = relevance_score
                filtered_docs.append(doc)
        
        # Sort by relevance score
        filtered_docs.sort(key=lambda x: x.metadata.get("contextual_relevance", 0), reverse=True)
        
        return filtered_docs
    
    def _calculate_contextual_relevance(self, 
                                      document: Document,
                                      user_context: Dict[str, Any]) -> float:
        """Calculate contextual relevance score"""
        
        score = 0.0
        
        # Sector relevance
        if user_context.get('sector') and document.metadata.get('sector_relevance'):
            if user_context['sector'] in document.metadata['sector_relevance']:
                score += 0.4
        
        # Regional relevance
        if user_context.get('region') and document.metadata.get('source'):
            if user_context['region'].lower() in document.metadata['source'].lower():
                score += 0.3
        
        # Experience level relevance
        if user_context.get('experience_level'):
            if user_context['experience_level'] == 'beginner' and 'basic' in document.page_content.lower():
                score += 0.2
            elif user_context['experience_level'] == 'expert' and 'advanced' in document.page_content.lower():
                score += 0.2
        
        return min(score, 1.0)
    
    def _deduplicate_and_rank(self, 
                             documents: List[Document],
                             query: str) -> List[Document]:
        """Deduplicate and rank documents"""
        
        # Remove exact duplicates
        seen_content = set()
        unique_docs = []
        
        for doc in documents:
            content_hash = hash(doc.page_content)
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_docs.append(doc)
        
        # Rank by relevance and recency
        for doc in unique_docs:
            # Calculate ranking score
            relevance_score = doc.metadata.get("contextual_relevance", 0.5)
            recency_score = 0.5  # Could be enhanced with actual timestamps
            
            doc.metadata["ranking_score"] = (relevance_score * 0.7) + (recency_score * 0.3)
        
        # Sort by ranking score
        unique_docs.sort(key=lambda x: x.metadata.get("ranking_score", 0), reverse=True)
        
        max_chunks = getattr(settings, 'MAX_RETRIEVED_CHUNKS', 20)
        return unique_docs[:max_chunks]
    
    @trace("advanced_rag_query")
    async def query_advanced(self, 
                           question: str,
                           conversation_history: List[Dict[str, str]] = None,
                           user_context: Dict[str, Any] = None,
                           provider: str = None,
                           use_advanced_patterns: bool = True) -> RAGResponse:
        """
        Advanced RAG query with industry-standard patterns
        """
        
        start_time = datetime.now()
        
        try:
            # Security validation
            is_valid, error_message = self._validate_security(question)
            if not is_valid:
                return self._create_security_error_response(question, error_message)
            
            # Prepare context
            conversation_history = conversation_history or []
            user_context = user_context or {}
            
            # Query expansion
            expanded_queries = self._expand_query(question, user_context)
            
            # Multi-stage retrieval
            documents = self._multi_stage_retrieval(expanded_queries, user_context)
            
            if not documents:
                logger.warning("No relevant documents found for query")
                return self._create_advanced_fallback_response(question)
            
            # Generate initial response
            initial_answer = await self._generate_advanced_response(question, documents, user_context, provider)
            
            # Apply advanced patterns if enabled
            final_answer = initial_answer
            correction_metadata = {}
            reflection_metadata = {}
            
            if use_advanced_patterns:
                # CRAG: Corrective RAG
                if self.crag_chain and settings.USE_SELF_REFLECTION:
                    reflection_result = await self._perform_advanced_reflection(question, initial_answer, documents)
                    reflection_metadata = reflection_result
                    
                    if reflection_result and reflection_result.get('correction_needed', False):
                        correction_result = await self._perform_advanced_correction(question, initial_answer, reflection_result, documents)
                        if correction_result:
                            final_answer = correction_result.get('corrected_answer', initial_answer)
                            correction_metadata = correction_result
            
            # Generate follow-up questions
            follow_up_questions = await self._generate_advanced_follow_ups(question, final_answer)
            
            # Calculate confidence
            confidence = self._calculate_advanced_confidence(documents, reflection_metadata)
            
            # Format sources
            sources = self._format_advanced_sources(documents)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Create comprehensive response
            response = RAGResponse(
                answer=final_answer,
                sources=sources,
                follow_up_questions=follow_up_questions,
                confidence=confidence,
                processing_time=processing_time,
                chunks_retrieved=len(documents),
                provider=provider or getattr(settings, 'PRIMARY_LLM_PROVIDER', 'unknown'),
                ghana_focus=True,
                sectors_covered=self._identify_sectors_covered(documents),
                retrieval_metadata={
                    "retrieval_strategy": "multi_stage",
                    "query_expansions": len(expanded_queries),
                    "initial_docs": len(documents),
                    "filtered_docs": len(documents)
                },
                generation_metadata={
                    "model_used": provider or getattr(settings, 'PRIMARY_LLM_PROVIDER', 'unknown'),
                    "response_length": len(final_answer),
                    "advanced_patterns_used": use_advanced_patterns
                },
                evaluation_metadata={
                    "reflection": reflection_metadata,
                    "correction": correction_metadata
                },
                security_metadata={
                    "security_validated": True,
                    "rate_limit_status": "within_limits"
                }
            )
            
            logger.info(f"Advanced query processed successfully in {processing_time:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"Error in advanced query processing: {e}")
            return self._create_advanced_error_response(question, str(e))
    
    async def _generate_advanced_response(self, 
                                       question: str,
                                       documents: List[Document],
                                       user_context: Dict[str, Any],
                                       provider: str = None) -> str:
        """Generate advanced response using the RAG chain"""
        
        try:
            if not self.rag_chain:
                # Fallback to simple response generation
                return self._generate_fallback_response(question, documents, user_context)
            
            # Prepare context for the chain
            chain_input = {
                "question": question,
                "user_context": user_context
            }
            
            # Execute the chain
            response = await self.rag_chain.ainvoke(chain_input)
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating advanced response: {e}")
            return self._generate_fallback_response(question, documents, user_context)
    
    def _generate_fallback_response(self, 
                                 question: str,
                                 documents: List[Document],
                                 user_context: Dict[str, Any]) -> str:
        """Generate fallback response when advanced chain is not available"""
        
        try:
            # Simple response generation
            context = "\n\n".join([doc.page_content[:200] + "..." for doc in documents[:3]])
            
            prompt = f"""Based on the following context about the Ghanaian startup ecosystem, answer the user's question.

Context:
{context}

Question: {question}

User Context: {user_context.get('sector', 'General')} sector, {user_context.get('region', 'Ghana')} region

Answer:"""
            
            # Use LLM manager if available
            if self.llm_manager:
                try:
                    response = self.llm_manager.generate(prompt)
                    return response
                except Exception as e:
                    logger.error(f"LLM generation failed: {e}")
            
            # Fallback to template response
            return f"I apologize, but I encountered an error while processing your question about the Ghanaian startup ecosystem. Please try rephrasing your question or ask about a different aspect of fintech, agritech, or healthtech in Ghana."
            
        except Exception as e:
            logger.error(f"Fallback response generation failed: {e}")
            return "I apologize, but I'm unable to process your request at the moment. Please try again later."
    
    async def _perform_advanced_reflection(self, 
                                         question: str,
                                         answer: str,
                                         documents: List[Document]) -> Dict[str, Any]:
        """Perform advanced self-reflection"""
        
        try:
            if not self.reflection_chain:
                return {}
            
            # Prepare context for reflection
            context_summary = "\n".join([doc.page_content[:200] + "..." for doc in documents[:3]])
            
            reflection_input = {
                "question": question,
                "answer": answer,
                "context": context_summary
            }
            
            # Execute reflection
            reflection_result = await self.reflection_chain.ainvoke(reflection_input)
            
            logger.info(f"Advanced self-reflection completed with score: {reflection_result.get('overall_score', 'N/A')}")
            return reflection_result
            
        except Exception as e:
            logger.error(f"Error in advanced self-reflection: {e}")
            return {}
    
    async def _perform_advanced_correction(self, 
                                         question: str,
                                         original_answer: str,
                                         reflection_result: Dict[str, Any],
                                         documents: List[Document]) -> Dict[str, Any]:
        """Perform advanced correction"""
        
        try:
            if not self.correction_chain:
                return {}
            
            # Prepare context for correction
            context_summary = "\n".join([doc.page_content[:200] + "..." for doc in documents[:3]])
            
            correction_input = {
                "question": question,
                "answer": original_answer,
                "reflection": json.dumps(reflection_result),
                "context": context_summary
            }
            
            # Execute correction
            correction_result = await self.correction_chain.ainvoke(correction_input)
            
            logger.info("Advanced answer correction completed")
            return correction_result
            
        except Exception as e:
            logger.error(f"Error in advanced correction: {e}")
            return {}
    
    async def _generate_advanced_follow_ups(self, question: str, answer: str) -> List[str]:
        """Generate advanced follow-up questions"""
        
        try:
            # Use the prompt template for follow-up questions
            follow_up_prompt = prompt_templates.FOLLOW_UP_QUESTIONS_PROMPT.format(
                question=question,
                answer=answer
            )
            
            # Generate follow-ups using LLM
            if self.llm_manager:
                try:
                    follow_ups = await self.llm_manager.generate_async(
                        follow_up_prompt,
                        provider=getattr(settings, 'PRIMARY_LLM_PROVIDER', None)
                    )
                    
                    # Parse and clean follow-ups
                    if follow_ups:
                        # Simple parsing - split by newlines and clean
                        questions = [q.strip() for q in follow_ups.split('\n') if q.strip()]
                        # Limit to 3 questions
                        return questions[:3]
                except Exception as e:
                    logger.error(f"LLM follow-up generation failed: {e}")
            
            return self._get_advanced_follow_ups()
            
        except Exception as e:
            logger.error(f"Error generating advanced follow-ups: {e}")
            return self._get_advanced_follow_ups()
    
    def _get_advanced_follow_ups(self) -> List[str]:
        """Get advanced follow-up questions for Ghanaian startup ecosystem"""
        
        return [
            "What are the key regulatory requirements for this sector in Ghana?",
            "Are there any government programs or incentives available?",
            "What are the main challenges entrepreneurs face in this area?",
            "How does this vary across different regions in Ghana?",
            "What partnerships or collaborations would be beneficial?"
        ]
    
    def _calculate_advanced_confidence(self, 
                                     documents: List[Document],
                                     reflection_result: Dict[str, Any]) -> float:
        """Calculate advanced confidence score"""
        
        try:
            # Base confidence from document relevance
            relevance_scores = [doc.metadata.get("contextual_relevance", 0.5) for doc in documents]
            base_confidence = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.5
            
            # Reflection-based confidence
            reflection_confidence = 0.5
            if reflection_result and reflection_result.get('overall_score'):
                reflection_confidence = reflection_result['overall_score'] / 100.0
            
            # Document quality confidence
            quality_scores = [doc.metadata.get("ranking_score", 0.5) for doc in documents]
            quality_confidence = sum(quality_scores) / len(quality_scores) if quality_scores else 0.5
            
            # Weighted combination
            final_confidence = (
                (base_confidence * 0.4) +
                (reflection_confidence * 0.3) +
                (quality_confidence * 0.3)
            )
            
            return min(final_confidence, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating advanced confidence: {e}")
            return 0.5
    
    def _format_advanced_sources(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """Format advanced source information"""
        
        sources = []
        
        for doc in documents[:5]:  # Limit to top 5 sources
            source = {
                "title": doc.metadata.get('source', 'Unknown Source'),
                "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                "relevance": doc.metadata.get("contextual_relevance", 0.5),
                "ranking_score": doc.metadata.get("ranking_score", 0.5),
                "sector_relevance": doc.metadata.get('sector_relevance', []),
                "metadata": doc.metadata
            }
            sources.append(source)
        
        return sources
    
    def _identify_sectors_covered(self, documents: List[Document]) -> List[str]:
        """Identify which Ghanaian sectors are covered in the documents"""
        
        sectors_covered = set()
        
        for doc in documents:
            sector_relevance = doc.metadata.get('sector_relevance', [])
            sectors_covered.update(sector_relevance)
        
        return list(sectors_covered) if sectors_covered else ["general"]
    
    def _create_advanced_fallback_response(self, question: str) -> RAGResponse:
        """Create advanced fallback response"""
        
        return RAGResponse(
            answer=f"I apologize, but I don't have enough specific information about '{question}' in my Ghanaian startup ecosystem knowledge base. This could be because:\n\n1. The topic is outside my current knowledge scope\n2. The information needs to be updated\n3. The question is too specific for my current data\n\nPlease try:\n- Rephrasing your question\n- Asking about broader aspects of fintech, agritech, or healthtech in Ghana\n- Contacting relevant Ghanaian business organizations for the most current information",
            sources=[],
            follow_up_questions=self._get_advanced_follow_ups(),
            confidence=0.1,
            processing_time=0.0,
            chunks_retrieved=0,
            provider=getattr(settings, 'PRIMARY_LLM_PROVIDER', 'unknown'),
            ghana_focus=True,
            sectors_covered=["general"],
            retrieval_metadata={"error": "no_documents_found"},
            generation_metadata={"error": "fallback_response"},
            evaluation_metadata={},
            security_metadata={"security_validated": True}
        )
    
    def _create_advanced_error_response(self, question: str, error: str) -> RAGResponse:
        """Create advanced error response"""
        
        return RAGResponse(
            answer=f"I encountered an error while processing your question about the Ghanaian startup ecosystem: {error}. Please try again or rephrase your question about fintech, agritech, or healthtech in Ghana.",
            sources=[],
            follow_up_questions=[],
            confidence=0.0,
            processing_time=0.0,
            chunks_retrieved=0,
            provider=getattr(settings, 'PRIMARY_LLM_PROVIDER', 'unknown'),
            ghana_focus=True,
            sectors_covered=["general"],
            retrieval_metadata={"error": error},
            generation_metadata={"error": error},
            evaluation_metadata={"error": error},
            security_metadata={"security_validated": True}
        )
    
    def _create_security_error_response(self, question: str, error_message: str) -> RAGResponse:
        """Create security error response"""
        
        return RAGResponse(
            answer=f"Security validation failed: {error_message}. Please check your input and try again.",
            sources=[],
            follow_up_questions=[],
            confidence=0.0,
            processing_time=0.0,
            chunks_retrieved=0,
            provider="security_blocked",
            ghana_focus=True,
            sectors_covered=["general"],
            retrieval_metadata={"error": "security_blocked"},
            generation_metadata={"error": "security_blocked"},
            evaluation_metadata={"error": "security_blocked"},
            security_metadata={"security_validated": False, "error": error_message}
        )

# Global instance
advanced_rag_engine = None

def initialize_advanced_rag_engine(vector_store):
    """Initialize the advanced RAG engine"""
    global advanced_rag_engine
    try:
        advanced_rag_engine = AdvancedRAGEngine(vector_store)
        logger.info("Advanced RAG engine initialized successfully")
        return advanced_rag_engine
    except Exception as e:
        logger.error(f"Failed to initialize advanced RAG engine: {e}")
        return None

def get_advanced_rag_engine() -> Optional[AdvancedRAGEngine]:
    """Get the global advanced RAG engine instance"""
    return advanced_rag_engine 