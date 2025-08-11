import os
import logging
from typing import Dict, Any, List, Optional, Literal
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

class SettingsValidationError(Exception):
    """Custom exception for settings validation errors"""
    pass

class ModernRAGSettings:
    """
    Modern RAG Configuration with multi-provider support and advanced features
    """
    
    def __init__(self):
        """Initialize settings with validation"""
        try:
            self._load_api_keys()
            self._load_llm_config()
            self._load_embedding_config()
            self._load_vector_store_config()
            self._load_retrieval_config()
            self._load_evaluation_config()
            self._load_business_context()
            self._load_ui_config()
            self._load_performance_config()
            self._load_integration_config()
            
            self.validate_config()
            logger.info("Modern RAG settings loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize settings: {e}")
            raise SettingsValidationError(f"Configuration error: {e}")
    
    def _load_api_keys(self):
        """Load API keys for multiple providers"""
        # Primary LLM Provider
        self.PRIMARY_LLM_PROVIDER = os.getenv("PRIMARY_LLM_PROVIDER", "openai")
        
        # LLM Provider Keys
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
        self.GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        self.COHERE_API_KEY = os.getenv("COHERE_API_KEY")
        
        # Embedding Provider Keys
        self.PRIMARY_EMBEDDING_PROVIDER = os.getenv("PRIMARY_EMBEDDING_PROVIDER", "sentence-transformers")
        
        # Vector Store Keys
        self.PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        self.PINECONE_ENV = os.getenv("PINECONE_ENV")
        
        # Evaluation & Monitoring
        self.LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
        self.LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
        self.LANGFUSE_HOST = os.getenv("LANGFUSE_HOST")
        
        # Validate at least one LLM provider
        available_providers = []
        if self.OPENAI_API_KEY:
            available_providers.append("openai")
        if self.ANTHROPIC_API_KEY:
            available_providers.append("anthropic")
        if self.GOOGLE_API_KEY:
            available_providers.append("google")
        
        if not available_providers:
            logger.warning("No LLM API keys configured - some features will be disabled")
        else:
            logger.info(f"Available LLM providers: {available_providers}")
    
    def _load_llm_config(self):
        """Load LLM provider configurations"""
        self.LLM_CONFIGS = {
            "openai": {
                "model": os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
                "temperature": float(os.getenv("OPENAI_TEMPERATURE", "0.3")),
                "max_tokens": int(os.getenv("OPENAI_MAX_TOKENS", "1000")),
                "fallback_model": "gpt-3.5-turbo"
            },
            "anthropic": {
                "model": os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307"),
                "temperature": float(os.getenv("ANTHROPIC_TEMPERATURE", "0.3")),
                "max_tokens": int(os.getenv("ANTHROPIC_MAX_TOKENS", "1000")),
                "fallback_model": "claude-3-haiku-20240307"
            },
            "google": {
                "model": os.getenv("GOOGLE_MODEL", "gemini-pro"),
                "temperature": float(os.getenv("GOOGLE_TEMPERATURE", "0.3")),
                "max_tokens": int(os.getenv("GOOGLE_MAX_TOKENS", "1000")),
                "fallback_model": "gemini-pro"
            },
            "ollama": {
                "model": os.getenv("OLLAMA_MODEL", "llama2"),
                "temperature": float(os.getenv("OLLAMA_TEMPERATURE", "0.3")),
                "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                "fallback_model": "llama2"
            }
        }
    
    def _load_embedding_config(self):
        """Load embedding provider configurations"""
        self.EMBEDDING_CONFIGS = {
            "openai": {
                "model": os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
                "dimensions": int(os.getenv("OPENAI_EMBEDDING_DIMS", "1536"))
            },
            "sentence-transformers": {
                "model": os.getenv("ST_EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5"),
                "dimensions": int(os.getenv("ST_EMBEDDING_DIMS", "1024")),
                "device": os.getenv("ST_DEVICE", "cpu")
            },
            "google": {
                "model": os.getenv("GOOGLE_EMBEDDING_MODEL", "models/embedding-001"),
                "dimensions": int(os.getenv("GOOGLE_EMBEDDING_DIMS", "768"))
            },
            "cohere": {
                "model": os.getenv("COHERE_EMBEDDING_MODEL", "embed-english-v3.0"),
                "dimensions": int(os.getenv("COHERE_EMBEDDING_DIMS", "1024"))
            }
        }
    
    def _load_vector_store_config(self):
        """Load vector store configurations"""
        self.VECTOR_STORE_TYPE = os.getenv("VECTOR_STORE_TYPE", "chroma")
        
        # ChromaDB Configuration
        self.CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "./vector_db")
        self.COLLECTION_NAME = os.getenv("COLLECTION_NAME", "african_business_knowledge")
        
        # Pinecone Configuration
        self.PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "ubuntu-ai")
        
        # FAISS Configuration
        self.FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "./vector_db/faiss_index")
        
        os.makedirs(self.CHROMA_PERSIST_DIRECTORY, exist_ok=True)
    
    def _load_retrieval_config(self):
        """Load advanced retrieval configurations"""
        # Retrieval Strategy
        self.RETRIEVAL_STRATEGY = os.getenv("RETRIEVAL_STRATEGY", "hybrid")  # hybrid, semantic, keyword
        
        # Similarity Search
        self.SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.3"))
        self.MAX_RETRIEVED_CHUNKS = int(os.getenv("MAX_RETRIEVED_CHUNKS", "20"))
        
        # Hybrid Search Weights
        self.SEMANTIC_WEIGHT = float(os.getenv("SEMANTIC_WEIGHT", "0.7"))
        self.KEYWORD_WEIGHT = float(os.getenv("KEYWORD_WEIGHT", "0.3"))
        
        # Re-ranking
        self.USE_RERANKING = os.getenv("USE_RERANKING", "true").lower() == "true"
        self.RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-2-v2")
        self.RERANKED_TOP_K = int(os.getenv("RERANKED_TOP_K", "5"))
        
        # Chunking Strategy
        self.CHUNKING_STRATEGY = os.getenv("CHUNKING_STRATEGY", "recursive")  # recursive, semantic, adaptive
        self.CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
        self.CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
        
        # Context Configuration
        self.CONTEXT_WINDOW = int(os.getenv("CONTEXT_WINDOW", "4000"))
        self.MAX_CONTEXT_CHUNKS = int(os.getenv("MAX_CONTEXT_CHUNKS", "10"))
    
    def _load_evaluation_config(self):
        """Load evaluation and monitoring configurations"""
        # RAGAS Evaluation
        self.USE_RAGAS_EVALUATION = os.getenv("USE_RAGAS_EVALUATION", "true").lower() == "true"
        self.RAGAS_METRICS = os.getenv("RAGAS_METRICS", "faithfulness,answer_relevancy,context_precision").split(",")
        
        # LangFuse Monitoring
        self.USE_LANGFUSE = os.getenv("USE_LANGFUSE", "false").lower() == "true"
        
        # Self-Reflection
        self.USE_SELF_REFLECTION = os.getenv("USE_SELF_REFLECTION", "true").lower() == "true"
        self.REFLECTION_MODEL = os.getenv("REFLECTION_MODEL", "auto")  # auto uses primary provider
        
        # Evaluation Dataset
        self.EVAL_DATASET_PATH = os.getenv("EVAL_DATASET_PATH", "./data/eval_dataset.json")
    
    def _load_business_context(self):
        """Load business context data"""
        self.AFRICAN_COUNTRIES = [
            "Nigeria", "Kenya", "South Africa", "Ghana", "Egypt", "Morocco",
            "Tunisia", "Uganda", "Tanzania", "Rwanda", "Senegal", "Ivory Coast",
            "Ethiopia", "Cameroon", "Botswana", "Zambia", "Zimbabwe", "Malawi",
            "Mozambique", "Madagascar", "Mali", "Burkina Faso", "Guinea",
            "Benin", "Togo", "Liberia", "Sierra Leone", "Chad", "Niger",
            "Central African Republic", "Democratic Republic of Congo",
            "Republic of Congo", "Gabon", "Equatorial Guinea", "São Tomé and Príncipe",
            "Cape Verde", "Comoros", "Mauritius", "Seychelles", "Djibouti",
            "Eritrea", "Somalia", "South Sudan", "Sudan", "Libya", "Algeria",
            "Western Sahara", "Mauritania", "Gambia", "Guinea-Bissau",
            "Burundi", "Lesotho", "Swaziland", "Angola", "Namibia"
        ]
        
        self.BUSINESS_SECTORS = [
            "Fintech", "Agritech", "Healthtech", "Edtech", "E-commerce",
            "Logistics", "Energy", "Clean Technology", "Manufacturing",
            "Real Estate", "Tourism", "Media", "Telecommunications",
            "Retail", "Food & Beverage", "Transportation", "Construction",
            "Mining", "Oil & Gas", "Water", "Waste Management"
        ]
        
        self.FUNDING_STAGES = [
            "Idea", "Pre-seed", "Seed", "Series A", "Series B", "Series C",
            "Growth", "Bridge", "Mezzanine", "IPO", "Grant", "Government"
        ]
    
    def _load_ui_config(self):
        """Load UI configuration"""
        self.APP_TITLE = os.getenv("APP_TITLE", "UbuntuAI - Modern African Business Intelligence")
        self.APP_DESCRIPTION = os.getenv("APP_DESCRIPTION", "Next-generation AI assistant for African entrepreneurship")
        self.CHAT_PLACEHOLDER = os.getenv("CHAT_PLACEHOLDER", "Ask me anything about African business...")
    
    def _load_performance_config(self):
        """Load performance configurations"""
        # Caching
        self.CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))
        self.USE_SEMANTIC_CACHE = os.getenv("USE_SEMANTIC_CACHE", "true").lower() == "true"
        
        # Concurrency
        self.MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "10"))
        self.REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))
        
        # Batch Processing
        self.BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
        self.ENABLE_STREAMING = os.getenv("ENABLE_STREAMING", "true").lower() == "true"
    
    def _load_integration_config(self):
        """Load integration configurations"""
        # WhatsApp/Twilio
        self.TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
        self.TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
        self.TWILIO_WHATSAPP_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER")
        
        # Mobile Optimization
        self.MOBILE_MAX_CONTEXT = int(os.getenv("MOBILE_MAX_CONTEXT", "2000"))
        self.MOBILE_MAX_RESPONSE = int(os.getenv("MOBILE_MAX_RESPONSE", "500"))
        
        # Agent Configuration
        self.USE_LANGCHAIN_AGENTS = os.getenv("USE_LANGCHAIN_AGENTS", "true").lower() == "true"
        self.AGENT_MAX_ITERATIONS = int(os.getenv("AGENT_MAX_ITERATIONS", "5"))
        self.AGENT_TIMEOUT = int(os.getenv("AGENT_TIMEOUT", "120"))
    
    def get_llm_config(self, provider: str = None) -> Dict[str, Any]:
        """Get LLM configuration for specified provider"""
        provider = provider or self.PRIMARY_LLM_PROVIDER
        return self.LLM_CONFIGS.get(provider, self.LLM_CONFIGS["openai"])
    
    def get_embedding_config(self, provider: str = None) -> Dict[str, Any]:
        """Get embedding configuration for specified provider"""
        provider = provider or self.PRIMARY_EMBEDDING_PROVIDER
        return self.EMBEDDING_CONFIGS.get(provider, self.EMBEDDING_CONFIGS["sentence-transformers"])
    
    def get_available_llm_providers(self) -> List[str]:
        """Get list of available LLM providers based on API keys"""
        providers = []
        if self.OPENAI_API_KEY:
            providers.append("openai")
        if self.ANTHROPIC_API_KEY:
            providers.append("anthropic")
        if self.GOOGLE_API_KEY:
            providers.append("google")
        # Ollama is always available if running locally
        providers.append("ollama")
        return providers
    
    def get_available_embedding_providers(self) -> List[str]:
        """Get list of available embedding providers"""
        providers = ["sentence-transformers"]  # Always available
        if self.OPENAI_API_KEY:
            providers.append("openai")
        if self.GOOGLE_API_KEY:
            providers.append("google")
        if self.COHERE_API_KEY:
            providers.append("cohere")
        return providers
    
    def validate_config(self) -> bool:
        """Validate configuration settings"""
        errors = []
        
        # Check at least one LLM provider
        if not self.get_available_llm_providers():
            errors.append("No LLM API keys configured")
        
        # Validate numeric ranges
        if not 0 <= self.SIMILARITY_THRESHOLD <= 1:
            errors.append("SIMILARITY_THRESHOLD must be between 0 and 1")
        
        if not 0 <= self.SEMANTIC_WEIGHT <= 1:
            errors.append("SEMANTIC_WEIGHT must be between 0 and 1")
        
        if abs(self.SEMANTIC_WEIGHT + self.KEYWORD_WEIGHT - 1.0) > 0.01:
            errors.append("SEMANTIC_WEIGHT + KEYWORD_WEIGHT must equal 1.0")
        
        # Validate paths
        try:
            os.makedirs(self.CHROMA_PERSIST_DIRECTORY, exist_ok=True)
        except (OSError, PermissionError) as e:
            errors.append(f"Cannot create vector database directory: {e}")
        
        if errors:
            raise SettingsValidationError("; ".join(errors))
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Export settings as dictionary (excluding sensitive data)"""
        return {
            "app_title": self.APP_TITLE,
            "primary_llm_provider": self.PRIMARY_LLM_PROVIDER,
            "primary_embedding_provider": self.PRIMARY_EMBEDDING_PROVIDER,
            "vector_store_type": self.VECTOR_STORE_TYPE,
            "retrieval_strategy": self.RETRIEVAL_STRATEGY,
            "chunking_strategy": self.CHUNKING_STRATEGY,
            "use_reranking": self.USE_RERANKING,
            "use_ragas_evaluation": self.USE_RAGAS_EVALUATION,
            "available_llm_providers": self.get_available_llm_providers(),
            "available_embedding_providers": self.get_available_embedding_providers(),
            "supported_countries": len(self.AFRICAN_COUNTRIES),
            "supported_sectors": len(self.BUSINESS_SECTORS)
        }

# Global settings instance
try:
    settings = ModernRAGSettings()
except SettingsValidationError as e:
    logger.error(f"Failed to load settings: {e}")
    # Create minimal fallback
    settings = type('FallbackSettings', (), {
        'APP_TITLE': 'UbuntuAI - Configuration Error',
        'validate_config': lambda: False,
        'get_available_llm_providers': lambda: [],
        'PRIMARY_LLM_PROVIDER': 'openai'
    })()
    logger.warning("Using fallback settings due to configuration errors")