import os
import logging
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

class SettingsValidationError(Exception):
    """Custom exception for settings validation errors"""
    pass

class Settings:
    """
    Application configuration settings with validation and error handling.
    Loads from environment variables with sensible defaults.
    """
    
    def __init__(self):
        """Initialize settings with validation"""
        try:
            self._load_api_keys()
            self._load_database_config()
            self._load_model_config()
            self._load_business_context()
            self._load_ui_config()
            self._load_performance_config()
            self._load_integration_config()
            
            # Validate critical settings
            self.validate_config()
            logger.info("Settings loaded and validated successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize settings: {e}")
            raise SettingsValidationError(f"Configuration error: {e}")
    
    def _load_api_keys(self):
        """Load and validate API keys"""
        # Use Google Gemini API exclusively
        self.GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        self.PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        self.PINECONE_ENV = os.getenv("PINECONE_ENV")
        
        if not self.GOOGLE_API_KEY:
            logger.warning("GOOGLE_API_KEY not set - AI features will be disabled")
    
    def _load_database_config(self):
        """Load vector database configuration"""
        self.CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "./vector_db")
        self.COLLECTION_NAME = os.getenv("COLLECTION_NAME", "african_business_knowledge")
        
        # Ensure directory exists
        os.makedirs(self.CHROMA_PERSIST_DIRECTORY, exist_ok=True)
    
    def _load_model_config(self):
        """Load AI model configuration"""
        # Gemini embedding model configuration
        self.EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
        self.EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "3072"))  # Gemini default
        self.EMBEDDING_TASK_TYPE = os.getenv("EMBEDDING_TASK_TYPE", "RETRIEVAL_DOCUMENT")
        
        # Text processing configuration
        self.CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1024"))
        self.CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
        self.MAX_INPUT_TOKENS = int(os.getenv("MAX_INPUT_TOKENS", "2048"))  # Gemini embedding limit
        
        # RAG Configuration
        self.SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.3"))
        self.MAX_RETRIEVED_CHUNKS = int(os.getenv("MAX_RETRIEVED_CHUNKS", "10"))
        self.CONTEXT_WINDOW = int(os.getenv("CONTEXT_WINDOW", "4000"))
        
        # Gemini LLM configuration
        self.LLM_MODEL = os.getenv("LLM_MODEL", "gemini-1.5-pro")
        self.TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))
        self.MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1000"))
        self.TOP_P = float(os.getenv("TOP_P", "0.9"))
        self.TOP_K = int(os.getenv("TOP_K", "40"))
        
        # Validate ranges
        if not 0 <= self.SIMILARITY_THRESHOLD <= 1:
            raise SettingsValidationError("SIMILARITY_THRESHOLD must be between 0 and 1")
        if not 0 <= self.TEMPERATURE <= 2:
            raise SettingsValidationError("TEMPERATURE must be between 0 and 2")
        if not 0 <= self.TOP_P <= 1:
            raise SettingsValidationError("TOP_P must be between 0 and 1")
    
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
        
        self.GHANA_REGIONS = [
            "Greater Accra", "Ashanti", "Northern", "Western", "Eastern",
            "Volta", "Central", "Upper East", "Upper West", "Brong-Ahafo",
            "Western North", "Savannah", "North East", "Ahafo", "Bono", "Oti"
        ]
        
        self.GHANA_MAJOR_CITIES = [
            "Accra", "Kumasi", "Tamale", "Takoradi", "Cape Coast", "Sunyani",
            "Ho", "Koforidua", "Wa", "Bolgatanga", "Tema", "Techiman"
        ]
        
        self.BUSINESS_SECTORS = [
            "Fintech", "Agritech", "Healthtech", "Edtech", "E-commerce",
            "Logistics", "Energy", "Clean Technology", "Manufacturing",
            "Real Estate", "Tourism", "Media", "Telecommunications",
            "Retail", "Food & Beverage", "Transportation", "Construction",
            "Mining", "Oil & Gas", "Water", "Waste Management"
        ]
        
        self.FUNDING_STAGES = [
            "Pre-seed", "Seed", "Series A", "Series B", "Series C",
            "Growth", "Bridge", "Mezzanine", "IPO", "Grant", "Government"
        ]
    
    def _load_ui_config(self):
        """Load UI configuration"""
        self.APP_TITLE = os.getenv("APP_TITLE", "UbuntuAI - African Business Intelligence")
        self.APP_DESCRIPTION = os.getenv("APP_DESCRIPTION", "Your AI-powered guide to African entrepreneurship")
        self.CHAT_PLACEHOLDER = os.getenv("CHAT_PLACEHOLDER", "Ask me anything about African business, funding, regulations, or market insights...")
    
    def _load_performance_config(self):
        """Load performance and caching configuration"""
        self.CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))
        self.MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "10"))
        self.REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))
        
        if self.MAX_CONCURRENT_REQUESTS <= 0:
            raise SettingsValidationError("MAX_CONCURRENT_REQUESTS must be positive")
    
    def _load_integration_config(self):
        """Load integration configurations"""
        # WhatsApp/Twilio Configuration
        self.TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
        self.TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
        self.TWILIO_WHATSAPP_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER")
        
        # Agent Configuration
        self.USE_LANGCHAIN_AGENTS = os.getenv("USE_LANGCHAIN_AGENTS", "true").lower() == "true"
        self.AGENT_MAX_ITERATIONS = int(os.getenv("AGENT_MAX_ITERATIONS", "5"))
        self.AGENT_TIMEOUT = int(os.getenv("AGENT_TIMEOUT", "120"))
        
        # WhatsApp optimization settings
        self.WHATSAPP_MAX_MESSAGE_LENGTH = int(os.getenv("WHATSAPP_MAX_MESSAGE_LENGTH", "1600"))
        self.WHATSAPP_SESSION_TIMEOUT = int(os.getenv("WHATSAPP_SESSION_TIMEOUT", "86400"))  # 24 hours
        self.WHATSAPP_MAX_SESSIONS = int(os.getenv("WHATSAPP_MAX_SESSIONS", "1000"))
    
    def get_gemini_config(self) -> Dict[str, Any]:
        """Get Gemini model configuration"""
        return {
            "model": self.LLM_MODEL,
            "temperature": self.TEMPERATURE,
            "top_p": self.TOP_P,
            "top_k": self.TOP_K,
            "max_output_tokens": self.MAX_TOKENS
        }
    
    def get_mobile_gemini_config(self) -> Dict[str, Any]:
        """Get optimized Gemini config for mobile/WhatsApp users"""
        return {
            "model": "gemini-1.5-flash",  # Faster model for mobile
            "temperature": 0.3,
            "top_p": 0.9,
            "top_k": 40,
            "max_output_tokens": min(500, self.MAX_TOKENS)
        }
    
    def get_embedding_config(self) -> Dict[str, Any]:
        """Get embedding configuration"""
        return {
            "model": self.EMBEDDING_MODEL,
            "dimensions": self.EMBEDDING_DIMENSIONS,
            "task_type": self.EMBEDDING_TASK_TYPE,
            "max_input_tokens": self.MAX_INPUT_TOKENS
        }
    
    def get_chunking_config(self) -> Dict[str, Any]:
        """Get text chunking configuration"""
        return {
            "chunk_size": self.CHUNK_SIZE,
            "chunk_overlap": self.CHUNK_OVERLAP,
            "max_input_tokens": self.MAX_INPUT_TOKENS
        }
    
    def get_whatsapp_config(self) -> Optional[Dict[str, Any]]:
        """Get WhatsApp integration configuration if available"""
        if not all([self.TWILIO_ACCOUNT_SID, self.TWILIO_AUTH_TOKEN, self.TWILIO_WHATSAPP_NUMBER]):
            return None
            
        return {
            "account_sid": self.TWILIO_ACCOUNT_SID,
            "auth_token": self.TWILIO_AUTH_TOKEN,
            "whatsapp_number": self.TWILIO_WHATSAPP_NUMBER,
            "max_message_length": self.WHATSAPP_MAX_MESSAGE_LENGTH,
            "session_timeout": self.WHATSAPP_SESSION_TIMEOUT,
            "max_sessions": self.WHATSAPP_MAX_SESSIONS
        }
    
    def validate_config(self) -> bool:
        """
        Validate configuration settings
        
        Returns:
            bool: True if configuration is valid
            
        Raises:
            SettingsValidationError: If critical configuration is invalid
        """
        errors = []
        
        # Check critical API keys
        if not self.GOOGLE_API_KEY:
            errors.append("GOOGLE_API_KEY is required for AI functionality")
        
        # Validate numeric ranges
        if not 0 <= self.SIMILARITY_THRESHOLD <= 1:
            errors.append("SIMILARITY_THRESHOLD must be between 0 and 1")
            
        if self.MAX_RETRIEVED_CHUNKS <= 0:
            errors.append("MAX_RETRIEVED_CHUNKS must be positive")
            
        if self.CONTEXT_WINDOW <= 0:
            errors.append("CONTEXT_WINDOW must be positive")
            
        if self.EMBEDDING_DIMENSIONS not in [768, 1536, 3072]:
            logger.warning(f"EMBEDDING_DIMENSIONS {self.EMBEDDING_DIMENSIONS} is not optimal. Recommended: 768, 1536, or 3072")
        
        # Validate directory permissions
        try:
            os.makedirs(self.CHROMA_PERSIST_DIRECTORY, exist_ok=True)
            test_file = os.path.join(self.CHROMA_PERSIST_DIRECTORY, ".test")
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
        except (OSError, PermissionError) as e:
            errors.append(f"Cannot write to vector database directory: {e}")
        
        if errors:
            raise SettingsValidationError("; ".join(errors))
        
        # Log warnings for optional configurations
        if not self.get_whatsapp_config():
            logger.warning("WhatsApp integration not configured - mobile features will be limited")
            
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Export settings as dictionary (excluding sensitive data)"""
        return {
            "app_title": self.APP_TITLE,
            "llm_model": self.LLM_MODEL,
            "embedding_model": self.EMBEDDING_MODEL,
            "embedding_dimensions": self.EMBEDDING_DIMENSIONS,
            "chunk_size": self.CHUNK_SIZE,
            "similarity_threshold": self.SIMILARITY_THRESHOLD,
            "max_chunks": self.MAX_RETRIEVED_CHUNKS,
            "has_google_key": bool(self.GOOGLE_API_KEY),
            "has_whatsapp_config": bool(self.get_whatsapp_config()),
            "supported_countries": len(self.AFRICAN_COUNTRIES),
            "supported_sectors": len(self.BUSINESS_SECTORS)
        }

# Global settings instance
try:
    settings = Settings()
except SettingsValidationError as e:
    logger.error(f"Failed to load settings: {e}")
    # Create a minimal fallback settings for development
    settings = type('FallbackSettings', (), {
        'GOOGLE_API_KEY': None,
        'CHROMA_PERSIST_DIRECTORY': './vector_db',
        'COLLECTION_NAME': 'african_business_knowledge',
        'APP_TITLE': 'UbuntuAI - Configuration Error',
        'validate_config': lambda: False
    })()
    logger.warning("Using fallback settings due to configuration errors")