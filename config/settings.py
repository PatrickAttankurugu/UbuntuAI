#!/usr/bin/env python3
"""
Configuration settings for UbuntuAI
Centralized configuration management for the African Business Intelligence Platform
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import logging

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Settings:
    """Application settings and configuration"""
    
    def __init__(self):
        # Base paths
        self.BASE_DIR = Path(__file__).parent.parent
        self.DATA_DIR = self.BASE_DIR / "data"
        self.VECTOR_DB_DIR = self.BASE_DIR / "vector_db"
        self.LOGS_DIR = self.BASE_DIR / "logs"
        
        # Create necessary directories
        self._create_directories()
        
        # API Keys
        self.GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
        self.ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
        
        # Twilio Configuration (for WhatsApp)
        self.TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
        self.TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
        self.TWILIO_WHATSAPP_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER", "")
        
        # LLM Configuration
        self.PRIMARY_LLM_PROVIDER = os.getenv("PRIMARY_LLM_PROVIDER", "google")
        self.LLM_MODEL = os.getenv("LLM_MODEL", "gemini-1.5-flash")
        self.EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-004")
        self.EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "768"))
        
        # Vector Store Configuration
        self.VECTOR_STORE_TYPE = os.getenv("VECTOR_STORE_TYPE", "chroma")
        self.VECTOR_STORE_PATH = str(self.VECTOR_DB_DIR / "chroma_db")
        self.COLLECTION_NAME = os.getenv("COLLECTION_NAME", "ubuntuai_knowledge")
        
        # RAG Configuration
        self.RETRIEVAL_STRATEGY = os.getenv("RETRIEVAL_STRATEGY", "hybrid")
        self.MAX_RETRIEVAL_RESULTS = int(os.getenv("MAX_RETRIEVAL_RESULTS", "5"))
        self.CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
        self.CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
        
        # Ghana-specific Configuration
        self.SUPPORTED_SECTORS = [
            "fintech", "agritech", "healthtech", "edtech", 
            "logistics", "ecommerce", "renewable_energy"
        ]
        self.GHANA_REGIONS = [
            "Greater Accra", "Ashanti", "Western", "Central", "Eastern",
            "Volta", "Northern", "Upper East", "Upper West", "Bono",
            "Bono East", "Ahafo", "Savannah", "North East", "Oti", "Western North"
        ]
        
        # Performance Configuration
        self.MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "10"))
        self.REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))
        self.CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))
        
        # Security Configuration
        self.ENABLE_RATE_LIMITING = os.getenv("ENABLE_RATE_LIMITING", "true").lower() == "true"
        self.MAX_REQUESTS_PER_MINUTE = int(os.getenv("MAX_REQUESTS_PER_MINUTE", "60"))
        self.ENABLE_CONTENT_FILTERING = os.getenv("ENABLE_CONTENT_FILTERING", "true").lower() == "true"
        
        # Logging Configuration
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
        self.ENABLE_FILE_LOGGING = os.getenv("ENABLE_FILE_LOGGING", "true").lower() == "true"
        
        # Validation
        self._validate_configuration()
    
    def _create_directories(self):
        """Create necessary directories if they don't exist"""
        directories = [
            self.DATA_DIR,
            self.DATA_DIR / "processed",
            self.DATA_DIR / "sources",
            self.VECTOR_DB_DIR,
            self.LOGS_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _validate_configuration(self):
        """Validate critical configuration settings"""
        errors = []
        
        # Check required API keys
        if not self.GOOGLE_API_KEY:
            errors.append("GOOGLE_API_KEY is required")
        
        # Check if directories exist
        if not self.DATA_DIR.exists():
            errors.append(f"Data directory {self.DATA_DIR} does not exist")
        
        # Log configuration status
        if errors:
            logger.error(f"Configuration validation failed: {', '.join(errors)}")
            for error in errors:
                logger.error(f"  - {error}")
        else:
            logger.info("Configuration validation passed")
    
    def get_available_llm_providers(self) -> List[str]:
        """Get list of available LLM providers based on configured API keys"""
        providers = []
        
        if self.GOOGLE_API_KEY:
            providers.append("google")
        if self.OPENAI_API_KEY:
            providers.append("openai")
        if self.ANTHROPIC_API_KEY:
            providers.append("anthropic")
        
        return providers
    
    def is_llm_provider_available(self, provider: str) -> bool:
        """Check if a specific LLM provider is available"""
        return provider in self.get_available_llm_providers()
    
    def get_llm_config(self, provider: str) -> Dict[str, Any]:
        """Get configuration for a specific LLM provider"""
        configs = {
            "google": {
                "api_key": self.GOOGLE_API_KEY,
                "model": self.LLM_MODEL,
                "temperature": 0.7,
                "max_tokens": 2048
            },
            "openai": {
                "api_key": self.OPENAI_API_KEY,
                "model": "gpt-4",
                "temperature": 0.7,
                "max_tokens": 2048
            },
            "anthropic": {
                "api_key": self.ANTHROPIC_API_KEY,
                "model": "claude-3-sonnet-20240229",
                "temperature": 0.7,
                "max_tokens": 2048
            }
        }
        
        return configs.get(provider, {})
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary for logging/debugging"""
        return {
            "primary_llm_provider": self.PRIMARY_LLM_PROVIDER,
            "available_llm_providers": self.get_available_llm_providers(),
            "vector_store_type": self.VECTOR_STORE_TYPE,
            "retrieval_strategy": self.RETRIEVAL_STRATEGY,
            "supported_sectors": self.SUPPORTED_SECTORS,
            "ghana_regions": self.GHANA_REGIONS,
            "data_directory": str(self.DATA_DIR),
            "vector_db_directory": str(self.VECTOR_DB_DIR)
        }
    
    def update_setting(self, key: str, value: Any):
        """Update a setting value"""
        if hasattr(self, key):
            setattr(self, key, value)
            logger.info(f"Updated setting {key} = {value}")
        else:
            logger.warning(f"Unknown setting key: {key}")

# Create global settings instance
settings = Settings()

# Export commonly used settings
GOOGLE_API_KEY = settings.GOOGLE_API_KEY
OPENAI_API_KEY = settings.OPENAI_API_KEY
ANTHROPIC_API_KEY = settings.ANTHROPIC_API_KEY
TWILIO_ACCOUNT_SID = settings.TWILIO_ACCOUNT_SID
TWILIO_AUTH_TOKEN = settings.TWILIO_AUTH_TOKEN
TWILIO_WHATSAPP_NUMBER = settings.TWILIO_WHATSAPP_NUMBER
PRIMARY_LLM_PROVIDER = settings.PRIMARY_LLM_PROVIDER
LLM_MODEL = settings.LLM_MODEL
EMBEDDING_MODEL = settings.EMBEDDING_MODEL
EMBEDDING_DIMENSIONS = settings.EMBEDDING_DIMENSIONS
VECTOR_STORE_TYPE = settings.VECTOR_STORE_TYPE
RETRIEVAL_STRATEGY = settings.RETRIEVAL_STRATEGY
MAX_RETRIEVAL_RESULTS = settings.MAX_RETRIEVAL_RESULTS
CHUNK_SIZE = settings.CHUNK_SIZE
CHUNK_OVERLAP = settings.CHUNK_OVERLAP
SUPPORTED_SECTORS = settings.SUPPORTED_SECTORS
GHANA_REGIONS = settings.GHANA_REGIONS