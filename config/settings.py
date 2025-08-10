import os
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

class Settings:
    def __init__(self):
        # API Keys
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        self.PINECONE_ENV = os.getenv("PINECONE_ENV")
        
        # Vector Database Configuration
        self.CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "./vector_db")
        self.COLLECTION_NAME = "african_business_knowledge"
        
        # Embedding Configuration
        self.EMBEDDING_MODEL = "text-embedding-ada-002"
        self.EMBEDDING_DIMENSIONS = 1536
        self.CHUNK_SIZE = 1024
        self.CHUNK_OVERLAP = 200
        
        # RAG Configuration
        self.SIMILARITY_THRESHOLD = 0.7
        self.MAX_RETRIEVED_CHUNKS = 10
        self.CONTEXT_WINDOW = 8000
        self.TEMPERATURE = 0.3
        self.MAX_TOKENS = 1000
        
        # Business Context Configuration
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
            "Pre-seed", "Seed", "Series A", "Series B", "Series C",
            "Growth", "Bridge", "Mezzanine", "IPO", "Grant", "Government"
        ]
        
        # UI Configuration
        self.APP_TITLE = "UbuntuAI - African Business Intelligence"
        self.APP_DESCRIPTION = "Your AI-powered guide to African entrepreneurship"
        self.CHAT_PLACEHOLDER = "Ask me anything about African business, funding, regulations, or market insights..."
        
        # Performance Configuration
        self.CACHE_TTL = 3600
        self.MAX_CONCURRENT_REQUESTS = 10
        self.REQUEST_TIMEOUT = 30
        
    def get_model_config(self) -> Dict[str, Any]:
        return {
            "temperature": self.TEMPERATURE,
            "max_tokens": self.MAX_TOKENS,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0
        }
    
    def get_embedding_config(self) -> Dict[str, Any]:
        return {
            "model": self.EMBEDDING_MODEL,
            "dimensions": self.EMBEDDING_DIMENSIONS
        }
    
    def get_chunking_config(self) -> Dict[str, Any]:
        return {
            "chunk_size": self.CHUNK_SIZE,
            "chunk_overlap": self.CHUNK_OVERLAP
        }
    
    def validate_config(self) -> bool:
        if not self.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required")
        
        return True

settings = Settings()