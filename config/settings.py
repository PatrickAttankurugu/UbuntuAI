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
        
        # WhatsApp/Twilio Configuration
        self.TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
        self.TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
        self.TWILIO_WHATSAPP_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER")
        
        # Agent Configuration
        self.USE_LANGCHAIN_AGENTS = os.getenv("USE_LANGCHAIN_AGENTS", "true").lower() == "true"
        self.AGENT_MAX_ITERATIONS = int(os.getenv("AGENT_MAX_ITERATIONS", "5"))
        self.AGENT_TIMEOUT = int(os.getenv("AGENT_TIMEOUT", "120"))  # seconds
        
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
        
        # Business Context Configuration - Enhanced for Ghana
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
        
        # Ghana-specific regions and cities
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
            "Mining", "Oil & Gas", "Water", "Waste Management",
            # Ghana-specific sectors
            "Cocoa Processing", "Gold Mining", "Palm Oil", "Cassava Processing",
            "Poultry Farming", "Aquaculture", "Timber", "Textiles"
        ]
        
        # Enhanced funding stages with Ghana context
        self.FUNDING_STAGES = [
            "Pre-seed", "Seed", "Series A", "Series B", "Series C",
            "Growth", "Bridge", "Mezzanine", "IPO", "Grant", "Government",
            # Ghana-specific
            "Microfinance", "Community Fund", "Cooperative", "Susu Group"
        ]
        
        # Ghana business development priorities
        self.GHANA_BUSINESS_PRIORITIES = [
            "Digital inclusion", "Financial inclusion", "Agricultural productivity",
            "Job creation", "Export promotion", "Import substitution",
            "Rural development", "Women empowerment", "Youth employment"
        ]
        
        # UI Configuration
        self.APP_TITLE = "UbuntuAI - African Business Intelligence"
        self.APP_DESCRIPTION = "Your AI-powered guide to African entrepreneurship"
        self.CHAT_PLACEHOLDER = "Ask me anything about African business, funding, regulations, or market insights..."
        
        # Performance Configuration - Enhanced for low-resource environments
        self.CACHE_TTL = 3600
        self.MAX_CONCURRENT_REQUESTS = 10
        self.REQUEST_TIMEOUT = 30
        
        # WhatsApp optimization settings
        self.WHATSAPP_MAX_MESSAGE_LENGTH = 1600
        self.WHATSAPP_SESSION_TIMEOUT = 24 * 60 * 60  # 24 hours in seconds
        self.WHATSAPP_MAX_SESSIONS = 1000
        
        # Agent workflow settings
        self.AGENT_RESPONSE_CACHE = True
        self.AGENT_MAX_TOOL_CALLS = 10
        self.SCORING_CONFIDENCE_THRESHOLD = 0.6
        
        # Low-bandwidth optimizations
        self.OPTIMIZE_FOR_MOBILE = True
        self.COMPRESS_RESPONSES = True
        self.MAX_RESPONSE_TOKENS = 500  # For mobile users
    
    def get_model_config(self) -> Dict[str, Any]:
        return {
            "temperature": self.TEMPERATURE,
            "max_tokens": self.MAX_TOKENS,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0
        }
    
    def get_mobile_model_config(self) -> Dict[str, Any]:
        """Optimized config for mobile/WhatsApp users"""
        return {
            "temperature": 0.3,
            "max_tokens": self.MAX_RESPONSE_TOKENS,
            "top_p": 0.9,
            "frequency_penalty": 0.1,
            "presence_penalty": 0.1
        }
    
    def get_agent_config(self) -> Dict[str, Any]:
        """Configuration for LangChain agents"""
        return {
            "max_iterations": self.AGENT_MAX_ITERATIONS,
            "timeout": self.AGENT_TIMEOUT,
            "use_agents": self.USE_LANGCHAIN_AGENTS,
            "max_tool_calls": self.AGENT_MAX_TOOL_CALLS
        }
    
    def get_ghana_context(self) -> Dict[str, Any]:
        """Get Ghana-specific business context"""
        return {
            "regions": self.GHANA_REGIONS,
            "major_cities": self.GHANA_MAJOR_CITIES,
            "business_priorities": self.GHANA_BUSINESS_PRIORITIES,
            "local_sectors": [s for s in self.BUSINESS_SECTORS if s in [
                "Cocoa Processing", "Gold Mining", "Palm Oil", "Cassava Processing",
                "Poultry Farming", "Aquaculture", "Timber", "Textiles"
            ]]
        }
    
    def get_whatsapp_config(self) -> Dict[str, Any]:
        """Get WhatsApp integration configuration"""
        return {
            "account_sid": self.TWILIO_ACCOUNT_SID,
            "auth_token": self.TWILIO_AUTH_TOKEN,
            "whatsapp_number": self.TWILIO_WHATSAPP_NUMBER,
            "max_message_length": self.WHATSAPP_MAX_MESSAGE_LENGTH,
            "session_timeout": self.WHATSAPP_SESSION_TIMEOUT,
            "max_sessions": self.WHATSAPP_MAX_SESSIONS
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
        
        # Validate WhatsApp config if agents are enabled
        if self.USE_LANGCHAIN_AGENTS:
            if not all([self.TWILIO_ACCOUNT_SID, self.TWILIO_AUTH_TOKEN, self.TWILIO_WHATSAPP_NUMBER]):
                print("WARNING: WhatsApp integration not fully configured. Some features may be disabled.")
        
        return True

settings = Settings()