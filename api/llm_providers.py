"""
Multi-Provider LLM Abstraction for UbuntuAI
Supports OpenAI, Anthropic, Google Gemini, and Ollama with automatic fallback
"""

from typing import Dict, Any, Optional, List, Union, AsyncGenerator
from abc import ABC, abstractmethod
import logging
from langchain_core.language_models import BaseLLM
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_core.outputs import LLMResult
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from config.settings import settings

logger = logging.getLogger(__name__)

class LLMProviderError(Exception):
    """Custom exception for LLM provider errors"""
    pass

class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    def __init__(self, provider_name: str):
        self.provider_name = provider_name
        self.is_available = False
        self.llm = None
        self._initialize()
    
    @abstractmethod
    def _initialize(self):
        """Initialize the LLM provider"""
        pass
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text response"""
        pass
    
    @abstractmethod
    def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """Generate streaming response"""
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model"""
        return {
            "provider": self.provider_name,
            "is_available": self.is_available,
            "model": getattr(self.llm, 'model_name', 'unknown') if self.llm else None
        }

class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM Provider"""
    
    def _initialize(self):
        try:
            if not settings.OPENAI_API_KEY:
                raise LLMProviderError("OpenAI API key not configured")
            
            config = settings.get_llm_config("openai")
            self.llm = ChatOpenAI(
                api_key=settings.OPENAI_API_KEY,
                model=config["model"],
                temperature=config["temperature"],
                max_tokens=config["max_tokens"]
            )
            self.is_available = True
            logger.info(f"OpenAI provider initialized with model: {config['model']}")
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI provider: {e}")
            self.is_available = False
    
    def generate(self, prompt: str, **kwargs) -> str:
        if not self.is_available:
            raise LLMProviderError("OpenAI provider not available")
        
        try:
            messages = [HumanMessage(content=prompt)]
            if kwargs.get('system_message'):
                messages.insert(0, SystemMessage(content=kwargs['system_message']))
            
            response = self.llm.invoke(messages)
            return response.content
            
        except Exception as e:
            logger.error(f"OpenAI generation error: {e}")
            raise LLMProviderError(f"OpenAI generation failed: {e}")
    
    async def generate_stream(self, prompt: str, **kwargs):
        if not self.is_available:
            raise LLMProviderError("OpenAI provider not available")
        
        try:
            messages = [HumanMessage(content=prompt)]
            if kwargs.get('system_message'):
                messages.insert(0, SystemMessage(content=kwargs['system_message']))
            
            async for chunk in self.llm.astream(messages):
                if chunk.content:
                    yield chunk.content
                    
        except Exception as e:
            logger.error(f"OpenAI streaming error: {e}")
            raise LLMProviderError(f"OpenAI streaming failed: {e}")

class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude LLM Provider"""
    
    def _initialize(self):
        try:
            if not settings.ANTHROPIC_API_KEY:
                raise LLMProviderError("Anthropic API key not configured")
            
            config = settings.get_llm_config("anthropic")
            self.llm = ChatAnthropic(
                anthropic_api_key=settings.ANTHROPIC_API_KEY,
                model=config["model"],
                temperature=config["temperature"],
                max_tokens=config["max_tokens"]
            )
            self.is_available = True
            logger.info(f"Anthropic provider initialized with model: {config['model']}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic provider: {e}")
            self.is_available = False
    
    def generate(self, prompt: str, **kwargs) -> str:
        if not self.is_available:
            raise LLMProviderError("Anthropic provider not available")
        
        try:
            messages = [HumanMessage(content=prompt)]
            if kwargs.get('system_message'):
                messages.insert(0, SystemMessage(content=kwargs['system_message']))
            
            response = self.llm.invoke(messages)
            return response.content
            
        except Exception as e:
            logger.error(f"Anthropic generation error: {e}")
            raise LLMProviderError(f"Anthropic generation failed: {e}")
    
    async def generate_stream(self, prompt: str, **kwargs):
        if not self.is_available:
            raise LLMProviderError("Anthropic provider not available")
        
        try:
            messages = [HumanMessage(content=prompt)]
            if kwargs.get('system_message'):
                messages.insert(0, SystemMessage(content=kwargs['system_message']))
            
            async for chunk in self.llm.astream(messages):
                if chunk.content:
                    yield chunk.content
                    
        except Exception as e:
            logger.error(f"Anthropic streaming error: {e}")
            raise LLMProviderError(f"Anthropic streaming failed: {e}")

class GoogleProvider(BaseLLMProvider):
    """Google Gemini LLM Provider"""
    
    def _initialize(self):
        try:
            if not settings.GOOGLE_API_KEY:
                raise LLMProviderError("Google API key not configured")
            
            config = settings.get_llm_config("google")
            self.llm = ChatGoogleGenerativeAI(
                google_api_key=settings.GOOGLE_API_KEY,
                model=config["model"],
                temperature=config["temperature"],
                max_output_tokens=config["max_tokens"]
            )
            self.is_available = True
            logger.info(f"Google provider initialized with model: {config['model']}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Google provider: {e}")
            self.is_available = False
    
    def generate(self, prompt: str, **kwargs) -> str:
        if not self.is_available:
            raise LLMProviderError("Google provider not available")
        
        try:
            messages = [HumanMessage(content=prompt)]
            if kwargs.get('system_message'):
                messages.insert(0, SystemMessage(content=kwargs['system_message']))
            
            response = self.llm.invoke(messages)
            return response.content
            
        except Exception as e:
            logger.error(f"Google generation error: {e}")
            raise LLMProviderError(f"Google generation failed: {e}")
    
    async def generate_stream(self, prompt: str, **kwargs):
        if not self.is_available:
            raise LLMProviderError("Google provider not available")
        
        try:
            messages = [HumanMessage(content=prompt)]
            if kwargs.get('system_message'):
                messages.insert(0, SystemMessage(content=kwargs['system_message']))
            
            async for chunk in self.llm.astream(messages):
                if chunk.content:
                    yield chunk.content
                    
        except Exception as e:
            logger.error(f"Google streaming error: {e}")
            raise LLMProviderError(f"Google streaming failed: {e}")

class OllamaProvider(BaseLLMProvider):
    """Ollama Local LLM Provider"""
    
    def _initialize(self):
        try:
            config = settings.get_llm_config("ollama")
            self.llm = ChatOllama(
                model=config["model"],
                temperature=config["temperature"],
                base_url=config.get("base_url", "http://localhost:11434")
            )
            
            # Test if Ollama is available
            try:
                test_response = self.llm.invoke([HumanMessage(content="Hi")])
                self.is_available = True
                logger.info(f"Ollama provider initialized with model: {config['model']}")
            except Exception:
                self.is_available = False
                logger.warning("Ollama service not available")
            
        except Exception as e:
            logger.error(f"Failed to initialize Ollama provider: {e}")
            self.is_available = False
    
    def generate(self, prompt: str, **kwargs) -> str:
        if not self.is_available:
            raise LLMProviderError("Ollama provider not available")
        
        try:
            messages = [HumanMessage(content=prompt)]
            if kwargs.get('system_message'):
                messages.insert(0, SystemMessage(content=kwargs['system_message']))
            
            response = self.llm.invoke(messages)
            return response.content
            
        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            raise LLMProviderError(f"Ollama generation failed: {e}")
    
    async def generate_stream(self, prompt: str, **kwargs):
        if not self.is_available:
            raise LLMProviderError("Ollama provider not available")
        
        try:
            messages = [HumanMessage(content=prompt)]
            if kwargs.get('system_message'):
                messages.insert(0, SystemMessage(content=kwargs['system_message']))
            
            async for chunk in self.llm.astream(messages):
                if chunk.content:
                    yield chunk.content
                    
        except Exception as e:
            logger.error(f"Ollama streaming error: {e}")
            raise LLMProviderError(f"Ollama streaming failed: {e}")

class LLMProviderManager:
    """Manages multiple LLM providers with automatic fallback"""
    
    def __init__(self):
        self.providers = {}
        self.primary_provider = settings.PRIMARY_LLM_PROVIDER
        self.fallback_order = ["openai", "anthropic", "google", "ollama"]
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize all available providers"""
        provider_classes = {
            "openai": OpenAIProvider,
            "anthropic": AnthropicProvider,
            "google": GoogleProvider,
            "ollama": OllamaProvider
        }
        
        for provider_name, provider_class in provider_classes.items():
            try:
                provider = provider_class(provider_name)
                self.providers[provider_name] = provider
                logger.info(f"Provider {provider_name}: {'Available' if provider.is_available else 'Unavailable'}")
            except Exception as e:
                logger.error(f"Failed to initialize {provider_name} provider: {e}")
    
    def get_available_providers(self) -> List[str]:
        """Get list of available providers"""
        return [name for name, provider in self.providers.items() if provider.is_available]
    
    def get_provider(self, provider_name: str = None) -> BaseLLMProvider:
        """Get a specific provider or primary provider"""
        if provider_name and provider_name in self.providers:
            if self.providers[provider_name].is_available:
                return self.providers[provider_name]
            else:
                raise LLMProviderError(f"Provider {provider_name} is not available")
        
        # Use primary provider
        if self.primary_provider in self.providers and self.providers[self.primary_provider].is_available:
            return self.providers[self.primary_provider]
        
        # Fallback to any available provider
        for provider_name in self.fallback_order:
            if provider_name in self.providers and self.providers[provider_name].is_available:
                logger.warning(f"Falling back to {provider_name} provider")
                return self.providers[provider_name]
        
        raise LLMProviderError("No LLM providers available")
    
    def generate(self, prompt: str, provider: str = None, **kwargs) -> str:
        """Generate text using specified or best available provider"""
        llm_provider = self.get_provider(provider)
        return llm_provider.generate(prompt, **kwargs)
    
    async def generate_stream(self, prompt: str, provider: str = None, **kwargs):
        """Generate streaming text using specified or best available provider"""
        llm_provider = self.get_provider(provider)
        async for chunk in llm_provider.generate_stream(prompt, **kwargs):
            yield chunk
    
    def get_langchain_llm(self, provider: str = None) -> BaseLLM:
        """Get LangChain-compatible LLM instance"""
        llm_provider = self.get_provider(provider)
        return llm_provider.llm
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all providers"""
        status = {
            "primary_provider": self.primary_provider,
            "available_providers": self.get_available_providers(),
            "provider_details": {}
        }
        
        for name, provider in self.providers.items():
            status["provider_details"][name] = provider.get_model_info()
        
        return status

# Global provider manager instance
try:
    llm_manager = LLMProviderManager()
    logger.info(f"LLM Provider Manager initialized with {len(llm_manager.get_available_providers())} available providers")
except Exception as e:
    logger.error(f"Failed to initialize LLM Provider Manager: {e}")
    llm_manager = None