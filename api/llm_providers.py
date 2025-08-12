#!/usr/bin/env python3
"""
LLM Provider Manager for UbuntuAI
Manages different LLM providers with Google Gemini as primary
"""

import os
import logging
from typing import List, Dict, Any, Optional, Union
import google.generativeai as genai
from datetime import datetime

from config.settings import settings

logger = logging.getLogger(__name__)

class LLMProvider:
    """Base class for LLM providers"""
    
    def __init__(self, name: str):
        self.name = name
        self.is_available = False
        self.model_name = ""
        self._initialize()
    
    def _initialize(self):
        """Initialize the provider - to be implemented by subclasses"""
        pass
    
    def generate(self, prompt: str, **kwargs) -> Optional[str]:
        """Generate text from prompt - to be implemented by subclasses"""
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """Get provider information"""
        return {
            "name": self.name,
            "is_available": self.is_available,
            "model_name": self.model_name
        }

class GoogleGeminiProvider(LLMProvider):
    """Google Gemini LLM provider"""
    
    def _initialize(self):
        """Initialize Google Gemini provider"""
        try:
            if not settings.GOOGLE_API_KEY:
                logger.warning("Google API key not configured")
                return
            
            # Configure Google API
            genai.configure(api_key=settings.GOOGLE_API_KEY)
            
            # Test the API with a simple request
            model = genai.GenerativeModel('gemini-1.5-flash')
            test_response = model.generate_content("Hello")
            
            if test_response and test_response.text:
                self.is_available = True
                self.model_name = settings.LLM_MODEL
                logger.info(f"Google Gemini provider initialized successfully with model: {self.model_name}")
            else:
                logger.warning("Google Gemini API test failed")
                
        except Exception as e:
            logger.error(f"Failed to initialize Google Gemini provider: {e}")
            self.is_available = False
    
    def generate(self, prompt: str, **kwargs) -> Optional[str]:
        """Generate text using Google Gemini"""
        if not self.is_available:
            logger.error("Google Gemini provider not available")
            return None
        
        try:
            # Get model configuration
            model_config = settings.get_llm_config("google")
            temperature = kwargs.get('temperature', model_config.get('temperature', 0.7))
            max_tokens = kwargs.get('max_tokens', model_config.get('max_tokens', 2048))
            
            # Create model instance
            model = genai.GenerativeModel(self.model_name)
            
            # Generate content
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens
                )
            )
            
            if response and response.text:
                logger.debug(f"Generated response of length {len(response.text)}")
                return response.text.strip()
            else:
                logger.warning("Google Gemini returned empty response")
                return None
                
        except Exception as e:
            logger.error(f"Error generating text with Google Gemini: {e}")
            return None

class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider"""
    
    def _initialize(self):
        """Initialize OpenAI provider"""
        try:
            if not settings.OPENAI_API_KEY:
                logger.warning("OpenAI API key not configured")
                return
            
            # For now, just mark as available if API key exists
            # In a full implementation, you would test the API here
            self.is_available = True
            self.model_name = "gpt-4"
            logger.info("OpenAI provider initialized (API key configured)")
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI provider: {e}")
            self.is_available = False
    
    def generate(self, prompt: str, **kwargs) -> Optional[str]:
        """Generate text using OpenAI (placeholder implementation)"""
        if not self.is_available:
            logger.error("OpenAI provider not available")
            return None
        
        # This is a placeholder - in a real implementation you would use the OpenAI API
        logger.warning("OpenAI provider not fully implemented - using fallback")
        return f"OpenAI response placeholder for: {prompt[:100]}..."

class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider"""
    
    def _initialize(self):
        """Initialize Anthropic provider"""
        try:
            if not settings.ANTHROPIC_API_KEY:
                logger.warning("Anthropic API key not configured")
                return
            
            # For now, just mark as available if API key exists
            self.is_available = True
            self.model_name = "claude-3-sonnet-20240229"
            logger.info("Anthropic provider initialized (API key configured)")
            
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic provider: {e}")
            self.is_available = False
    
    def generate(self, prompt: str, **kwargs) -> Optional[str]:
        """Generate text using Anthropic (placeholder implementation)"""
        if not self.is_available:
            logger.error("Anthropic provider not available")
            return None
        
        # This is a placeholder - in a real implementation you would use the Anthropic API
        logger.warning("Anthropic provider not fully implemented - using fallback")
        return f"Anthropic response placeholder for: {prompt[:100]}..."

class LLMProviderManager:
    """Manager for multiple LLM providers"""
    
    def __init__(self):
        """Initialize the LLM provider manager"""
        self.providers = {}
        self.primary_provider = settings.PRIMARY_LLM_PROVIDER
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize all available LLM providers"""
        # Initialize Google Gemini
        try:
            google_provider = GoogleGeminiProvider("google")
            self.providers["google"] = google_provider
        except Exception as e:
            logger.error(f"Failed to initialize Google provider: {e}")
        
        # Initialize OpenAI
        try:
            openai_provider = OpenAIProvider("openai")
            self.providers["openai"] = openai_provider
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI provider: {e}")
        
        # Initialize Anthropic
        try:
            anthropic_provider = AnthropicProvider("anthropic")
            self.providers["anthropic"] = anthropic_provider
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic provider: {e}")
        
        logger.info(f"Initialized {len(self.providers)} LLM providers")
    
    def get_provider(self, provider_name: str = None) -> Optional[LLMProvider]:
        """Get a specific LLM provider"""
        provider_name = provider_name or self.primary_provider
        
        if provider_name in self.providers:
            provider = self.providers[provider_name]
            if provider.is_available:
                return provider
            else:
                logger.warning(f"Provider {provider_name} not available")
        
        # Fallback to any available provider
        for name, provider in self.providers.items():
            if provider.is_available:
                logger.warning(f"Falling back to {name} provider")
                return provider
        
        logger.error("No LLM providers available")
        return None
    
    def generate(self, prompt: str, provider: str = None, **kwargs) -> Optional[str]:
        """Generate text using the specified or best available provider"""
        llm_provider = self.get_provider(provider)
        
        if not llm_provider:
            logger.error("No LLM provider available for text generation")
            return None
        
        try:
            return llm_provider.generate(prompt, **kwargs)
        except Exception as e:
            logger.error(f"Text generation failed with {llm_provider.name}: {e}")
            return None
    
    def get_available_providers(self) -> List[str]:
        """Get list of available providers"""
        return [name for name, provider in self.providers.items() if provider.is_available]
    
    def get_provider_info(self, provider: str = None) -> Dict[str, Any]:
        """Get detailed provider information"""
        if provider:
            if provider in self.providers:
                return self.providers[provider].get_info()
            else:
                return {"error": f"Provider {provider} not found"}
        
        # Return info for all providers
        info = {
            "primary_provider": self.primary_provider,
            "available_providers": self.get_available_providers(),
            "provider_details": {}
        }
        
        for name, provider in self.providers.items():
            info["provider_details"][name] = provider.get_info()
        
        return info
    
    def switch_primary_provider(self, provider_name: str) -> bool:
        """Switch to a different primary provider"""
        if provider_name in self.providers and self.providers[provider_name].is_available:
            self.primary_provider = provider_name
            logger.info(f"Switched primary provider to {provider_name}")
            return True
        else:
            logger.error(f"Cannot switch to {provider_name} - not available")
            return False
    
    def test_provider(self, provider_name: str) -> bool:
        """Test a specific provider"""
        if provider_name not in self.providers:
            logger.error(f"Provider {provider_name} not found")
            return False
        
        provider = self.providers[provider_name]
        if not provider.is_available:
            logger.warning(f"Provider {provider_name} not available")
            return False
        
        try:
            # Test with a simple prompt
            test_prompt = "Hello, this is a test message."
            response = provider.generate(test_prompt)
            
            if response:
                logger.info(f"Provider {provider_name} test successful")
                return True
            else:
                logger.warning(f"Provider {provider_name} test failed - no response")
                return False
                
        except Exception as e:
            logger.error(f"Provider {provider_name} test failed: {e}")
            return False

# Global LLM provider manager instance
try:
    llm_manager = LLMProviderManager()
    logger.info("LLM provider manager initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize LLM provider manager: {e}")
    llm_manager = None

# Convenience functions
def get_llm_provider(provider_name: str = None) -> Optional[LLMProvider]:
    """Get an LLM provider"""
    if llm_manager:
        return llm_manager.get_provider(provider_name)
    else:
        logger.error("LLM provider manager not available")
        return None

def generate_text(prompt: str, provider: str = None, **kwargs) -> Optional[str]:
    """Generate text using an LLM provider"""
    if llm_manager:
        return llm_manager.generate(prompt, provider, **kwargs)
    else:
        logger.error("LLM provider manager not available")
        return None

def get_available_llm_providers() -> List[str]:
    """Get list of available LLM providers"""
    if llm_manager:
        return llm_manager.get_available_providers()
    else:
        logger.error("LLM provider manager not available")
        return []