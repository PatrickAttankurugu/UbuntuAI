import asyncio
import json
import openai
from typing import Dict, List, Any, Optional, AsyncIterator, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import re
from enum import Enum
from abc import ABC, abstractmethod
import websockets
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse
from config.settings import settings

class InterfaceType(Enum):
    WHATSAPP = "whatsapp"
    SMS = "sms"
    VOICE = "voice"
    WEB_CHAT = "web_chat"
    USSD = "ussd"

class MessageType(Enum):
    TEXT = "text"
    AUDIO = "audio"
    IMAGE = "image"
    DOCUMENT = "document"
    LOCATION = "location"
    QUICK_REPLY = "quick_reply"

@dataclass
class StreamingMessage:
    message_id: str
    user_id: str
    interface_type: InterfaceType
    message_type: MessageType
    content: str
    metadata: Dict[str, Any]
    timestamp: datetime
    language: str = "en"
    context: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.context is None:
            self.context = {}

@dataclass
class StreamingResponse:
    response_id: str
    user_id: str
    interface_type: InterfaceType
    content: str
    response_type: str
    chunks: List[str]
    metadata: Dict[str, Any]
    quick_replies: List[str] = None
    attachments: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.quick_replies is None:
            self.quick_replies = []
        if self.attachments is None:
            self.attachments = []

class BaseStreamingInterface(ABC):
    """Base class for streaming interfaces"""
    
    def __init__(self, interface_type: InterfaceType):
        self.interface_type = interface_type
        self.active_sessions = {}
        self.message_handlers = {}
        self.response_formatters = {}
        
    @abstractmethod
    async def send_message(self, user_id: str, response: StreamingResponse) -> bool:
        """Send message to user"""
        pass
    
    @abstractmethod
    async def receive_message(self, raw_message: Dict[str, Any]) -> StreamingMessage:
        """Process received message"""
        pass
    
    @abstractmethod
    def optimize_for_bandwidth(self, content: str) -> str:
        """Optimize content for low bandwidth"""
        pass

class WhatsAppStreamingInterface(BaseStreamingInterface):
    """WhatsApp streaming interface with low-bandwidth optimization"""
    
    def __init__(self):
        super().__init__(InterfaceType.WHATSAPP)
        self.twilio_client = Client(
            settings.TWILIO_ACCOUNT_SID,
            settings.TWILIO_AUTH_TOKEN
        )
        self.whatsapp_number = settings.TWILIO_WHATSAPP_NUMBER
        
        # WhatsApp-specific optimizations
        self.max_message_length = 1600
        self.chunk_size = 800
        self.emoji_substitutions = {
            "excellent": "🟢",
            "good": "🟡", 
            "poor": "🔴",
            "money": "💰",
            "business": "🏢",
            "growth": "📈",
            "warning": "⚠️"
        }
        
    async def send_message(self, user_id: str, response: StreamingResponse) -> bool:
        """Send WhatsApp message with streaming support"""
        
        try:
            # Optimize content for WhatsApp
            optimized_content = self.optimize_for_bandwidth(response.content)
            
            # Handle long messages by chunking
            if len(optimized_content) > self.max_message_length:
                chunks = self._chunk_message(optimized_content)
                
                for i, chunk in enumerate(chunks):
                    await self._send_whatsapp_chunk(user_id, chunk, i, len(chunks))
                    
                    # Small delay between chunks to avoid rate limiting
                    if i < len(chunks) - 1:
                        await asyncio.sleep(0.5)
            else:
                await self._send_whatsapp_chunk(user_id, optimized_content)
            
            # Send quick replies if available
            if response.quick_replies:
                quick_reply_text = "Options:\n" + "\n".join([
                    f"{i+1}. {reply}" for i, reply in enumerate(response.quick_replies[:3])
                ])
                await self._send_whatsapp_chunk(user_id, quick_reply_text)
            
            return True
            
        except Exception as e:
            print(f"WhatsApp send error: {e}")
            return False
    
    async def receive_message(self, raw_message: Dict[str, Any]) -> StreamingMessage:
        """Process received WhatsApp message"""
        
        user_id = raw_message.get('From', '').replace('whatsapp:', '')
        content = raw_message.get('Body', '').strip()
        
        # Detect message type
        message_type = MessageType.TEXT
        if raw_message.get('MediaUrl0'):
            message_type = MessageType.IMAGE
        elif raw_message.get('Latitude') and raw_message.get('Longitude'):
            message_type = MessageType.LOCATION
        
        # Extract metadata
        metadata = {
            "media_url": raw_message.get('MediaUrl0'),
            "media_type": raw_message.get('MediaContentType0'),
            "location": {
                "lat": raw_message.get('Latitude'),
                "lng": raw_message.get('Longitude')
            } if raw_message.get('Latitude') else None,
            "profile_name": raw_message.get('ProfileName'),
            "wa_id": raw_message.get('WaId')
        }
        
        # Detect language (basic detection)
        language = self._detect_language(content)
        
        return StreamingMessage(
            message_id=f"wa_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            user_id=user_id,
            interface_type=InterfaceType.WHATSAPP,
            message_type=message_type,
            content=content,
            metadata=metadata,
            timestamp=datetime.now(),
            language=language
        )
    
    def optimize_for_bandwidth(self, content: str) -> str:
        """Optimize content for WhatsApp and low bandwidth"""
        
        optimized = content
        
        # Replace common words with emojis
        for word, emoji in self.emoji_substitutions.items():
            optimized = re.sub(f'\\b{word}\\b', emoji, optimized, flags=re.IGNORECASE)
        
        # Remove excessive formatting
        optimized = re.sub(r'\*\*\*+', '**', optimized)
        optimized = re.sub(r'\n\n\n+', '\n\n', optimized)
        
        # Shorten common phrases
        optimized = optimized.replace('please', 'pls')
        optimized = optimized.replace('you are', "you're")
        optimized = optimized.replace('can not', "can't")
        
        # Remove URLs and replace with placeholders
        optimized = re.sub(r'https?://[^\s]+', '[link]', optimized)
        
        return optimized
    
    async def _send_whatsapp_chunk(self, user_id: str, content: str, chunk_num: int = 0, total_chunks: int = 1):
        """Send individual WhatsApp message chunk"""
        
        # Add chunk indicator for multi-part messages
        if total_chunks > 1:
            content = f"({chunk_num + 1}/{total_chunks}) {content}"
        
        message = self.twilio_client.messages.create(
            body=content,
            from_=f"whatsapp:{self.whatsapp_number}",
            to=f"whatsapp:{user_id}"
        )
        
        return message.sid
    
    def _chunk_message(self, content: str) -> List[str]:
        """Split long message into WhatsApp-friendly chunks"""
        
        chunks = []
        current_chunk = ""
        
        # Split by paragraphs first
        paragraphs = content.split('\n\n')
        
        for paragraph in paragraphs:
            if len(current_chunk + paragraph) < self.chunk_size:
                current_chunk += paragraph + '\n\n'
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                
                # If single paragraph is too long, split by sentences
                if len(paragraph) > self.chunk_size:
                    sentences = re.split(r'[.!?]+', paragraph)
                    for sentence in sentences:
                        if sentence.strip():
                            if len(current_chunk + sentence) < self.chunk_size:
                                current_chunk += sentence + '. '
                            else:
                                if current_chunk:
                                    chunks.append(current_chunk.strip())
                                current_chunk = sentence + '. '
                else:
                    current_chunk = paragraph + '\n\n'
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _detect_language(self, content: str) -> str:
        """Basic language detection for African languages"""
        
        content_lower = content.lower()
        
        # Simple keyword-based detection
        twi_keywords = ['akwaaba', 'medaase', 'ɛwo hɛn', 'aane', 'daabi']
        if any(keyword in content_lower for keyword in twi_keywords):
            return 'tw'
        
        hausa_keywords = ['sannu', 'na gode', 'barka', 'hankali']
        if any(keyword in content_lower for keyword in hausa_keywords):
            return 'ha'
        
        # Default to English
        return 'en'

class SMSStreamingInterface(BaseStreamingInterface):
    """SMS interface for basic feature phones"""
    
    def __init__(self):
        super().__init__(InterfaceType.SMS)
        self.twilio_client = Client(
            settings.TWILIO_ACCOUNT_SID,
            settings.TWILIO_AUTH_TOKEN
        )
        self.max_message_length = 160
        
    async def send_message(self, user_id: str, response: StreamingResponse) -> bool:
        """Send SMS message"""
        
        try:
            # Heavy optimization for SMS
            optimized_content = self.optimize_for_bandwidth(response.content)
            
            # Split into SMS-sized chunks
            chunks = self._split_sms_chunks(optimized_content)
            
            for chunk in chunks:
                message = self.twilio_client.messages.create(
                    body=chunk,
                    from_=settings.TWILIO_PHONE_NUMBER,
                    to=user_id
                )
            
            return True
            
        except Exception as e:
            print(f"SMS send error: {e}")
            return False
    
    async def receive_message(self, raw_message: Dict[str, Any]) -> StreamingMessage:
        """Process received SMS"""
        
        return StreamingMessage(
            message_id=f"sms_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            user_id=raw_message.get('From', ''),
            interface_type=InterfaceType.SMS,
            message_type=MessageType.TEXT,
            content=raw_message.get('Body', ''),
            metadata={},
            timestamp=datetime.now()
        )
    
    def optimize_for_bandwidth(self, content: str) -> str:
        """Aggressive optimization for SMS"""
        
        # Extreme abbreviations for SMS
        abbreviations = {
            'business': 'biz',
            'information': 'info',
            'application': 'app',
            'opportunity': 'opp',
            'registration': 'reg',
            'government': 'gov',
            'development': 'dev',
            'investment': 'inv',
            'recommendation': 'rec',
            'assessment': 'eval'
        }
        
        optimized = content
        for full, abbrev in abbreviations.items():
            optimized = re.sub(f'\\b{full}\\b', abbrev, optimized, flags=re.IGNORECASE)
        
        # Remove formatting
        optimized = re.sub(r'\*\*([^*]+)\*\*', r'\1', optimized)
        optimized = re.sub(r'\n+', ' ', optimized)
        
        return optimized
    
    def _split_sms_chunks(self, content: str) -> List[str]:
        """Split content into SMS chunks"""
        
        chunks = []
        words = content.split()
        current_chunk = ""
        
        for word in words:
            if len(current_chunk + " " + word) < self.max_message_length:
                current_chunk += " " + word if current_chunk else word
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = word
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks

class VoiceStreamingInterface(BaseStreamingInterface):
    """Voice interface for audio interactions"""
    
    def __init__(self):
        super().__init__(InterfaceType.VOICE)
        self.twilio_client = Client(
            settings.TWILIO_ACCOUNT_SID,
            settings.TWILIO_AUTH_TOKEN
        )
        
    async def send_message(self, user_id: str, response: StreamingResponse) -> bool:
        """Send voice message (TTS)"""
        
        try:
            # Convert text to speech-optimized format
            speech_content = self.optimize_for_speech(response.content)
            
            # Use Twilio's TTS capabilities
            call = self.twilio_client.calls.create(
                twiml=f'<Response><Say voice="alice">{speech_content}</Say></Response>',
                to=user_id,
                from_=settings.TWILIO_PHONE_NUMBER
            )
            
            return True
            
        except Exception as e:
            print(f"Voice send error: {e}")
            return False
    
    async def receive_message(self, raw_message: Dict[str, Any]) -> StreamingMessage:
        """Process received voice message"""
        
        # Extract speech-to-text result
        speech_text = raw_message.get('SpeechResult', '')
        
        return StreamingMessage(
            message_id=f"voice_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            user_id=raw_message.get('From', ''),
            interface_type=InterfaceType.VOICE,
            message_type=MessageType.AUDIO,
            content=speech_text,
            metadata={
                "confidence": raw_message.get('Confidence', 0),
                "call_sid": raw_message.get('CallSid')
            },
            timestamp=datetime.now()
        )
    
    def optimize_for_bandwidth(self, content: str) -> str:
        """Optimize for voice delivery"""
        return self.optimize_for_speech(content)
    
    def optimize_for_speech(self, content: str) -> str:
        """Optimize content for text-to-speech"""
        
        # Remove formatting that doesn't work well with TTS
        speech_content = re.sub(r'\*\*([^*]+)\*\*', r'\1', content)
        speech_content = re.sub(r'[•\-]', 'and', speech_content)
        
        # Add natural pauses
        speech_content = speech_content.replace('\n', '. ')
        speech_content = speech_content.replace(':', ', ')
        
        # Expand abbreviations for clarity
        speech_content = speech_content.replace('&', 'and')
        speech_content = speech_content.replace('%', 'percent')
        speech_content = speech_content.replace('@', 'at')
        
        return speech_content

class WebChatStreamingInterface(BaseStreamingInterface):
    """Web chat interface with WebSocket support"""
    
    def __init__(self):
        super().__init__(InterfaceType.WEB_CHAT)
        self.websocket_connections = {}
        
    async def send_message(self, user_id: str, response: StreamingResponse) -> bool:
        """Send web chat message via WebSocket"""
        
        try:
            if user_id in self.websocket_connections:
                websocket = self.websocket_connections[user_id]
                
                # Send message in chunks for streaming effect
                for chunk in response.chunks:
                    await websocket.send(json.dumps({
                        "type": "chunk",
                        "content": chunk,
                        "user_id": user_id
                    }))
                    await asyncio.sleep(0.1)  # Streaming delay
                
                # Send final message
                await websocket.send(json.dumps({
                    "type": "complete",
                    "response": asdict(response)
                }))
                
                return True
            
            return False
            
        except Exception as e:
            print(f"WebChat send error: {e}")
            return False
    
    async def receive_message(self, raw_message: Dict[str, Any]) -> StreamingMessage:
        """Process received web chat message"""
        
        return StreamingMessage(
            message_id=f"web_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            user_id=raw_message.get('user_id'),
            interface_type=InterfaceType.WEB_CHAT,
            message_type=MessageType.TEXT,
            content=raw_message.get('content', ''),
            metadata=raw_message.get('metadata', {}),
            timestamp=datetime.now()
        )
    
    def optimize_for_bandwidth(self, content: str) -> str:
        """Web chat can handle full content"""
        return content
    
    async def handle_websocket_connection(self, websocket, path):
        """Handle WebSocket connections"""
        
        try:
            user_id = None
            async for message in websocket:
                data = json.loads(message)
                
                if data.get('type') == 'connect':
                    user_id = data.get('user_id')
                    self.websocket_connections[user_id] = websocket
                    
                elif data.get('type') == 'message':
                    # Process incoming message
                    streaming_message = await self.receive_message(data)
                    # Handle message processing here
                    
        except websockets.exceptions.ConnectionClosed:
            if user_id and user_id in self.websocket_connections:
                del self.websocket_connections[user_id]

class USSDStreamingInterface(BaseStreamingInterface):
    """USSD interface for feature phones"""
    
    def __init__(self):
        super().__init__(InterfaceType.USSD)
        self.session_storage = {}
        
    async def send_message(self, user_id: str, response: StreamingResponse) -> bool:
        """Send USSD response"""
        
        try:
            # USSD has strict length limits
            optimized_content = self.optimize_for_bandwidth(response.content)
            
            # Create USSD menu if quick replies exist
            if response.quick_replies:
                menu_content = optimized_content + "\n\n"
                for i, reply in enumerate(response.quick_replies[:9]):  # Max 9 options
                    menu_content += f"{i+1}. {reply}\n"
                menu_content += "0. Back"
                
                return menu_content
            
            return optimized_content
            
        except Exception as e:
            print(f"USSD send error: {e}")
            return False
    
    async def receive_message(self, raw_message: Dict[str, Any]) -> StreamingMessage:
        """Process USSD input"""
        
        return StreamingMessage(
            message_id=f"ussd_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            user_id=raw_message.get('phoneNumber', ''),
            interface_type=InterfaceType.USSD,
            message_type=MessageType.TEXT,
            content=raw_message.get('text', ''),
            metadata={
                "session_id": raw_message.get('sessionId'),
                "service_code": raw_message.get('serviceCode')
            },
            timestamp=datetime.now()
        )
    
    def optimize_for_bandwidth(self, content: str) -> str:
        """Extreme optimization for USSD"""
        
        # USSD has very strict limits
        optimized = content[:140]  # Strict character limit
        
        # Ultra-compact formatting
        optimized = optimized.replace('Business', 'Biz')
        optimized = optimized.replace('Assessment', 'Eval')
        optimized = optimized.replace('Recommendation', 'Rec')
        
        # Remove all formatting
        optimized = re.sub(r'[*#\n]', ' ', optimized)
        
        return optimized

class StreamingInterfaceManager:
    """Manages multiple streaming interfaces"""
    
    def __init__(self):
        self.interfaces = {
            InterfaceType.WHATSAPP: WhatsAppStreamingInterface(),
            InterfaceType.SMS: SMSStreamingInterface(),
            InterfaceType.VOICE: VoiceStreamingInterface(),
            InterfaceType.WEB_CHAT: WebChatStreamingInterface(),
            InterfaceType.USSD: USSDStreamingInterface()
        }
        
        self.response_adapters = {}
        self._initialize_adapters()
    
    def _initialize_adapters(self):
        """Initialize response adapters for each interface"""
        
        for interface_type, interface in self.interfaces.items():
            self.response_adapters[interface_type] = ResponseAdapter(interface)
    
    async def send_response(self, 
                          user_id: str,
                          interface_type: InterfaceType,
                          content: str,
                          response_type: str = "text",
                          quick_replies: List[str] = None) -> bool:
        """Send response through appropriate interface"""
        
        if interface_type not in self.interfaces:
            return False
        
        # Create streaming response
        response = StreamingResponse(
            response_id=f"resp_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            user_id=user_id,
            interface_type=interface_type,
            content=content,
            response_type=response_type,
            chunks=self._create_chunks(content, interface_type),
            metadata={},
            quick_replies=quick_replies or []
        )
        
        # Adapt response for interface
        adapted_response = self.response_adapters[interface_type].adapt_response(response)
        
        # Send through interface
        return await self.interfaces[interface_type].send_message(user_id, adapted_response)
    
    def _create_chunks(self, content: str, interface_type: InterfaceType) -> List[str]:
        """Create chunks for streaming response"""
        
        if interface_type == InterfaceType.WEB_CHAT:
            # For web chat, create word-by-word chunks for typing effect
            words = content.split()
            chunks = []
            current_chunk = ""
            
            for word in words:
                current_chunk += word + " "
                if len(current_chunk.split()) >= 5:  # 5 words per chunk
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
            
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            return chunks
        else:
            # For other interfaces, return single chunk
            return [content]

class ResponseAdapter:
    """Adapts responses for specific interfaces"""
    
    def __init__(self, interface: BaseStreamingInterface):
        self.interface = interface
    
    def adapt_response(self, response: StreamingResponse) -> StreamingResponse:
        """Adapt response for the specific interface"""
        
        # Optimize content for the interface
        adapted_content = self.interface.optimize_for_bandwidth(response.content)
        
        # Create adapted response
        adapted_response = StreamingResponse(
            response_id=response.response_id,
            user_id=response.user_id,
            interface_type=response.interface_type,
            content=adapted_content,
            response_type=response.response_type,
            chunks=[adapted_content],  # Single chunk for most interfaces
            metadata=response.metadata,
            quick_replies=response.quick_replies[:3] if response.quick_replies else []  # Limit options
        )
        
        return adapted_response

# Factory functions
def create_streaming_interface_manager():
    return StreamingInterfaceManager()

def create_whatsapp_interface():
    return WhatsAppStreamingInterface()

def create_web_chat_interface():
    return WebChatStreamingInterface()