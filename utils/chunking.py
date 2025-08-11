"""
Advanced Text Chunking for UbuntuAI using LangChain Text Splitters
Supports multiple chunking strategies with business context awareness
"""

import re
import json
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod

# LangChain Text Splitters
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    SpacyTextSplitter,
    CharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    HTMLHeaderTextSplitter,
    PythonCodeTextSplitter,
    LatexTextSplitter
)
from langchain_core.documents import Document

# Semantic chunking (if available)
try:
    from langchain_experimental.text_splitter import SemanticChunker
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False

from config.settings import settings

logger = logging.getLogger(__name__)

@dataclass
class ChunkingResult:
    """Result of document chunking"""
    chunks: List[Document]
    metadata: Dict[str, Any]
    statistics: Dict[str, Any]

class BaseChunkingStrategy(ABC):
    """Abstract base class for chunking strategies"""
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        self.splitter = None
        self._initialize_splitter()
    
    @abstractmethod
    def _initialize_splitter(self):
        """Initialize the text splitter"""
        pass
    
    @abstractmethod
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        pass
    
    def split_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Document]:
        """Split text into chunks"""
        document = Document(page_content=text, metadata=metadata or {})
        return self.split_documents([document])

class RecursiveChunkingStrategy(BaseChunkingStrategy):
    """Recursive character-based chunking strategy"""
    
    def _initialize_splitter(self):
        """Initialize recursive character text splitter"""
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
            separators=[
                "\n\n",  # Double newline (paragraphs)
                "\n",    # Single newline
                ". ",    # Sentence endings
                "! ",    # Exclamations
                "? ",    # Questions
                "; ",    # Semicolons
                ", ",    # Commas
                " ",     # Spaces
                ""       # Characters
            ]
        )
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents using recursive strategy"""
        return self.splitter.split_documents(documents)

class TokenBasedChunkingStrategy(BaseChunkingStrategy):
    """Token-based chunking strategy"""
    
    def _initialize_splitter(self):
        """Initialize token-based text splitter"""
        # Convert chunk size from characters to approximate tokens
        token_chunk_size = self.chunk_size // 4  # Rough estimate: 1 token ≈ 4 chars
        token_overlap = self.chunk_overlap // 4
        
        self.splitter = TokenTextSplitter(
            chunk_size=token_chunk_size,
            chunk_overlap=token_overlap
        )
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents using token-based strategy"""
        return self.splitter.split_documents(documents)

class SemanticChunkingStrategy(BaseChunkingStrategy):
    """Semantic chunking strategy using embeddings"""
    
    def _initialize_splitter(self):
        """Initialize semantic chunker"""
        if not SEMANTIC_AVAILABLE:
            logger.warning("Semantic chunking not available - falling back to recursive")
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            return
        
        try:
            from utils.embeddings import embedding_manager
            embeddings = embedding_manager.get_langchain_embeddings()
            
            if embeddings:
                self.splitter = SemanticChunker(
                    embeddings=embeddings,
                    breakpoint_threshold_type="percentile",
                    breakpoint_threshold_amount=95
                )
            else:
                logger.warning("No embeddings available for semantic chunking")
                self.splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap
                )
                
        except Exception as e:
            logger.error(f"Failed to initialize semantic chunker: {e}")
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents using semantic strategy"""
        return self.splitter.split_documents(documents)

class MarkdownAwareChunkingStrategy(BaseChunkingStrategy):
    """Markdown-aware chunking strategy"""
    
    def _initialize_splitter(self):
        """Initialize markdown-aware splitter"""
        # First split by headers
        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
                ("####", "Header 4"),
            ]
        )
        
        # Then split by content
        self.content_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents using markdown-aware strategy"""
        all_chunks = []
        
        for doc in documents:
            # First split by headers
            header_chunks = self.header_splitter.split_text(doc.page_content)
            
            # Convert to documents with combined metadata
            header_docs = []
            for chunk in header_chunks:
                combined_metadata = {**doc.metadata, **chunk.metadata}
                header_docs.append(Document(
                    page_content=chunk.page_content,
                    metadata=combined_metadata
                ))
            
            # Then split by content size
            final_chunks = self.content_splitter.split_documents(header_docs)
            all_chunks.extend(final_chunks)
        
        return all_chunks

class BusinessContextChunkingStrategy(BaseChunkingStrategy):
    """Business context-aware chunking strategy"""
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        super().__init__(chunk_size, chunk_overlap)
        self.business_keywords = self._load_business_keywords()
        self.entity_patterns = self._load_entity_patterns()
    
    def _initialize_splitter(self):
        """Initialize business-aware splitter"""
        self.base_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=[
                "\n\n",
                "\n",
                ". ",
                "! ",
                "? ",
                "; ",
                " ",
                ""
            ]
        )
    
    def _load_business_keywords(self) -> Dict[str, List[str]]:
        """Load business context keywords"""
        return {
            'funding': ['investment', 'funding', 'capital', 'vc', 'venture', 'grant', 'loan'],
            'regulatory': ['registration', 'compliance', 'legal', 'regulation', 'permit', 'license'],
            'market': ['market', 'industry', 'competition', 'customer', 'demand', 'supply'],
            'success': ['success', 'growth', 'scale', 'expansion', 'exit', 'acquisition'],
            'technology': ['fintech', 'agritech', 'healthtech', 'edtech', 'technology', 'digital']
        }
    
    def _load_entity_patterns(self) -> Dict[str, str]:
        """Load entity recognition patterns"""
        return {
            'countries': r'\b(?:' + '|'.join(settings.AFRICAN_COUNTRIES) + r')\b',
            'amounts': r'\$[\d,]+(?:\.\d{2})?[MBK]?|\d+\s*(?:million|billion)',
            'companies': r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Ltd|Limited|Inc|Corp|Company|Technologies|Tech|Solutions)\b',
            'dates': r'\b\d{1,2}/\d{1,2}/\d{4}\b|\b\d{4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b'
        }
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents with business context awareness"""
        all_chunks = []
        
        for doc in documents:
            # First split using base splitter
            base_chunks = self.base_splitter.split_documents([doc])
            
            # Enhance each chunk with business context
            for chunk in base_chunks:
                enhanced_chunk = self._enhance_chunk_metadata(chunk)
                all_chunks.append(enhanced_chunk)
        
        return all_chunks
    
    def _enhance_chunk_metadata(self, chunk: Document) -> Document:
        """Enhance chunk with business context metadata"""
        content = chunk.page_content.lower()
        enhanced_metadata = chunk.metadata.copy()
        
        # Identify business contexts
        contexts = []
        for context, keywords in self.business_keywords.items():
            if any(keyword in content for keyword in keywords):
                contexts.append(context)
        
        enhanced_metadata['business_contexts'] = contexts
        
        # Extract entities
        entities = {}
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, chunk.page_content, re.IGNORECASE)
            if matches:
                entities[entity_type] = list(set(matches))  # Remove duplicates
        
        enhanced_metadata['extracted_entities'] = entities
        
        # Calculate relevance scores
        relevance_scores = self._calculate_relevance_scores(content)
        enhanced_metadata['relevance_scores'] = relevance_scores
        
        return Document(
            page_content=chunk.page_content,
            metadata=enhanced_metadata
        )
    
    def _calculate_relevance_scores(self, content: str) -> Dict[str, float]:
        """Calculate relevance scores for different business contexts"""
        word_count = len(content.split())
        scores = {}
        
        for context, keywords in self.business_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in content)
            scores[context] = min(matches / word_count * 100, 1.0) if word_count > 0 else 0.0
        
        return scores

class AdaptiveChunkingStrategy(BaseChunkingStrategy):
    """Adaptive chunking strategy that adjusts based on content type"""
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        super().__init__(chunk_size, chunk_overlap)
        self.strategies = self._initialize_strategies()
    
    def _initialize_splitter(self):
        """Initialize adaptive splitter (not used directly)"""
        pass
    
    def _initialize_strategies(self) -> Dict[str, BaseChunkingStrategy]:
        """Initialize different chunking strategies"""
        return {
            'markdown': MarkdownAwareChunkingStrategy(self.chunk_size, self.chunk_overlap),
            'business': BusinessContextChunkingStrategy(self.chunk_size, self.chunk_overlap),
            'semantic': SemanticChunkingStrategy(self.chunk_size, self.chunk_overlap),
            'recursive': RecursiveChunkingStrategy(self.chunk_size, self.chunk_overlap),
            'token': TokenBasedChunkingStrategy(self.chunk_size, self.chunk_overlap)
        }
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents using adaptive strategy selection"""
        all_chunks = []
        
        for doc in documents:
            # Determine best strategy for this document
            strategy = self._select_strategy(doc)
            
            # Split using selected strategy
            chunks = strategy.split_documents([doc])
            
            # Add strategy info to metadata
            for chunk in chunks:
                chunk.metadata['chunking_strategy'] = strategy.__class__.__name__
            
            all_chunks.extend(chunks)
        
        return all_chunks
    
    def _select_strategy(self, document: Document) -> BaseChunkingStrategy:
        """Select the best chunking strategy for a document"""
        content = document.page_content
        metadata = document.metadata
        
        # Check for markdown content
        if self._is_markdown_content(content):
            return self.strategies['markdown']
        
        # Check for business content
        if self._is_business_content(content, metadata):
            return self.strategies['business']
        
        # Check if semantic chunking would be beneficial
        if len(content) > 2000 and self._is_complex_content(content):
            return self.strategies['semantic']
        
        # Default to recursive
        return self.strategies['recursive']
    
    def _is_markdown_content(self, content: str) -> bool:
        """Check if content contains markdown formatting"""
        markdown_patterns = [
            r'^#{1,6}\s',  # Headers
            r'\*\*.*?\*\*',  # Bold
            r'\*.*?\*',  # Italics
            r'```.*?```',  # Code blocks
            r'`.*?`',  # Inline code
            r'\[.*?\]\(.*?\)',  # Links
        ]
        
        for pattern in markdown_patterns:
            if re.search(pattern, content, re.MULTILINE | re.DOTALL):
                return True
        
        return False
    
    def _is_business_content(self, content: str, metadata: Dict[str, Any]) -> bool:
        """Check if content is business-focused"""
        # Check metadata first
        if metadata.get('type') in ['funding_opportunity', 'regulatory_info', 'success_story']:
            return True
        
        # Check content for business keywords
        business_keywords = [
            'funding', 'investment', 'startup', 'entrepreneur', 'business',
            'market', 'revenue', 'capital', 'venture', 'company', 'industry'
        ]
        
        content_lower = content.lower()
        keyword_count = sum(1 for keyword in business_keywords if keyword in content_lower)
        
        return keyword_count >= 3
    
    def _is_complex_content(self, content: str) -> bool:
        """Check if content would benefit from semantic chunking"""
        # Look for complex document structures
        indicators = [
            len(content.split('\n\n')) > 5,  # Multiple paragraphs
            len(re.findall(r'[.!?]+', content)) > 10,  # Multiple sentences
            len(set(content.split())) / len(content.split()) > 0.7  # High vocabulary diversity
        ]
        
        return sum(indicators) >= 2

class ModernChunkingManager:
    """
    Modern chunking manager that coordinates different strategies
    """
    
    def __init__(self):
        self.strategies = {
            'recursive': RecursiveChunkingStrategy(),
            'token': TokenBasedChunkingStrategy(),
            'semantic': SemanticChunkingStrategy(),
            'markdown': MarkdownAwareChunkingStrategy(),
            'business': BusinessContextChunkingStrategy(),
            'adaptive': AdaptiveChunkingStrategy()
        }
        
        self.default_strategy = settings.CHUNKING_STRATEGY
        logger.info(f"Chunking manager initialized with strategy: {self.default_strategy}")
    
    def chunk_documents(self, 
                       documents: List[Document],
                       strategy: str = None,
                       **kwargs) -> ChunkingResult:
        """Chunk documents using specified strategy"""
        
        strategy_name = strategy or self.default_strategy
        
        if strategy_name not in self.strategies:
            logger.warning(f"Unknown strategy {strategy_name}, using recursive")
            strategy_name = 'recursive'
        
        chunking_strategy = self.strategies[strategy_name]
        
        # Perform chunking
        chunks = chunking_strategy.split_documents(documents)
        
        # Calculate statistics
        statistics = self._calculate_statistics(documents, chunks)
        
        # Create metadata
        metadata = {
            'strategy_used': strategy_name,
            'original_documents': len(documents),
            'total_chunks': len(chunks),
            'timestamp': json.dumps(datetime.now().isoformat())
        }
        
        return ChunkingResult(
            chunks=chunks,
            metadata=metadata,
            statistics=statistics
        )
    
    def chunk_text(self, 
                   text: str,
                   metadata: Dict[str, Any] = None,
                   strategy: str = None,
                   **kwargs) -> ChunkingResult:
        """Chunk a single text string"""
        
        document = Document(page_content=text, metadata=metadata or {})
        return self.chunk_documents([document], strategy, **kwargs)
    
    def _calculate_statistics(self, 
                            original_docs: List[Document],
                            chunks: List[Document]) -> Dict[str, Any]:
        """Calculate chunking statistics"""
        
        # Original document stats
        original_lengths = [len(doc.page_content) for doc in original_docs]
        original_total = sum(original_lengths)
        
        # Chunk stats
        chunk_lengths = [len(chunk.page_content) for chunk in chunks]
        chunk_total = sum(chunk_lengths)
        
        # Calculate overlap
        overlap_chars = max(0, chunk_total - original_total)
        overlap_percentage = (overlap_chars / original_total * 100) if original_total > 0 else 0
        
        return {
            'original_documents': len(original_docs),
            'total_chunks': len(chunks),
            'avg_original_length': sum(original_lengths) / len(original_lengths) if original_lengths else 0,
            'avg_chunk_length': sum(chunk_lengths) / len(chunk_lengths) if chunk_lengths else 0,
            'min_chunk_length': min(chunk_lengths) if chunk_lengths else 0,
            'max_chunk_length': max(chunk_lengths) if chunk_lengths else 0,
            'total_original_chars': original_total,
            'total_chunk_chars': chunk_total,
            'overlap_chars': overlap_chars,
            'overlap_percentage': round(overlap_percentage, 2),
            'compression_ratio': round(original_total / chunk_total, 3) if chunk_total > 0 else 0
        }
    
    def get_strategy_info(self, strategy: str = None) -> Dict[str, Any]:
        """Get information about chunking strategies"""
        
        if strategy:
            if strategy in self.strategies:
                strategy_obj = self.strategies[strategy]
                return {
                    'name': strategy,
                    'class': strategy_obj.__class__.__name__,
                    'chunk_size': strategy_obj.chunk_size,
                    'chunk_overlap': strategy_obj.chunk_overlap,
                    'available': True
                }
            else:
                return {'name': strategy, 'available': False}
        
        # Return info for all strategies
        return {
            'default_strategy': self.default_strategy,
            'available_strategies': list(self.strategies.keys()),
            'strategy_details': {
                name: {
                    'class': strategy.__class__.__name__,
                    'chunk_size': strategy.chunk_size,
                    'chunk_overlap': strategy.chunk_overlap
                }
                for name, strategy in self.strategies.items()
            }
        }
    
    def set_default_strategy(self, strategy: str) -> bool:
        """Set the default chunking strategy"""
        if strategy in self.strategies:
            self.default_strategy = strategy
            logger.info(f"Default chunking strategy set to: {strategy}")
            return True
        else:
            logger.error(f"Unknown strategy: {strategy}")
            return False

# Global chunking manager instance
try:
    chunking_manager = ModernChunkingManager()
    logger.info("Modern chunking manager initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize chunking manager: {e}")
    chunking_manager = None

# For backward compatibility
chunker = chunking_manager