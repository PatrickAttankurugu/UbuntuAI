import re
from typing import List, Dict, Any, Optional, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config.settings import settings

class BusinessContextChunker:
    def __init__(self, 
                 chunk_size: int = None, 
                 chunk_overlap: int = None):
        
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        
        # Use character-based token counting for Gemini (approximately 1 token = 4 characters)
        self.char_to_token_ratio = 4
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self._count_tokens,
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", " ", ""]
        )
        
        self.business_keywords = {
            'funding': ['investment', 'vc', 'venture capital', 'angel', 'seed', 'series', 
                       'grant', 'loan', 'funding', 'capital', 'investor', 'raise'],
            'regulatory': ['regulation', 'compliance', 'legal', 'law', 'registration', 
                          'license', 'permit', 'tax', 'policy', 'government'],
            'market': ['market', 'industry', 'sector', 'competition', 'demand', 'supply',
                      'consumer', 'customer', 'trend', 'analysis', 'size', 'growth'],
            'success': ['success', 'unicorn', 'exit', 'acquisition', 'ipo', 'growth',
                       'scale', 'expansion', 'achievement', 'milestone']
        }
    
    def _count_tokens(self, text: str) -> int:
        """Estimate token count for Gemini (approximately 1 token = 4 characters)"""
        return len(text) // self.char_to_token_ratio
    
    def _identify_business_context(self, text: str) -> List[str]:
        text_lower = text.lower()
        contexts = []
        
        for context, keywords in self.business_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                contexts.append(context)
        
        return contexts
    
    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        entities = {
            'countries': [],
            'companies': [],
            'sectors': [],
            'amounts': [],
            'dates': []
        }
        
        for country in settings.AFRICAN_COUNTRIES:
            if country.lower() in text.lower():
                entities['countries'].append(country)
        
        for sector in settings.BUSINESS_SECTORS:
            if sector.lower() in text.lower():
                entities['sectors'].append(sector)
        
        amount_pattern = r'\$[\d,]+(?:\.\d{2})?[MBK]?|\d+\s*million|\d+\s*billion'
        amounts = re.findall(amount_pattern, text, re.IGNORECASE)
        entities['amounts'].extend(amounts)
        
        date_pattern = r'\b\d{1,2}/\d{1,2}/\d{4}\b|\b\d{4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b'
        dates = re.findall(date_pattern, text, re.IGNORECASE)
        entities['dates'].extend(dates)
        
        company_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Ltd|Limited|Inc|Corporation|Corp|Company|Co|Group|Holdings|Technologies|Tech|Solutions|Systems|Services|Ventures|Capital|Partners|Fund|Investments)\b'
        companies = re.findall(company_pattern, text)
        entities['companies'].extend(companies[:5])
        
        return entities
    
    def _create_chunk_metadata(self, 
                              chunk: str, 
                              source_metadata: Dict[str, Any], 
                              chunk_index: int) -> Dict[str, Any]:
        metadata = source_metadata.copy()
        metadata.update({
            'chunk_index': chunk_index,
            'chunk_size': self._count_tokens(chunk),
            'business_contexts': self._identify_business_context(chunk),
            'entities': self._extract_entities(chunk),
            'chunk_summary': self._create_chunk_summary(chunk)
        })
        return metadata
    
    def _create_chunk_summary(self, chunk: str) -> str:
        sentences = chunk.split('. ')
        if len(sentences) > 2:
            return f"{sentences[0]}. {sentences[1]}."
        return sentences[0] if sentences else chunk[:200]
    
    def _preserve_context_boundaries(self, chunks: List[str]) -> List[str]:
        enhanced_chunks = []
        
        for i, chunk in enumerate(chunks):
            enhanced_chunk = chunk
            
            if i > 0:
                prev_chunk_end = chunks[i-1].split()[-10:]
                if prev_chunk_end:
                    context_bridge = " ".join(prev_chunk_end)
                    enhanced_chunk = f"[Previous context: ...{context_bridge}]\n\n{enhanced_chunk}"
            
            if i < len(chunks) - 1:
                next_chunk_start = chunks[i+1].split()[:10]
                if next_chunk_start:
                    context_bridge = " ".join(next_chunk_start)
                    enhanced_chunk = f"{enhanced_chunk}\n\n[Next context: {context_bridge}...]"
            
            enhanced_chunks.append(enhanced_chunk)
        
        return enhanced_chunks
    
    def chunk_document(self, 
                      text: str, 
                      metadata: Dict[str, Any] = None,
                      preserve_context: bool = True) -> List[Dict[str, Any]]:
        if not text.strip():
            return []
        
        base_metadata = metadata or {}
        chunks = self.text_splitter.split_text(text)
        
        if preserve_context:
            chunks = self._preserve_context_boundaries(chunks)
        
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = self._create_chunk_metadata(chunk, base_metadata, i)
            
            processed_chunks.append({
                'content': chunk,
                'metadata': chunk_metadata,
                'token_count': self._count_tokens(chunk)
            })
        
        return processed_chunks
    
    def chunk_by_section(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        section_patterns = [
            r'\n#{1,3}\s+([^\n]+)',  # Markdown headers
            r'\n([A-Z][^.!?]*[:.]\s*$)',  # Title-like lines
            r'\n\s*(\d+\.?\s+[A-Z][^\n]*)',  # Numbered sections
            r'\n\s*([A-Z][A-Z\s]+:)',  # ALL CAPS headers
        ]
        
        sections = []
        current_section = ""
        current_header = ""
        
        lines = text.split('\n')
        
        for line in lines:
            is_header = False
            
            for pattern in section_patterns:
                match = re.search(pattern, f'\n{line}')
                if match:
                    if current_section.strip():
                        sections.append({
                            'header': current_header,
                            'content': current_section.strip()
                        })
                    current_header = match.group(1).strip()
                    current_section = ""
                    is_header = True
                    break
            
            if not is_header:
                current_section += line + "\n"
        
        if current_section.strip():
            sections.append({
                'header': current_header,
                'content': current_section.strip()
            })
        
        all_chunks = []
        
        for section in sections:
            section_text = f"{section['header']}\n\n{section['content']}" if section['header'] else section['content']
            section_metadata = (metadata or {}).copy()
            section_metadata['section_header'] = section['header']
            
            chunks = self.chunk_document(section_text, section_metadata, preserve_context=False)
            all_chunks.extend(chunks)
        
        return all_chunks
    
    def adaptive_chunking(self, 
                         text: str, 
                         metadata: Dict[str, Any] = None,
                         target_contexts: List[str] = None) -> List[Dict[str, Any]]:
        base_chunks = self.chunk_document(text, metadata)
        
        if not target_contexts:
            return base_chunks
        
        enhanced_chunks = []
        
        for chunk_data in base_chunks:
            chunk = chunk_data['content']
            chunk_metadata = chunk_data['metadata']
            
            chunk_contexts = chunk_metadata.get('business_contexts', [])
            context_overlap = set(chunk_contexts) & set(target_contexts)
            
            if context_overlap:
                if self._count_tokens(chunk) > self.chunk_size * 0.8:
                    sub_chunks = self.text_splitter.split_text(chunk)
                    for i, sub_chunk in enumerate(sub_chunks):
                        sub_metadata = chunk_metadata.copy()
                        sub_metadata['sub_chunk_index'] = i
                        sub_metadata['parent_chunk'] = chunk_data['chunk_index'] if 'chunk_index' in chunk_metadata else 0
                        
                        enhanced_chunks.append({
                            'content': sub_chunk,
                            'metadata': sub_metadata,
                            'token_count': self._count_tokens(sub_chunk)
                        })
                else:
                    enhanced_chunks.append(chunk_data)
            else:
                enhanced_chunks.append(chunk_data)
        
        return enhanced_chunks
    
    def get_chunking_stats(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not chunks:
            return {}
        
        token_counts = [chunk['token_count'] for chunk in chunks]
        contexts = []
        entities = []
        
        for chunk in chunks:
            metadata = chunk.get('metadata', {})
            contexts.extend(metadata.get('business_contexts', []))
            chunk_entities = metadata.get('entities', {})
            for entity_type, entity_list in chunk_entities.items():
                entities.extend(entity_list)
        
        return {
            'total_chunks': len(chunks),
            'avg_tokens_per_chunk': sum(token_counts) / len(token_counts),
            'min_tokens': min(token_counts),
            'max_tokens': max(token_counts),
            'total_tokens': sum(token_counts),
            'unique_contexts': len(set(contexts)),
            'unique_entities': len(set(entities)),
            'context_distribution': {ctx: contexts.count(ctx) for ctx in set(contexts)}
        }

chunker = BusinessContextChunker()