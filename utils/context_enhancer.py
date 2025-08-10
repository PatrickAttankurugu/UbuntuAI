import re
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
from config.settings import settings

class BusinessContextEnhancer:
    def __init__(self):
        self.african_countries = set(settings.AFRICAN_COUNTRIES)
        self.business_sectors = set(settings.BUSINESS_SECTORS)
        self.funding_stages = set(settings.FUNDING_STAGES)
        
        self.country_synonyms = {
            "South Africa": ["RSA", "SA", "Republic of South Africa"],
            "Nigeria": ["Federal Republic of Nigeria"],
            "Kenya": ["Republic of Kenya"],
            "Ghana": ["Republic of Ghana"],
            "Egypt": ["Arab Republic of Egypt"],
            "Morocco": ["Kingdom of Morocco"],
            "Ethiopia": ["Federal Democratic Republic of Ethiopia"],
            "Tanzania": ["United Republic of Tanzania"],
            "Uganda": ["Republic of Uganda"],
            "Rwanda": ["Republic of Rwanda"]
        }
        
        self.sector_keywords = {
            "Fintech": ["financial technology", "digital payments", "mobile money", 
                       "digital banking", "blockchain", "cryptocurrency", "insurtech"],
            "Agritech": ["agriculture technology", "farming", "crop monitoring", 
                        "precision agriculture", "food security", "agricultural"],
            "Healthtech": ["health technology", "telemedicine", "digital health", 
                          "medical devices", "health services", "pharmaceuticals"],
            "Edtech": ["education technology", "e-learning", "online education", 
                      "educational software", "learning platforms"],
            "E-commerce": ["online retail", "e-commerce platform", "digital marketplace", 
                          "online shopping", "retail technology"],
            "Energy": ["renewable energy", "solar power", "clean energy", 
                      "energy storage", "grid solutions", "power generation"],
            "Logistics": ["supply chain", "delivery services", "transportation", 
                         "warehousing", "freight", "last-mile delivery"]
        }
        
        self.funding_keywords = {
            "Pre-seed": ["pre-seed funding", "initial funding", "founder funding"],
            "Seed": ["seed funding", "seed round", "seed investment"],
            "Series A": ["series a", "a round", "first institutional round"],
            "Series B": ["series b", "b round", "growth funding"],
            "Series C": ["series c", "c round", "expansion funding"],
            "Growth": ["growth capital", "growth funding", "scale funding"],
            "Grant": ["government grant", "development grant", "research grant"]
        }
    
    def extract_business_entities(self, text: str) -> Dict[str, List[str]]:
        entities = {
            'countries': [],
            'sectors': [],
            'companies': [],
            'funding_stages': [],
            'funding_amounts': [],
            'key_people': [],
            'organizations': [],
            'dates': []
        }
        
        text_lower = text.lower()
        
        # Extract countries
        for country in self.african_countries:
            if country.lower() in text_lower:
                entities['countries'].append(country)
        
        for country, synonyms in self.country_synonyms.items():
            for synonym in synonyms:
                if synonym.lower() in text_lower:
                    entities['countries'].append(country)
        
        # Extract sectors
        for sector, keywords in self.sector_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                entities['sectors'].append(sector)
        
        # Extract funding stages
        for stage, keywords in self.funding_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                entities['funding_stages'].append(stage)
        
        # Extract funding amounts
        amount_patterns = [
            r'\$\d{1,3}(?:,\d{3})*(?:\.\d{2})?(?:\s*(?:million|billion|M|B|K))?',
            r'\d+\s*(?:million|billion)\s*(?:USD|dollars?)',
            r'USD\s*\d{1,3}(?:,\d{3})*',
        ]
        
        for pattern in amount_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities['funding_amounts'].extend(matches)
        
        # Extract company names (basic pattern)
        company_patterns = [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Ltd|Limited|Inc|Corp|Company|Group|Technologies|Tech|Solutions|Capital|Ventures|Partners)\b',
            r'\b[A-Z][a-zA-Z]+(?:[A-Z][a-z]+)+\b',  # CamelCase names
        ]
        
        for pattern in company_patterns:
            matches = re.findall(pattern, text)
            entities['companies'].extend(matches[:10])  # Limit to avoid noise
        
        # Extract dates
        date_patterns = [
            r'\b\d{1,2}/\d{1,2}/\d{4}\b',
            r'\b\d{4}\b',
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b'
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities['dates'].extend(matches)
        
        # Extract key people (basic pattern for names with titles)
        people_pattern = r'\b(?:CEO|CTO|CFO|COO|Founder|Co-founder|President|Director|VP|Vice President)\s+([A-Z][a-z]+\s+[A-Z][a-z]+)'
        people_matches = re.findall(people_pattern, text, re.IGNORECASE)
        entities['key_people'].extend(people_matches)
        
        # Extract organizations
        org_patterns = [
            r'\b(?:World Bank|African Development Bank|AfDB|International Finance Corporation|IFC|USAID|DFID|GIZ|UN|United Nations|WHO|UNESCO)\b',
            r'\b[A-Z]{2,}(?:\s+[A-Z]{2,})*\b',  # Acronyms
        ]
        
        for pattern in org_patterns:
            matches = re.findall(pattern, text)
            entities['organizations'].extend(matches[:5])
        
        # Clean and deduplicate
        for key in entities:
            entities[key] = list(set(entities[key]))
        
        return entities
    
    def classify_content_type(self, text: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        text_lower = text.lower()
        
        classification = {
            'content_types': [],
            'confidence_scores': {},
            'primary_focus': None,
            'business_context': []
        }
        
        # Content type indicators
        indicators = {
            'funding_opportunity': [
                'apply for funding', 'funding available', 'grant program', 
                'investment opportunity', 'call for applications', 'deadline'
            ],
            'regulatory_info': [
                'business registration', 'compliance requirements', 'legal framework',
                'tax obligations', 'permit required', 'license application'
            ],
            'market_analysis': [
                'market size', 'growth rate', 'market trends', 'consumer behavior',
                'competitive landscape', 'market opportunity', 'industry analysis'
            ],
            'success_story': [
                'success story', 'case study', 'founded in', 'raised funding',
                'acquired by', 'ipo', 'unicorn', 'breakthrough'
            ],
            'news_article': [
                'breaking news', 'announced today', 'according to', 'reported',
                'press release', 'sources say', 'spokesperson'
            ],
            'research_report': [
                'study shows', 'research indicates', 'survey results', 'analysis reveals',
                'data suggests', 'findings show', 'methodology'
            ]
        }
        
        for content_type, keywords in indicators.items():
            matches = sum(1 for keyword in keywords if keyword in text_lower)
            if matches > 0:
                classification['content_types'].append(content_type)
                classification['confidence_scores'][content_type] = matches / len(keywords)
        
        # Determine primary focus
        if classification['confidence_scores']:
            primary_focus = max(classification['confidence_scores'], 
                              key=classification['confidence_scores'].get)
            classification['primary_focus'] = primary_focus
        
        # Business context classification
        business_contexts = []
        
        context_indicators = {
            'startup_ecosystem': ['startup', 'entrepreneur', 'innovation', 'incubator', 'accelerator'],
            'investment_climate': ['investment', 'capital', 'vc', 'venture', 'angel investor'],
            'regulatory_environment': ['regulation', 'policy', 'government', 'compliance', 'framework'],
            'market_dynamics': ['market', 'competition', 'demand', 'supply', 'customer'],
            'economic_indicators': ['gdp', 'inflation', 'economic growth', 'unemployment', 'trade'],
            'technology_adoption': ['digital', 'mobile', 'internet', 'technology', 'innovation']
        }
        
        for context, keywords in context_indicators.items():
            if any(keyword in text_lower for keyword in keywords):
                business_contexts.append(context)
        
        classification['business_context'] = business_contexts
        
        return classification
    
    def enhance_metadata(self, text: str, base_metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        enhanced_metadata = base_metadata.copy() if base_metadata else {}
        
        # Extract business entities
        entities = self.extract_business_entities(text)
        enhanced_metadata['extracted_entities'] = entities
        
        # Classify content
        classification = self.classify_content_type(text, base_metadata)
        enhanced_metadata.update(classification)
        
        # Add context scores
        context_scores = self.calculate_context_relevance(text)
        enhanced_metadata['context_relevance'] = context_scores
        
        # Add geographic focus
        if entities['countries']:
            enhanced_metadata['geographic_focus'] = entities['countries']
            enhanced_metadata['is_africa_focused'] = True
        else:
            enhanced_metadata['is_africa_focused'] = False
        
        # Add sector focus
        if entities['sectors']:
            enhanced_metadata['sector_focus'] = entities['sectors']
        
        # Add temporal relevance
        if entities['dates']:
            enhanced_metadata['temporal_references'] = entities['dates']
            enhanced_metadata['has_recent_info'] = self._has_recent_dates(entities['dates'])
        
        # Add business stage indicators
        stage_indicators = self._identify_business_stage(text)
        enhanced_metadata['business_stage_indicators'] = stage_indicators
        
        return enhanced_metadata
    
    def calculate_context_relevance(self, text: str) -> Dict[str, float]:
        text_lower = text.lower()
        word_count = len(text.split())
        
        relevance_scores = {}
        
        # African business relevance
        africa_keywords = ['africa', 'african'] + [c.lower() for c in self.african_countries]
        africa_matches = sum(1 for word in africa_keywords if word in text_lower)
        relevance_scores['african_business'] = min(africa_matches / word_count * 100, 1.0)
        
        # Startup ecosystem relevance
        startup_keywords = ['startup', 'entrepreneur', 'innovation', 'venture', 'funding']
        startup_matches = sum(1 for word in startup_keywords if word in text_lower)
        relevance_scores['startup_ecosystem'] = min(startup_matches / word_count * 100, 1.0)
        
        # Funding relevance
        funding_keywords = ['funding', 'investment', 'capital', 'grant', 'loan', 'vc']
        funding_matches = sum(1 for word in funding_keywords if word in text_lower)
        relevance_scores['funding'] = min(funding_matches / word_count * 100, 1.0)
        
        # Regulatory relevance
        regulatory_keywords = ['regulation', 'compliance', 'legal', 'policy', 'government']
        regulatory_matches = sum(1 for word in regulatory_keywords if word in text_lower)
        relevance_scores['regulatory'] = min(regulatory_matches / word_count * 100, 1.0)
        
        return relevance_scores
    
    def _has_recent_dates(self, dates: List[str]) -> bool:
        current_year = datetime.now().year
        
        for date_str in dates:
            if re.search(r'\b(202[0-9]|201[89])\b', date_str):
                year = int(re.search(r'\b(20\d{2})\b', date_str).group(1))
                if current_year - year <= 3:  # Within last 3 years
                    return True
        
        return False
    
    def _identify_business_stage(self, text: str) -> List[str]:
        text_lower = text.lower()
        
        stage_indicators = {
            'early_stage': ['idea stage', 'concept', 'pre-revenue', 'prototype', 'mvp'],
            'growth_stage': ['scaling', 'expansion', 'growth', 'series a', 'series b'],
            'mature_stage': ['established', 'profitable', 'market leader', 'ipo ready'],
            'seeking_funding': ['fundraising', 'seeking investment', 'looking for capital'],
            'recently_funded': ['just raised', 'closed round', 'announced funding']
        }
        
        identified_stages = []
        
        for stage, keywords in stage_indicators.items():
            if any(keyword in text_lower for keyword in keywords):
                identified_stages.append(stage)
        
        return identified_stages
    
    def create_search_enhancements(self, query: str, user_context: Dict[str, Any] = None) -> Dict[str, Any]:
        enhancements = {
            'expanded_query': query,
            'filter_criteria': {},
            'boost_factors': {},
            'semantic_expansions': []
        }
        
        # Extract entities from query
        query_entities = self.extract_business_entities(query)
        
        # Expand query with synonyms and related terms
        expanded_terms = []
        
        # Add country-specific expansions
        for country in query_entities['countries']:
            if country in self.country_synonyms:
                expanded_terms.extend(self.country_synonyms[country])
        
        # Add sector-specific expansions
        for sector in query_entities['sectors']:
            if sector in self.sector_keywords:
                expanded_terms.extend(self.sector_keywords[sector][:3])  # Limit to top 3
        
        if expanded_terms:
            enhancements['expanded_query'] = f"{query} {' '.join(expanded_terms)}"
        
        # Create filter criteria
        if query_entities['countries']:
            enhancements['filter_criteria']['countries'] = query_entities['countries']
        
        if query_entities['sectors']:
            enhancements['filter_criteria']['sectors'] = query_entities['sectors']
        
        if query_entities['funding_stages']:
            enhancements['filter_criteria']['funding_stages'] = query_entities['funding_stages']
        
        # Add user context if available
        if user_context:
            if 'preferred_countries' in user_context:
                enhancements['boost_factors']['countries'] = user_context['preferred_countries']
            
            if 'business_stage' in user_context:
                enhancements['boost_factors']['business_stage'] = user_context['business_stage']
        
        return enhancements

context_enhancer = BusinessContextEnhancer()