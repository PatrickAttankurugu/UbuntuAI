from typing import Dict, List

class PromptTemplates:
    
    SYSTEM_PROMPT = """You are UbuntuAI, an expert AI assistant specializing in African business ecosystems, entrepreneurship, and startup knowledge. Your expertise covers:

    - African startup ecosystems across 50+ countries
    - Funding opportunities (VC, angel investors, grants, government schemes)
    - Regulatory frameworks and business compliance
    - Market insights and economic data
    - Success stories and case studies
    - Cultural and regional business practices

    CORE PRINCIPLES:
    - Provide accurate, up-to-date information with proper citations
    - Consider regional differences and cultural contexts
    - Offer practical, actionable advice
    - Support entrepreneurs at all stages
    - Maintain awareness of economic and political nuances
    - Encourage sustainable and inclusive business practices

    RESPONSE GUIDELINES:
    - Always cite your sources when providing specific data or claims
    - Use examples from African contexts when possible
    - Consider multiple perspectives and regional variations
    - Provide concrete next steps when appropriate
    - Be encouraging while remaining realistic
    - Acknowledge limitations when information is uncertain

    Remember: You're here to empower African entrepreneurs with knowledge and insights that can help them build successful, impactful businesses."""

    RAG_PROMPT_TEMPLATE = """Based on the following context about African business ecosystems, please answer the user's question.

    Context:
    {context}

    Question: {question}

    Please provide a comprehensive answer that:
    1. Directly addresses the user's question
    2. Uses the provided context effectively
    3. Includes relevant citations and sources
    4. Offers practical insights and next steps
    5. Considers regional and cultural contexts

    If the context doesn't contain enough information to fully answer the question, please say so and provide what information you can, along with suggestions for where to find additional information.

    Answer:"""

    QUERY_CLASSIFICATION_PROMPT = """Classify the following query into one or more of these categories:
    
    Categories:
    - FUNDING: Questions about investment, VCs, grants, government funding
    - REGULATORY: Questions about business registration, compliance, legal requirements
    - MARKET: Questions about market data, trends, opportunities, competition
    - SUCCESS_STORIES: Questions about successful companies, case studies, founder stories
    - GENERAL: General business advice, strategy, operations
    - SECTOR: Industry-specific questions (fintech, agritech, healthtech, etc.)
    - COUNTRY: Country-specific business information
    
    Query: {query}
    
    Classification (return as JSON with categories and confidence scores):"""

    ENTITY_EXTRACTION_PROMPT = """Extract relevant business entities from the following query:
    
    Extract:
    - Countries (African countries mentioned)
    - Business sectors/industries
    - Company names
    - Funding stages
    - Specific programs or organizations
    
    Query: {query}
    
    Entities (return as JSON):"""

    FOLLOW_UP_QUESTIONS_PROMPT = """Based on the user's question and the answer provided, suggest 3-5 relevant follow-up questions that would help the user learn more about African business opportunities.

    Original Question: {question}
    Answer Provided: {answer}

    Follow-up Questions (return as a simple list):"""

    CONTEXT_ENHANCEMENT_PROMPT = """Given the user's query and conversation history, enhance the query with additional context that would be helpful for retrieving relevant information about African business ecosystems.

    Original Query: {query}
    Conversation History: {history}

    Enhanced Query (return the improved query string):"""

    FACT_CHECK_PROMPT = """Please fact-check the following statement against the provided context. Identify any inaccuracies or areas where the information might be outdated.

    Statement: {statement}
    Context: {context}

    Fact-check result (provide assessment and corrections if needed):"""

    BUSINESS_CONTEXT_PROMPTS = {
        "funding": """Focus on African funding landscape including:
        - Local and international VCs active in Africa
        - Government and development finance institutions
        - Angel investor networks and criteria
        - Grant opportunities and application processes
        - Funding stages and typical amounts
        - Success rates and timelines""",
        
        "regulatory": """Consider African regulatory environment:
        - Country-specific business registration processes
        - Tax implications and incentives for startups
        - Import/export requirements and procedures
        - Digital business compliance requirements
        - Intellectual property protection
        - Labor law and employment regulations""",
        
        "market": """Analyze African market dynamics:
        - Economic indicators and growth trends
        - Consumer behavior and preferences
        - Infrastructure and connectivity status
        - Competition landscape and market gaps
        - Distribution channels and partnerships
        - Regional trade agreements and opportunities""",
        
        "success_stories": """Highlight African success examples:
        - Unicorn companies and their growth strategies
        - Sector-specific success stories
        - Founder backgrounds and journeys
        - Key pivot points and lessons learned
        - Market entry and scaling strategies
        - Impact on local ecosystems"""
    }

    @staticmethod
    def get_context_aware_prompt(category: str, base_prompt: str) -> str:
        """Enhance prompt with category-specific context"""
        context_addition = PromptTemplates.BUSINESS_CONTEXT_PROMPTS.get(category, "")
        if context_addition:
            return f"{base_prompt}\n\nAdditional Context for {category.title()}:\n{context_addition}"
        return base_prompt

    @staticmethod
    def format_rag_prompt(context: str, question: str, category: str = None) -> str:
        """Format RAG prompt with context and question"""
        base_prompt = PromptTemplates.RAG_PROMPT_TEMPLATE.format(
            context=context,
            question=question
        )
        
        if category:
            return PromptTemplates.get_context_aware_prompt(category, base_prompt)
        return base_prompt

    @staticmethod
    def get_conversation_starter_prompts() -> List[str]:
        """Return empty list - no conversation starters"""
        return []

prompt_templates = PromptTemplates()