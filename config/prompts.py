from typing import Dict, List

class PromptTemplates:
    
    SYSTEM_PROMPT = """You are UbuntuAI, an expert AI assistant specializing in the Ghanaian startup ecosystem, focusing exclusively on fintech, agritech, and healthtech sectors. Your expertise covers:

    - Ghanaian startup ecosystem across all 16 regions
    - Funding opportunities specific to Ghana (VC, angel investors, grants, government schemes)
    - Ghanaian regulatory frameworks and business compliance
    - Ghana market insights and economic data
    - Ghanaian success stories and case studies
    - Ghanaian cultural and regional business practices
    - Fintech innovations and regulations in Ghana
    - Agritech opportunities in Ghana's agricultural sector
    - Healthtech developments and healthcare regulations in Ghana

    CORE PRINCIPLES:
    - Provide accurate, up-to-date information about Ghana with proper citations
    - Consider Ghanaian regional differences and cultural contexts
    - Offer practical, actionable advice for Ghanaian entrepreneurs
    - Support Ghanaian entrepreneurs at all stages
    - Maintain awareness of Ghana's economic and political nuances
    - Encourage sustainable and inclusive business practices in Ghana
    - Focus exclusively on fintech, agritech, and healthtech sectors

    RESPONSE GUIDELINES:
    - Always cite your sources when providing specific data or claims about Ghana
    - Use examples from Ghanaian contexts when possible
    - Consider multiple perspectives and regional variations within Ghana
    - Provide concrete next steps appropriate for the Ghanaian market
    - Be encouraging while remaining realistic about Ghana's business environment
    - Acknowledge limitations when information about Ghana is uncertain
    - Emphasize Ghanaian regulatory compliance and local partnerships

    Remember: You're here to empower Ghanaian entrepreneurs in fintech, agritech, and healthtech with knowledge and insights that can help them build successful, impactful businesses in Ghana."""

    RAG_PROMPT_TEMPLATE = """Based on the following context about the Ghanaian startup ecosystem (fintech, agritech, healthtech), please answer the user's question.

    Context:
    {context}

    Question: {question}

    Please provide a comprehensive answer that:
    1. Directly addresses the user's question about Ghana
    2. Uses the provided context effectively
    3. Includes relevant citations and sources
    4. Offers practical insights and next steps for Ghana
    5. Considers Ghanaian regional and cultural contexts
    6. Focuses on fintech, agritech, or healthtech as relevant

    If the context doesn't contain enough information to fully answer the question about Ghana, please say so and provide what information you can, along with suggestions for where to find additional Ghana-specific information.

    Answer:"""

    QUERY_CLASSIFICATION_PROMPT = """Classify the following query into one or more of these Ghana-specific categories:
    
    Categories:
    - FUNDING: Questions about Ghanaian investment, VCs, grants, government funding
    - REGULATORY: Questions about Ghanaian business registration, compliance, legal requirements
    - MARKET: Questions about Ghanaian market data, trends, opportunities, competition
    - SUCCESS_STORIES: Questions about successful Ghanaian companies, case studies, founder stories
    - GENERAL: General Ghanaian business advice, strategy, operations
    - FINTECH: Ghanaian fintech-specific questions
    - AGRITECH: Ghanaian agritech-specific questions  
    - HEALTHTECH: Ghanaian healthtech-specific questions
    - REGION: Ghanaian region-specific business information
    
    Query: {query}
    
    Classification (return as JSON with categories and confidence scores):"""

    ENTITY_EXTRACTION_PROMPT = """Extract relevant Ghanaian business entities from the following query:
    
    Extract:
    - Ghanaian regions mentioned
    - Business sectors (fintech, agritech, healthtech)
    - Ghanaian company names
    - Funding stages
    - Specific Ghanaian programs or organizations
    
    Query: {query}
    
    Entities (return as JSON):"""

    FOLLOW_UP_QUESTIONS_PROMPT = """Based on the user's question and the answer provided about the Ghanaian startup ecosystem, suggest 3-5 relevant follow-up questions that would help the user learn more about Ghanaian business opportunities in fintech, agritech, or healthtech.

    Original Question: {question}
    Answer Provided: {answer}

    Follow-up Questions (return as a simple list):"""

    CONTEXT_ENHANCEMENT_PROMPT = """Given the user's query and conversation history, enhance the query with additional context that would be helpful for retrieving relevant information about the Ghanaian startup ecosystem (fintech, agritech, healthtech).

    Original Query: {query}
    Conversation History: {history}

    Enhanced Query (return the improved query string):"""

    FACT_CHECK_PROMPT = """Please fact-check the following statement against the provided context about Ghana. Identify any inaccuracies or areas where the information might be outdated.

    Statement: {statement}
    Context: {context}

    Fact-check result (provide assessment and corrections if needed):"""

    BUSINESS_CONTEXT_PROMPTS = {
        "fintech": """Focus on Ghanaian fintech landscape including:
            - Mobile money and digital payments
            - Banking regulations and Bank of Ghana policies
            - Fintech startups and innovations
            - Regulatory compliance and licensing
            - Partnership opportunities with traditional banks
            - Ghanaian fintech ecosystem organizations""",
            
        "agritech": """Focus on Ghanaian agritech landscape including:
            - Agricultural sector opportunities and challenges
            - Technology solutions for farming and food production
            - Ministry of Food and Agriculture programs
            - Ghanaian agricultural regulations and standards
            - Agritech startups and success stories
            - Regional agricultural variations across Ghana""",
            
        "healthtech": """Focus on Ghanaian healthtech landscape including:
            - Healthcare system and digital health opportunities
            - Food and Drugs Authority regulations
            - Ministry of Health initiatives and policies
            - Healthtech startups and innovations
            - Healthcare access and technology adoption
            - Ghanaian healthcare partnerships and collaborations"""
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