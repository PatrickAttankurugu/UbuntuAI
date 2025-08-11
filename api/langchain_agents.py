"""
LangChain Agents for UbuntuAI
Specialized agents for Ghanaian startup ecosystem (fintech, agritech, healthtech)
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

# LangChain imports
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool, tool
from langchain_core.runnables import RunnableConfig

# LangSmith integration
try:
    from langsmith import trace
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False
    def trace(name):
        def decorator(func):
            return func
        return decorator

from config.settings import settings
from config.prompts import prompt_templates
from api.llm_providers import llm_manager

logger = logging.getLogger(__name__)

class GhanaBusinessTool(BaseTool):
    """Base tool for Ghanaian business operations"""
    
    def __init__(self, name: str, description: str, **kwargs):
        super().__init__(name=name, description=description, **kwargs)
        self.ghana_focus = True

@tool
def get_ghana_regulatory_info(sector: str, region: str = "Greater Accra") -> str:
    """
    Get regulatory information for a specific sector in Ghana
    
    Args:
        sector: The business sector (fintech, agritech, healthtech)
        region: The Ghanaian region (default: Greater Accra)
    
    Returns:
        Regulatory information for the specified sector and region
    """
    try:
        # This would integrate with actual regulatory databases
        # For now, return structured information based on sector
        
        regulatory_info = {
            "fintech": {
                "Bank of Ghana": "All fintech companies must register with Bank of Ghana",
                "GIPC": "Foreign investment registration required for international fintech",
                "GRA": "Tax registration and compliance required",
                "Data Protection": "Compliance with Data Protection Act 2012"
            },
            "agritech": {
                "MoFA": "Ministry of Food and Agriculture registration",
                "GSA": "Ghana Standards Authority certification for agricultural products",
                "EPA": "Environmental Protection Agency permits for large operations",
                "FDA": "Food and Drugs Authority for food processing"
            },
            "healthtech": {
                "FDA": "Food and Drugs Authority registration and approval",
                "MoH": "Ministry of Health compliance",
                "GHS": "Ghana Health Service partnership requirements",
                "Data Protection": "Patient data protection compliance"
            }
        }
        
        if sector.lower() not in regulatory_info:
            return f"Unknown sector: {sector}. Supported sectors: fintech, agritech, healthtech"
        
        info = regulatory_info[sector.lower()]
        result = f"Regulatory requirements for {sector} in {region}, Ghana:\n\n"
        
        for agency, requirement in info.items():
            result += f"• {agency}: {requirement}\n"
        
        result += f"\nNote: This information is for {region}. Requirements may vary by region."
        return result
        
    except Exception as e:
        logger.error(f"Error getting regulatory info: {e}")
        return f"Error retrieving regulatory information: {str(e)}"

@tool
def find_ghana_funding_opportunities(sector: str, stage: str = "seed") -> str:
    """
    Find funding opportunities for Ghanaian startups
    
    Args:
        sector: The business sector (fintech, agritech, healthtech)
        stage: Funding stage (idea, pre-seed, seed, series-a, etc.)
    
    Returns:
        Available funding opportunities for the specified sector and stage
    """
    try:
        # This would integrate with actual funding databases
        # For now, return structured information based on sector and stage
        
        funding_opportunities = {
            "fintech": {
                "idea": [
                    "Ghana Tech Lab - Innovation Hub Support",
                    "MEST Africa - Pre-incubation Program",
                    "Ghana Angel Investor Network - Early Stage Support"
                ],
                "pre-seed": [
                    "Ghana Enterprise Agency (GEA) - Startup Support",
                    "Kosmos Innovation Center - Agritech & Fintech",
                    "Impact Hub Accra - Incubation Program"
                ],
                "seed": [
                    "Ghana Venture Capital Trust Fund",
                    "Village Capital - Fintech Accelerator",
                    "MEST Africa - Seed Investment"
                ]
            },
            "agritech": {
                "idea": [
                    "MoFA - Agricultural Innovation Support",
                    "Ghana Tech Lab - Agritech Focus",
                    "Kosmos Innovation Center - Agritech Program"
                ],
                "pre-seed": [
                    "GEA - Agricultural Startup Support",
                    "USAID Feed the Future - Agritech Innovation",
                    "GIZ - Green Innovation Support"
                ],
                "seed": [
                    "Ghana Venture Capital Trust Fund - Agritech",
                    "Acumen - Agricultural Impact Investment",
                    "Root Capital - Agricultural Finance"
                ]
            },
            "healthtech": {
                "idea": [
                    "Ghana Health Service - Innovation Support",
                    "MoH - Digital Health Initiative",
                    "Ghana Tech Lab - Healthtech Focus"
                ],
                "pre-seed": [
                    "GEA - Healthtech Startup Support",
                    "USAID - Digital Health Innovation",
                    "WHO - Health Innovation Program"
                ],
                "seed": [
                    "Ghana Venture Capital Trust Fund - Healthtech",
                    "Acumen - Healthcare Impact Investment",
                    "Global Innovation Fund - Health Innovation"
                ]
            }
        }
        
        if sector.lower() not in funding_opportunities:
            return f"Unknown sector: {sector}. Supported sectors: fintech, agritech, healthtech"
        
        if stage.lower() not in funding_opportunities[sector.lower()]:
            return f"Unknown stage: {stage}. Supported stages: idea, pre-seed, seed"
        
        opportunities = funding_opportunities[sector.lower()][stage.lower()]
        
        result = f"Funding opportunities for {sector} startups at {stage} stage in Ghana:\n\n"
        for i, opportunity in enumerate(opportunities, 1):
            result += f"{i}. {opportunity}\n"
        
        result += f"\nNext steps:\n"
        result += f"1. Research each opportunity's specific requirements\n"
        result += f"2. Prepare your pitch deck and business plan\n"
        result += f"3. Contact the organizations directly\n"
        result += f"4. Network with other Ghanaian entrepreneurs in your sector"
        
        return result
        
    except Exception as e:
        logger.error(f"Error finding funding opportunities: {e}")
        return f"Error retrieving funding opportunities: {str(e)}"

@tool
def analyze_ghana_market_opportunity(sector: str, region: str = "Greater Accra") -> str:
    """
    Analyze market opportunities for a specific sector in Ghana
    
    Args:
        sector: The business sector (fintech, agritech, healthtech)
        region: The Ghanaian region to analyze
    
    Returns:
        Market analysis and opportunities for the specified sector and region
    """
    try:
        # This would integrate with actual market research databases
        # For now, return structured analysis based on sector and region
        
        market_analysis = {
            "fintech": {
                "market_size": "Ghana's fintech market is growing rapidly with mobile money adoption at 40%+",
                "key_opportunities": [
                    "Digital payments and mobile money solutions",
                    "Microfinance and lending platforms",
                    "Insurance technology (InsurTech)",
                    "Blockchain and cryptocurrency services",
                    "Financial inclusion solutions"
                ],
                "challenges": [
                    "Regulatory compliance complexity",
                    "Limited digital infrastructure in rural areas",
                    "Competition from established banks",
                    "Cybersecurity concerns"
                ],
                "regional_variations": {
                    "Greater Accra": "High digital adoption, competitive market",
                    "Ashanti": "Growing fintech adoption, moderate competition",
                    "Western": "Emerging market, lower competition",
                    "Northern": "Early stage, high growth potential"
                }
            },
            "agritech": {
                "market_size": "Ghana's agricultural sector contributes 20% to GDP with high tech adoption potential",
                "key_opportunities": [
                    "Precision farming and IoT solutions",
                    "Agricultural supply chain optimization",
                    "Crop monitoring and disease detection",
                    "Agricultural fintech and insurance",
                    "Sustainable farming technologies"
                ],
                "challenges": [
                    "Limited access to technology in rural areas",
                    "Seasonal nature of agriculture",
                    "Infrastructure limitations",
                    "Traditional farming practices resistance"
                ],
                "regional_variations": {
                    "Greater Accra": "Urban farming, high-tech adoption",
                    "Ashanti": "Mixed farming, moderate tech adoption",
                    "Western": "Cocoa farming, emerging tech adoption",
                    "Northern": "Subsistence farming, high growth potential"
                }
            },
            "healthtech": {
                "market_size": "Ghana's healthcare market is expanding with increasing digital adoption",
                "key_opportunities": [
                    "Telemedicine and remote healthcare",
                    "Electronic health records and management",
                    "Health monitoring and wearable devices",
                    "Pharmaceutical supply chain optimization",
                    "Mental health technology solutions"
                ],
                "challenges": [
                    "Limited healthcare infrastructure in rural areas",
                    "Regulatory approval processes",
                    "Data privacy and security concerns",
                    "Integration with existing healthcare systems"
                ],
                "regional_variations": {
                    "Greater Accra": "Advanced healthcare, high tech adoption",
                    "Ashanti": "Moderate healthcare, growing tech adoption",
                    "Western": "Basic healthcare, emerging tech adoption",
                    "Northern": "Limited healthcare, high growth potential"
                }
            }
        }
        
        if sector.lower() not in market_analysis:
            return f"Unknown sector: {sector}. Supported sectors: fintech, agritech, healthtech"
        
        analysis = market_analysis[sector.lower()]
        regional_info = analysis["regional_variations"].get(region, "Regional information not available")
        
        result = f"Market Analysis for {sector} in {region}, Ghana:\n\n"
        result += f"Market Overview:\n{analysis['market_size']}\n\n"
        
        result += f"Key Opportunities:\n"
        for i, opportunity in enumerate(analysis['key_opportunities'], 1):
            result += f"{i}. {opportunity}\n"
        
        result += f"\nChallenges:\n"
        for i, challenge in enumerate(analysis['challenges'], 1):
            result += f"{i}. {challenge}\n"
        
        result += f"\nRegional Context ({region}):\n{regional_info}\n\n"
        
        result += f"Recommendations:\n"
        result += f"1. Focus on solving regional-specific challenges\n"
        result += f"2. Partner with local organizations and communities\n"
        result += f"3. Consider regulatory requirements early\n"
        result += f"4. Build solutions that work with existing infrastructure"
        
        return result
        
    except Exception as e:
        logger.error(f"Error analyzing market opportunity: {e}")
        return f"Error analyzing market opportunity: {str(e)}"

@tool
def get_ghana_business_networking_events(sector: str, region: str = "Greater Accra") -> str:
    """
    Get information about business networking events in Ghana
    
    Args:
        sector: The business sector (fintech, agritech, healthtech)
        region: The Ghanaian region to focus on
    
    Returns:
        Upcoming networking events and opportunities
    """
    try:
        # This would integrate with actual event databases
        # For now, return structured information about regular events
        
        networking_events = {
            "fintech": [
                "Ghana Fintech Week - Annual conference in Accra",
                "Mobile Money Conference - Quarterly events",
                "Digital Banking Summit - Bi-annual conference",
                "Fintech Meetups - Monthly networking events",
                "Ghana Tech Summit - Annual tech conference"
            ],
            "agritech": [
                "Ghana Agritech Conference - Annual event",
                "Agricultural Innovation Summit - Bi-annual",
                "FarmTech Ghana - Quarterly workshops",
                "Agritech Meetups - Monthly networking",
                "Ghana Food Security Forum - Annual conference"
            ],
            "healthtech": [
                "Ghana Health Innovation Summit - Annual conference",
                "Digital Health Ghana - Quarterly workshops",
                "HealthTech Meetups - Monthly networking",
                "Ghana Medical Conference - Annual event",
                "Healthcare Innovation Forum - Bi-annual"
            ]
        }
        
        if sector.lower() not in networking_events:
            return f"Unknown sector: {sector}. Supported sectors: fintech, agritech, healthtech"
        
        events = networking_events[sector.lower()]
        
        result = f"Networking Events for {sector} in Ghana:\n\n"
        for i, event in enumerate(events, 1):
            result += f"{i}. {event}\n"
        
        result += f"\nGeneral Networking Opportunities:\n"
        result += f"• MEST Africa - Regular startup events and workshops\n"
        result += f"• Ghana Tech Lab - Community meetups and hackathons\n"
        result += f"• Impact Hub Accra - Networking events and workshops\n"
        result += f"• Kosmos Innovation Center - Sector-specific events\n"
        result += f"• Ghana Angel Investor Network - Investor networking events\n\n"
        
        result += f"Tips for Networking in Ghana:\n"
        result += f"1. Attend events regularly to build relationships\n"
        result += f"2. Join WhatsApp groups for your sector\n"
        result += f"3. Connect with local business associations\n"
        result += f"4. Participate in online communities and forums\n"
        result += f"5. Follow up with contacts after events"
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting networking events: {e}")
        return f"Error retrieving networking events: {str(e)}"

def create_ghana_business_agent(llm_provider: str = None) -> AgentExecutor:
    """
    Create a specialized agent for Ghanaian business operations
    
    Args:
        llm_provider: Specific LLM provider to use
    
    Returns:
        Configured agent executor for Ghanaian business operations
    """
    
    try:
        # Get the LLM
        llm = llm_manager.get_langchain_llm(llm_provider)
        
        # Define tools
        tools = [
            get_ghana_regulatory_info,
            find_ghana_funding_opportunities,
            analyze_ghana_market_opportunity,
            get_ghana_business_networking_events
        ]
        
        # Create the system prompt
        system_prompt = f"""You are UbuntuAI, a specialized AI assistant for the Ghanaian startup ecosystem. Your expertise covers fintech, agritech, and healthtech sectors across all 16 regions of Ghana.

Your mission is to help Ghanaian entrepreneurs navigate:
- Regulatory requirements and compliance
- Funding opportunities and investment
- Market analysis and opportunities
- Business networking and community building

CORE PRINCIPLES:
- Focus exclusively on Ghana and Ghanaian business context
- Provide accurate, up-to-date information about Ghana
- Consider regional differences across Ghana's 16 regions
- Offer practical, actionable advice for Ghanaian entrepreneurs
- Maintain awareness of Ghana's economic and cultural context

AVAILABLE TOOLS:
- get_ghana_regulatory_info: Get regulatory requirements for specific sectors and regions
- find_ghana_funding_opportunities: Find funding opportunities for specific sectors and stages
- analyze_ghana_market_opportunity: Analyze market opportunities for specific sectors and regions
- get_ghana_business_networking_events: Get information about networking events

RESPONSE GUIDELINES:
- Always provide Ghana-specific information and context
- Use the available tools to get accurate information
- Consider the user's sector and region when providing advice
- Offer practical next steps and actionable recommendations
- Cite specific Ghanaian organizations, programs, and regulations
- Acknowledge when information might be limited or outdated

Remember: You're here to empower Ghanaian entrepreneurs with knowledge and insights that can help them build successful, impactful businesses in Ghana's fintech, agritech, and healthtech sectors."""

        # Create the prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        # Create the agent
        agent = create_openai_functions_agent(llm, tools, prompt)
        
        # Create the executor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            max_iterations=5,
            early_stopping_method="generate",
            handle_parsing_errors=True
        )
        
        logger.info(f"Ghana business agent created successfully with {llm_provider or 'default'} provider")
        return agent_executor
        
    except Exception as e:
        logger.error(f"Error creating Ghana business agent: {e}")
        raise

def create_ghana_sector_specialist_agent(sector: str, llm_provider: str = None) -> AgentExecutor:
    """
    Create a specialized agent for a specific Ghanaian sector
    
    Args:
        sector: The sector to specialize in (fintech, agritech, healthtech)
        llm_provider: Specific LLM provider to use
    
    Returns:
        Configured agent executor for the specific sector
    """
    
    try:
        if sector.lower() not in settings.GHANA_STARTUP_SECTORS:
            raise ValueError(f"Invalid sector: {sector}. Must be one of {settings.GHANA_STARTUP_SECTORS}")
        
        # Get the LLM
        llm = llm_manager.get_langchain_llm(llm_provider)
        
        # Define sector-specific tools
        base_tools = [
            get_ghana_regulatory_info,
            find_ghana_funding_opportunities,
            analyze_ghana_market_opportunity,
            get_ghana_business_networking_events
        ]
        
        # Create sector-specific system prompt
        sector_prompts = {
            "fintech": """You are UbuntuAI, a specialized AI assistant for Ghana's fintech sector. You help Ghanaian fintech entrepreneurs navigate regulatory compliance, funding opportunities, and market development.

Focus areas:
- Bank of Ghana regulations and compliance
- Mobile money and digital payments
- Financial inclusion and innovation
- Fintech partnerships and collaborations
- Ghanaian fintech ecosystem development""",
            
            "agritech": """You are UbuntuAI, a specialized AI assistant for Ghana's agritech sector. You help Ghanaian agritech entrepreneurs leverage technology to transform agriculture.

Focus areas:
- Ministry of Food and Agriculture programs
- Agricultural technology adoption
- Sustainable farming practices
- Agricultural supply chain optimization
- Ghanaian agricultural innovation""",
            
            "healthtech": """You are UbuntuAI, a specialized AI assistant for Ghana's healthtech sector. You help Ghanaian healthtech entrepreneurs improve healthcare delivery through technology.

Focus areas:
- Food and Drugs Authority compliance
- Digital health solutions
- Healthcare accessibility
- Medical technology innovation
- Ghanaian healthcare system integration"""
        }
        
        system_prompt = f"""{sector_prompts[sector.lower()]}

Your mission is to provide expert guidance for {sector} entrepreneurs in Ghana, considering:
- Sector-specific regulatory requirements
- Industry-specific funding opportunities
- Market dynamics and competition
- Technology trends and innovation
- Regional variations across Ghana

Use the available tools to provide accurate, actionable information specific to the {sector} sector in Ghana."""

        # Create the prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        # Create the agent
        agent = create_openai_functions_agent(llm, base_tools, prompt)
        
        # Create the executor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=base_tools,
            verbose=True,
            max_iterations=5,
            early_stopping_method="generate",
            handle_parsing_errors=True
        )
        
        logger.info(f"Ghana {sector} specialist agent created successfully")
        return agent_executor
        
    except Exception as e:
        logger.error(f"Error creating Ghana {sector} specialist agent: {e}")
        raise

def get_available_ghana_agents() -> List[str]:
    """Get list of available Ghanaian business agents"""
    return ["general", "fintech", "agritech", "healthtech"]

def create_ghana_agent_by_type(agent_type: str = "general", llm_provider: str = None) -> AgentExecutor:
    """
    Create a Ghanaian business agent by type
    
    Args:
        agent_type: Type of agent to create (general, fintech, agritech, healthtech)
        llm_provider: Specific LLM provider to use
    
    Returns:
        Configured agent executor
    """
    
    try:
        if agent_type == "general":
            return create_ghana_business_agent(llm_provider)
        elif agent_type in settings.GHANA_STARTUP_SECTORS:
            return create_ghana_sector_specialist_agent(agent_type, llm_provider)
        else:
            raise ValueError(f"Invalid agent type: {agent_type}. Must be 'general' or one of {settings.GHANA_STARTUP_SECTORS}")
            
    except Exception as e:
        logger.error(f"Error creating Ghana agent by type: {e}")
        raise