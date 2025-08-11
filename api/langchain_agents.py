"""
Modern LangChain Agents for UbuntuAI
Implements agentic workflows with tool integration and self-reflection
"""

import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import json

# LangChain Agent imports
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.tools import Tool, BaseTool
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActSingleInputOutputParser

# LangGraph for complex workflows
try:
    from langgraph.graph import StateGraph, END
    from langgraph.prebuilt import ToolExecutor
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False

from config.settings import settings
from api.llm_providers import llm_manager
from api.rag_engine import get_rag_engine
from api.scoring_engine import create_scoring_engine
from knowledge_base.funding_database import funding_db
from knowledge_base.regulatory_info import regulatory_db

logger = logging.getLogger(__name__)

class BusinessAssessmentTool(BaseTool):
    """Tool for conducting business assessments"""
    
    name = "business_assessment"
    description = """Conduct a comprehensive business assessment for African startups. 
    Input should be business information including description, sector, team size, stage, etc.
    Returns detailed scoring and recommendations."""
    
    def __init__(self):
        super().__init__()
        try:
            self.scoring_engines = create_scoring_engine()
        except Exception as e:
            logger.warning(f"Scoring engines not available: {e}")
            self.scoring_engines = None
    
    def _run(self, business_info: str) -> str:
        """Run business assessment"""
        
        if not self.scoring_engines:
            return "Business assessment service is currently unavailable."
        
        try:
            # Parse business info (expecting JSON or structured text)
            if business_info.startswith('{'):
                business_data = json.loads(business_info)
            else:
                # Extract info from text
                business_data = self._parse_business_text(business_info)
            
            # Conduct assessment
            scorer = self.scoring_engines['startup_scorer']
            result = scorer.score_startup(business_data)
            
            # Format response
            response = f"""BUSINESS ASSESSMENT RESULTS

Overall Score: {result.overall_score:.2f}/1.0

Component Breakdown:"""
            
            for component, score in result.component_scores.items():
                status = "Strong" if score > 0.7 else "Moderate" if score > 0.5 else "Needs Work"
                response += f"\n• {component.replace('_', ' ').title()}: {score:.2f} ({status})"
            
            if result.risk_factors:
                response += f"\n\nKey Risk Factors:"
                for risk in result.risk_factors[:3]:
                    response += f"\n• {risk}"
            
            if result.recommendations:
                response += f"\n\nRecommendations:"
                for rec in result.recommendations[:3]:
                    response += f"\n• {rec}"
            
            return response
            
        except Exception as e:
            logger.error(f"Business assessment failed: {e}")
            return f"Error conducting business assessment: {str(e)}"
    
    def _parse_business_text(self, text: str) -> Dict[str, Any]:
        """Parse business information from text"""
        
        # Basic parsing - could be enhanced with NLP
        business_data = {
            'business_description': text,
            'sector': 'general',
            'team_size': 1,
            'product_stage': 'idea',
            'generating_revenue': False,
            'local_team_members': True,
            'mobile_first': True,
            'local_market_knowledge': True
        }
        
        # Extract sector if mentioned
        sectors = settings.BUSINESS_SECTORS
        text_lower = text.lower()
        for sector in sectors:
            if sector.lower() in text_lower:
                business_data['sector'] = sector
                break
        
        # Extract team size
        import re
        team_match = re.search(r'(\d+)\s+(?:team|people|members)', text_lower)
        if team_match:
            business_data['team_size'] = int(team_match.group(1))
        
        return business_data

class FundingSearchTool(BaseTool):
    """Tool for searching funding opportunities"""
    
    name = "funding_search"
    description = """Search for funding opportunities in Africa. 
    Input should include criteria like country, sector, stage, funding type.
    Returns relevant funding opportunities with details."""
    
    def _run(self, search_criteria: str) -> str:
        """Search for funding opportunities"""
        
        try:
            # Parse search criteria
            criteria = self._parse_search_criteria(search_criteria)
            
            # Search funding database
            opportunities = funding_db.search_funding(**criteria)
            
            if not opportunities:
                return "No funding opportunities found matching your criteria. Try broadening your search parameters."
            
            # Format response
            response = f"FUNDING OPPORTUNITIES ({len(opportunities)} found)\n"
            
            for i, opp in enumerate(opportunities[:5], 1):
                response += f"\n{i}. {opp['name']} ({opp.get('type', 'N/A')})"
                response += f"\n   Investment: {opp.get('typical_investment', 'N/A')}"
                response += f"\n   Focus: {', '.join(opp.get('focus_sectors', []))}"
                response += f"\n   Location: {opp.get('country', 'N/A')}"
                
                if opp.get('application_process'):
                    response += f"\n   Application: {opp['application_process']}"
                
                if opp.get('website'):
                    response += f"\n   Website: {opp['website']}"
                
                response += "\n"
            
            response += "\nNext Steps:"
            response += "\n• Research each funder's portfolio and investment thesis"
            response += "\n• Prepare pitch deck tailored to each opportunity"
            response += "\n• Seek warm introductions when possible"
            
            return response
            
        except Exception as e:
            logger.error(f"Funding search failed: {e}")
            return f"Error searching funding opportunities: {str(e)}"
    
    def _parse_search_criteria(self, criteria_text: str) -> Dict[str, Any]:
        """Parse search criteria from text"""
        
        criteria = {}
        text_lower = criteria_text.lower()
        
        # Extract country
        for country in settings.AFRICAN_COUNTRIES:
            if country.lower() in text_lower:
                criteria['country'] = country
                break
        
        # Extract sector
        for sector in settings.BUSINESS_SECTORS:
            if sector.lower() in text_lower:
                criteria['sector'] = sector
                break
        
        # Extract stage
        for stage in settings.FUNDING_STAGES:
            if stage.lower() in text_lower:
                criteria['stage'] = stage
                break
        
        # Extract funding type
        funding_types = ['vc firm', 'accelerator', 'grant', 'government']
        for ftype in funding_types:
            if ftype in text_lower:
                criteria['funding_type'] = ftype.title()
                break
        
        return criteria

class RegulatoryGuidanceTool(BaseTool):
    """Tool for regulatory guidance and compliance information"""
    
    name = "regulatory_guidance"
    description = """Provide regulatory guidance for business operations in Africa.
    Input should include country and type of guidance needed (registration, tax, compliance).
    Returns relevant regulatory information and requirements."""
    
    def _run(self, guidance_request: str) -> str:
        """Provide regulatory guidance"""
        
        try:
            # Parse request
            country, guidance_type = self._parse_guidance_request(guidance_request)
            
            if guidance_type == 'registration':
                return self._get_registration_guidance(country)
            elif guidance_type == 'tax':
                return self._get_tax_guidance(country)
            else:
                return self._get_general_guidance(country)
                
        except Exception as e:
            logger.error(f"Regulatory guidance failed: {e}")
            return f"Error providing regulatory guidance: {str(e)}"
    
    def _parse_guidance_request(self, request: str) -> tuple:
        """Parse guidance request"""
        
        # Extract country
        country = "Ghana"  # Default
        for c in settings.AFRICAN_COUNTRIES:
            if c.lower() in request.lower():
                country = c
                break
        
        # Extract guidance type
        guidance_type = "general"
        if any(word in request.lower() for word in ['register', 'registration', 'incorporate']):
            guidance_type = "registration"
        elif any(word in request.lower() for word in ['tax', 'taxation', 'revenue']):
            guidance_type = "tax"
        
        return country, guidance_type
    
    def _get_registration_guidance(self, country: str) -> str:
        """Get business registration guidance"""
        
        reg_info = regulatory_db.get_business_registration_info(country)
        
        if not reg_info:
            return f"Registration information for {country} is not available in our database."
        
        response = f"BUSINESS REGISTRATION - {country.upper()}\n"
        response += f"\nAuthority: {reg_info.get('registration_authority')}"
        response += f"\nOnline Portal: {reg_info.get('online_portal')}"
        response += f"\nProcessing Time: {reg_info.get('processing_time')}"
        response += f"\nCost: {reg_info.get('cost')}"
        
        if reg_info.get('required_documents'):
            response += f"\n\nRequired Documents:"
            for doc in reg_info['required_documents']:
                response += f"\n• {doc}"
        
        if reg_info.get('business_types'):
            response += f"\n\nBusiness Types Available:"
            for btype in reg_info['business_types']:
                response += f"\n• {btype}"
        
        return response
    
    def _get_tax_guidance(self, country: str) -> str:
        """Get tax guidance"""
        
        tax_info = regulatory_db.get_tax_information(country)
        
        if not tax_info:
            return f"Tax information for {country} is not available in our database."
        
        response = f"TAX INFORMATION - {country.upper()}\n"
        response += f"\nCorporate Tax Rate: {tax_info.get('corporate_tax_rate')}"
        response += f"\nVAT Rate: {tax_info.get('vat_rate')}"
        response += f"\nTax Authority: {tax_info.get('tax_authority')}"
        
        if tax_info.get('incentives'):
            response += f"\n\nTax Incentives:"
            for incentive in tax_info['incentives']:
                response += f"\n• {incentive}"
        
        return response
    
    def _get_general_guidance(self, country: str) -> str:
        """Get general regulatory guidance"""
        
        return regulatory_db.generate_country_guide(country)

class KnowledgeSearchTool(BaseTool):
    """Tool for searching the RAG knowledge base"""
    
    name = "knowledge_search"
    description = """Search the African business knowledge base for specific information.
    Input should be a specific question or topic to search for.
    Returns relevant information from the knowledge base."""
    
    def _run(self, query: str) -> str:
        """Search knowledge base"""
        
        try:
            rag_engine = get_rag_engine()
            
            if not rag_engine:
                return "Knowledge search is currently unavailable."
            
            # Perform RAG query
            result = rag_engine.query(
                question=query,
                conversation_history=[],
                user_context={}
            )
            
            # Format response
            response = result.get('answer', 'No relevant information found.')
            
            # Add source information if available
            sources = result.get('sources', [])
            if sources:
                response += f"\n\nSources:"
                for i, source in enumerate(sources[:3], 1):
                    source_info = source.get('metadata', {}).get('source', 'Unknown')
                    response += f"\n{i}. {source_info}"
            
            return response
            
        except Exception as e:
            logger.error(f"Knowledge search failed: {e}")
            return f"Error searching knowledge base: {str(e)}"

class ModernBusinessAgent:
    """
    Modern business agent using LangChain with tool integration
    """
    
    def __init__(self):
        self.llm = None
        self.tools = []
        self.agent = None
        self.agent_executor = None
        self._initialize()
    
    def _initialize(self):
        """Initialize the agent with LLM and tools"""
        
        try:
            # Get LLM
            if not llm_manager or not llm_manager.get_available_providers():
                logger.error("No LLM providers available for agent")
                return
            
            self.llm = llm_manager.get_langchain_llm()
            
            # Initialize tools
            self.tools = [
                BusinessAssessmentTool(),
                FundingSearchTool(),
                RegulatoryGuidanceTool(),
                KnowledgeSearchTool()
            ]
            
            # Create agent
            self._create_agent()
            
            logger.info("Modern business agent initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}")
    
    def _create_agent(self):
        """Create the ReAct agent"""
        
        # Define the prompt
        prompt = PromptTemplate.from_template("""
You are UbuntuAI, an expert AI assistant specializing in African business ecosystems and entrepreneurship.

You have access to the following tools:
{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}
""")
        
        # Create agent
        self.agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        # Create agent executor
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            max_iterations=settings.AGENT_MAX_ITERATIONS,
            handle_parsing_errors=True,
            return_intermediate_steps=True
        )
    
    def query(self, 
             question: str,
             user_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process query through the agent"""
        
        if not self.agent_executor:
            return {
                "answer": "Agent is not properly initialized. Please check system configuration.",
                "tools_used": [],
                "success": False
            }
        
        try:
            # Add user context to question if available
            enhanced_question = question
            if user_context:
                context_info = []
                for key in ['country', 'sector', 'business_stage']:
                    if user_context.get(key):
                        context_info.append(f"{key}: {user_context[key]}")
                
                if context_info:
                    enhanced_question = f"{question}\n\nUser Context: {' | '.join(context_info)}"
            
            # Execute agent
            result = self.agent_executor.invoke({
                "input": enhanced_question
            })
            
            # Extract information
            answer = result.get("output", "No answer generated.")
            intermediate_steps = result.get("intermediate_steps", [])
            
            # Extract tools used
            tools_used = []
            for step in intermediate_steps:
                if isinstance(step, tuple) and len(step) >= 1:
                    action = step[0]
                    if hasattr(action, 'tool'):
                        tools_used.append(action.tool)
            
            return {
                "answer": answer,
                "tools_used": list(set(tools_used)),
                "intermediate_steps": len(intermediate_steps),
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Agent query failed: {e}")
            return {
                "answer": f"I encountered an error processing your request: {str(e)}",
                "tools_used": [],
                "success": False
            }
    
    async def query_async(self,
                         question: str,
                         user_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Async version of query processing"""
        
        # For now, just call the sync version
        # In future, can implement true async agent execution
        return self.query(question, user_context)
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tools"""
        return [tool.name for tool in self.tools]
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get agent status information"""
        return {
            "initialized": bool(self.agent_executor),
            "llm_available": bool(self.llm),
            "tools_count": len(self.tools),
            "available_tools": self.get_available_tools(),
            "max_iterations": settings.AGENT_MAX_ITERATIONS
        }

class GraphBasedAgent:
    """
    Advanced agent using LangGraph for complex workflows
    """
    
    def __init__(self):
        self.available = LANGGRAPH_AVAILABLE
        self.graph = None
        self.tools = []
        
        if self.available:
            self._initialize_graph()
    
    def _initialize_graph(self):
        """Initialize LangGraph workflow"""
        
        if not LANGGRAPH_AVAILABLE:
            logger.warning("LangGraph not available")
            return
        
        try:
            # Define the state
            from typing import TypedDict
            
            class AgentState(TypedDict):
                messages: List[BaseMessage]
                next_action: str
                tools_used: List[str]
                context: Dict[str, Any]
            
            # Create workflow graph
            workflow = StateGraph(AgentState)
            
            # Define nodes
            workflow.add_node("analyze", self._analyze_query)
            workflow.add_node("search", self._search_knowledge)
            workflow.add_node("assess", self._assess_business)
            workflow.add_node("recommend", self._generate_recommendations)
            
            # Define edges
            workflow.add_edge("analyze", "search")
            workflow.add_edge("search", "assess")
            workflow.add_edge("assess", "recommend")
            workflow.add_edge("recommend", END)
            
            # Set entry point
            workflow.set_entry_point("analyze")
            
            # Compile graph
            self.graph = workflow.compile()
            
            logger.info("LangGraph agent initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize LangGraph agent: {e}")
            self.available = False
    
    def _analyze_query(self, state):
        """Analyze the user query"""
        # Implementation for query analysis
        return state
    
    def _search_knowledge(self, state):
        """Search knowledge base"""
        # Implementation for knowledge search
        return state
    
    def _assess_business(self, state):
        """Assess business if applicable"""
        # Implementation for business assessment
        return state
    
    def _generate_recommendations(self, state):
        """Generate final recommendations"""
        # Implementation for recommendations
        return state

# Factory function for creating agents
def create_ghana_business_agent() -> ModernBusinessAgent:
    """Create and return a modern Ghana business agent"""
    return ModernBusinessAgent()

def create_graph_agent() -> Optional[GraphBasedAgent]:
    """Create and return a graph-based agent if available"""
    if LANGGRAPH_AVAILABLE:
        return GraphBasedAgent()
    else:
        logger.warning("LangGraph not available - cannot create graph agent")
        return None

# Global agent instance
try:
    business_agent = create_ghana_business_agent()
    logger.info("Business agent created successfully")
except Exception as e:
    logger.error(f"Failed to create business agent: {e}")
    business_agent = None