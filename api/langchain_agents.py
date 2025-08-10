from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import BaseTool, StructuredTool
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.callbacks import BaseCallbackHandler
from typing import Dict, List, Any, Optional, Type, Callable
from pydantic import BaseModel, Field
import json
import asyncio
from datetime import datetime
import logging

from api.rag_engine import rag_engine
from api.scoring_engine import create_scoring_engine
from knowledge_base.funding_database import funding_db
from knowledge_base.regulatory_info import regulatory_db
from config.settings import settings

logger = logging.getLogger(__name__)

# Pydantic models for tool inputs
class BusinessAssessmentInput(BaseModel):
    business_description: str = Field(description="Description of the business")
    sector: str = Field(description="Business sector/industry")
    team_size: int = Field(description="Number of team members")
    stage: str = Field(description="Business development stage")
    location: str = Field(description="Business location")

class FundingSearchInput(BaseModel):
    sector: str = Field(description="Business sector for funding search")
    stage: str = Field(description="Funding stage needed")
    amount_range: str = Field(description="Funding amount range needed")
    country: str = Field(default="Ghana", description="Country for funding search")

class RegulatoryQueryInput(BaseModel):
    query: str = Field(description="Regulatory or legal question")
    country: str = Field(default="Ghana", description="Country for regulatory information")
    business_type: str = Field(default="general", description="Type of business")

class MarketResearchInput(BaseModel):
    sector: str = Field(description="Industry sector for research")
    location: str = Field(description="Geographic location for market research")
    research_type: str = Field(description="Type of market research needed")

class ImpactTrackingInput(BaseModel):
    business_description: str = Field(description="Business description for impact assessment")
    target_beneficiaries: str = Field(description="Who benefits from this business")
    impact_metrics: List[str] = Field(description="Metrics to track for impact")

# Custom callback handler for agent monitoring
class AgentCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.actions = []
        self.observations = []
        self.thoughts = []

    def on_agent_action(self, action, **kwargs):
        self.actions.append({
            'tool': action.tool,
            'input': action.tool_input,
            'timestamp': datetime.now().isoformat()
        })

    def on_agent_finish(self, finish, **kwargs):
        logger.info(f"Agent finished with {len(self.actions)} actions")

# Business Tools
class BusinessAssessmentTool(BaseTool):
    name = "business_assessment"
    description = "Assess startup readiness and provide detailed scoring with Ghana-specific insights"
    args_schema: Type[BaseModel] = BusinessAssessmentInput

    def _run(self, business_description: str, sector: str, team_size: int, 
             stage: str, location: str) -> str:
        try:
            scoring_engines = create_scoring_engine()
            scorer = scoring_engines['startup_scorer']
            
            # Transform input to scoring format
            scoring_data = {
                'business_description': business_description,
                'sector': sector.lower(),
                'team_size': team_size,
                'product_stage': self._map_stage(stage),
                'local_team_members': location.lower() in ['ghana', 'accra', 'kumasi'],
                'mobile_first': True,  # Assume true for Ghana
                'local_market_knowledge': True,
                'target_market': f"{sector} customers in {location}"
            }
            
            result = scorer.score_startup(scoring_data)
            
            # Format comprehensive response
            assessment = f"""🏢 BUSINESS ASSESSMENT REPORT

📊 OVERALL SCORE: {result.overall_score}/1.0
Confidence Level: {result.confidence:.2f}

📈 COMPONENT BREAKDOWN:"""
            
            for component, score in result.component_scores.items():
                emoji = "🟢" if score > 0.7 else "🟡" if score > 0.5 else "🔴"
                assessment += f"\n{emoji} {component.replace('_', ' ').title()}: {score:.2f}/1.0"
            
            if result.risk_factors:
                assessment += "\n\n⚠️ RISK FACTORS:"
                for risk in result.risk_factors[:3]:  # Top 3 risks
                    assessment += f"\n• {risk}"
            
            if result.recommendations:
                assessment += "\n\n💡 KEY RECOMMENDATIONS:"
                for rec in result.recommendations[:3]:  # Top 3 recommendations
                    assessment += f"\n• {rec}"
            
            # Ghana-specific insights
            assessment += f"\n\n🇬🇭 GHANA MARKET INSIGHTS:"
            assessment += f"\n• {sector.title()} sector shows good potential in Ghana"
            assessment += f"\n• Consider mobile-first approach for {location} market"
            assessment += f"\n• Local partnerships recommended for market entry"
            
            return assessment
            
        except Exception as e:
            return f"Error in business assessment: {str(e)}"

    def _map_stage(self, stage: str) -> str:
        stage_mapping = {
            'idea': 'idea',
            'early': 'prototype', 
            'startup': 'prototype',
            'growing': 'beta',
            'growth': 'beta',
            'established': 'launched',
            'mature': 'launched'
        }
        return stage_mapping.get(stage.lower(), 'idea')

class FundingSearchTool(BaseTool):
    name = "funding_search"
    description = "Search for relevant funding opportunities based on business criteria"
    args_schema: Type[BaseModel] = FundingSearchInput

    def _run(self, sector: str, stage: str, amount_range: str, country: str) -> str:
        try:
            # Search using internal funding database
            search_filters = {
                'sector': sector,
                'stage': stage,
                'country': country
            }
            
            # Remove None values
            search_filters = {k: v for k, v in search_filters.items() if v}
            
            funding_opportunities = funding_db.search_funding(**search_filters)
            
            if not funding_opportunities:
                # Fallback to RAG search
                query = f"funding opportunities for {sector} {stage} startups in {country}"
                rag_response = rag_engine.query(query, user_context={'country': country})
                return rag_response.get('answer', 'No specific funding opportunities found.')
            
            # Format funding results
            funding_report = f"💰 FUNDING OPPORTUNITIES ({len(funding_opportunities)} found)\n\n"
            
            for i, opp in enumerate(funding_opportunities[:5], 1):  # Top 5
                funding_report += f"{i}. **{opp['name']}** ({opp.get('type', 'N/A')})\n"
                funding_report += f"   💵 Investment: {opp.get('typical_investment', 'N/A')}\n"
                funding_report += f"   🎯 Focus: {', '.join(opp.get('focus_sectors', []))}\n"
                funding_report += f"   📍 Location: {opp.get('country', 'N/A')}\n"
                
                if opp.get('application_process'):
                    funding_report += f"   📝 Application: {opp['application_process']}\n"
                
                if opp.get('website'):
                    funding_report += f"   🌐 Website: {opp['website']}\n"
                
                funding_report += "\n"
            
            # Add next steps
            funding_report += "🎯 NEXT STEPS:\n"
            funding_report += "1. Research each funder's portfolio companies\n"
            funding_report += "2. Prepare pitch deck and business plan\n"
            funding_report += "3. Get warm introductions when possible\n"
            funding_report += "4. Apply to 3-5 relevant opportunities\n"
            
            return funding_report
            
        except Exception as e:
            return f"Error searching funding: {str(e)}"

class RegulatoryGuidanceTool(BaseTool):
    name = "regulatory_guidance"
    description = "Provide regulatory and legal guidance for business setup and compliance"
    args_schema: Type[BaseModel] = RegulatoryQueryInput

    def _run(self, query: str, country: str, business_type: str) -> str:
        try:
            # Use RAG engine for regulatory queries
            enhanced_query = f"{query} business registration compliance {country} {business_type}"
            rag_response = rag_engine.query(enhanced_query, user_context={'country': country})
            
            regulatory_info = rag_response.get('answer', '')
            
            # Add structured guidance from internal database
            if country.lower() == 'ghana':
                reg_info = regulatory_db.get_business_registration_info('Ghana')
                if reg_info:
                    regulatory_info += f"\n\n📋 GHANA BUSINESS REGISTRATION:\n"
                    regulatory_info += f"Authority: {reg_info.get('registration_authority', 'N/A')}\n"
                    regulatory_info += f"Portal: {reg_info.get('online_portal', 'N/A')}\n"
                    regulatory_info += f"Timeline: {reg_info.get('processing_time', 'N/A')}\n"
                    regulatory_info += f"Cost: {reg_info.get('cost', 'N/A')}\n"
                
                # Add tax information
                tax_info = regulatory_db.get_tax_information('Ghana')
                if tax_info:
                    regulatory_info += f"\n💰 TAX INFORMATION:\n"
                    regulatory_info += f"Corporate Tax: {tax_info.get('corporate_tax_rate', 'N/A')}\n"
                    regulatory_info += f"VAT Rate: {tax_info.get('vat_rate', 'N/A')}\n"
            
            return regulatory_info
            
        except Exception as e:
            return f"Error getting regulatory guidance: {str(e)}"

class MarketResearchTool(BaseTool):
    name = "market_research"
    description = "Provide market research insights and methodologies for African markets"
    args_schema: Type[BaseModel] = MarketResearchInput

    def _run(self, sector: str, location: str, research_type: str) -> str:
        try:
            # Use RAG for market insights
            query = f"market research {sector} {location} {research_type} analysis"
            rag_response = rag_engine.query(query, user_context={'country': location})
            
            market_info = rag_response.get('answer', '')
            
            # Add structured research methodology
            research_guide = f"\n\n📊 MARKET RESEARCH METHODOLOGY:\n\n"
            
            if research_type.lower() in ['customer', 'demand']:
                research_guide += "🎯 CUSTOMER RESEARCH:\n"
                research_guide += "1. Conduct 15-20 customer interviews\n"
                research_guide += "2. Create online surveys (Google Forms)\n"
                research_guide += "3. Observe customer behavior in natural settings\n"
                research_guide += "4. Join relevant WhatsApp/Facebook groups\n"
                research_guide += "5. Test willingness to pay with MVP\n\n"
            
            if research_type.lower() in ['competition', 'competitive']:
                research_guide += "🏢 COMPETITIVE ANALYSIS:\n"
                research_guide += "1. Identify direct and indirect competitors\n"
                research_guide += "2. Analyze their pricing strategies\n"
                research_guide += "3. Study their marketing approaches\n"
                research_guide += "4. Assess their strengths and weaknesses\n"
                research_guide += "5. Find market gaps and opportunities\n\n"
            
            # Ghana-specific research tips
            if location.lower() in ['ghana', 'accra', 'kumasi']:
                research_guide += "🇬🇭 GHANA-SPECIFIC TIPS:\n"
                research_guide += f"• Visit local markets in {location}\n"
                research_guide += "• Connect with trade associations\n"
                research_guide += "• Use Ghana Statistical Service data\n"
                research_guide += "• Consider mobile money adoption rates\n"
                research_guide += "• Factor in seasonal spending patterns\n"
            
            return market_info + research_guide
            
        except Exception as e:
            return f"Error conducting market research: {str(e)}"

class ImpactTrackingTool(BaseTool):
    name = "impact_tracking"
    description = "Design impact measurement frameworks for social enterprises"
    args_schema: Type[BaseModel] = ImpactTrackingInput

    def _run(self, business_description: str, target_beneficiaries: str, 
             impact_metrics: List[str]) -> str:
        try:
            # Create MERL (Monitoring, Evaluation, Reporting, Learning) framework
            framework = f"📊 IMPACT MEASUREMENT FRAMEWORK\n\n"
            
            framework += f"🎯 BUSINESS: {business_description}\n"
            framework += f"👥 BENEFICIARIES: {target_beneficiaries}\n\n"
            
            framework += "📈 KEY PERFORMANCE INDICATORS (KPIs):\n"
            
            # Standard impact metrics
            standard_metrics = {
                'reach': 'Number of people reached/served',
                'employment': 'Jobs created (direct and indirect)',
                'income': 'Income increase for beneficiaries',
                'access': 'Improved access to services/products',
                'efficiency': 'Cost/time savings for users',
                'quality': 'Quality of life improvements',
                'environment': 'Environmental impact reduction',
                'skills': 'Skills/knowledge transferred'
            }
            
            # Add user-specified metrics
            for metric in impact_metrics:
                if metric.lower() in standard_metrics:
                    framework += f"• {standard_metrics[metric.lower()]}\n"
                else:
                    framework += f"• {metric}\n"
            
            framework += "\n📊 MEASUREMENT METHODOLOGY:\n"
            framework += "1. Baseline Assessment (before intervention)\n"
            framework += "2. Regular Monitoring (monthly/quarterly)\n"
            framework += "3. Outcome Evaluation (6-12 months)\n"
            framework += "4. Impact Assessment (1-2 years)\n"
            framework += "5. Learning and Adaptation (ongoing)\n\n"
            
            framework += "🔧 DATA COLLECTION TOOLS:\n"
            framework += "• Mobile surveys (KoBo Toolbox, Google Forms)\n"
            framework += "• WhatsApp surveys for rural areas\n"
            framework += "• Focus group discussions\n"
            framework += "• Financial records analysis\n"
            framework += "• Photo documentation\n"
            framework += "• Third-party verification\n\n"
            
            framework += "📊 REPORTING FREQUENCY:\n"
            framework += "• Weekly: Operational metrics\n"
            framework += "• Monthly: Financial and reach metrics\n"
            framework += "• Quarterly: Outcome indicators\n"
            framework += "• Annually: Impact assessment\n\n"
            
            # Ghana-specific considerations
            framework += "🇬🇭 GHANA CONTEXT CONSIDERATIONS:\n"
            framework += "• Use local languages for surveys\n"
            framework += "• Account for seasonal variations\n"
            framework += "• Consider informal economy impacts\n"
            framework += "• Partner with local organizations for credibility\n"
            framework += "• Use SMS/WhatsApp for remote data collection\n"
            
            return framework
            
        except Exception as e:
            return f"Error creating impact framework: {str(e)}"

# Main Agent Orchestrator
class GhanaBusinessAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.3,
            openai_api_key=settings.OPENAI_API_KEY
        )
        
        # Initialize tools
        self.tools = [
            BusinessAssessmentTool(),
            FundingSearchTool(),
            RegulatoryGuidanceTool(),
            MarketResearchTool(),
            ImpactTrackingTool()
        ]
        
        # Memory for conversation context
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=10  # Keep last 10 exchanges
        )
        
        # System prompt optimized for Ghana business context
        self.system_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are UbuntuAI, an expert AI assistant specializing in African business ecosystems, with deep expertise in Ghana's entrepreneurial landscape.

CORE CAPABILITIES:
- Business assessment and readiness scoring
- Funding opportunity identification and guidance
- Regulatory compliance and business registration
- Market research and competitive analysis
- Impact measurement for social enterprises

GHANA SPECIALIZATION:
- Deep understanding of Ghana's business environment
- Knowledge of local regulations, funding sources, and market dynamics
- Cultural sensitivity and local language awareness
- Mobile-first and low-resource optimization
- Rural and informal economy considerations

APPROACH:
1. Always consider the specific Ghanaian context
2. Provide practical, actionable advice
3. Use available tools to provide comprehensive analysis
4. Consider resource constraints and mobile accessibility
5. Encourage sustainable and inclusive business practices

TOOL USAGE:
- Use business_assessment for startup evaluation
- Use funding_search for investment opportunities
- Use regulatory_guidance for legal compliance
- Use market_research for market insights
- Use impact_tracking for social enterprise metrics

Always think step-by-step and use multiple tools when comprehensive analysis is needed."""),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessage(content="{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        # Create agent
        self.agent = create_openai_tools_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.system_prompt
        )
        
        # Create executor
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            max_iterations=5,
            handle_parsing_errors=True
        )

    async def process_query(self, query: str, user_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process user query using agent workflow"""
        try:
            callback = AgentCallbackHandler()
            
            # Add user context to query if provided
            enhanced_query = query
            if user_context:
                context_str = ", ".join([f"{k}: {v}" for k, v in user_context.items() if v])
                enhanced_query = f"{query}\n\nUser Context: {context_str}"
            
            # Execute agent
            result = await self.agent_executor.ainvoke(
                {"input": enhanced_query},
                callbacks=[callback]
            )
            
            return {
                "answer": result["output"],
                "actions_taken": callback.actions,
                "thoughts": callback.thoughts,
                "tools_used": [action["tool"] for action in callback.actions],
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Agent execution error: {str(e)}")
            return {
                "answer": f"I encountered an error processing your request: {str(e)}. Please try rephrasing your question.",
                "actions_taken": [],
                "thoughts": [],
                "tools_used": [],
                "success": False
            }

    def process_query_sync(self, query: str, user_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Synchronous version of process_query"""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.process_query(query, user_context))

    def get_available_workflows(self) -> List[Dict[str, str]]:
        """Get list of available agent workflows"""
        return [
            {
                "name": "Business Assessment",
                "description": "Comprehensive startup readiness evaluation",
                "trigger": "assess my business, evaluate startup, business score"
            },
            {
                "name": "Funding Search",
                "description": "Find relevant funding opportunities",
                "trigger": "funding options, investment opportunities, find investors"
            },
            {
                "name": "Regulatory Guidance", 
                "description": "Business registration and compliance help",
                "trigger": "register business, legal requirements, compliance"
            },
            {
                "name": "Market Research",
                "description": "Market analysis and research methodology",
                "trigger": "market research, competition analysis, customer insights"
            },
            {
                "name": "Impact Tracking",
                "description": "Social impact measurement frameworks",
                "trigger": "impact measurement, social metrics, MERL framework"
            }
        ]

# Factory function
def create_ghana_business_agent():
    """Create and return a configured Ghana business agent"""
    return GhanaBusinessAgent()

# Example usage patterns for testing
if __name__ == "__main__":
    agent = create_ghana_business_agent()
    
    # Test queries
    test_queries = [
        "I have a fintech startup in Accra with 3 team members. Can you assess my business readiness?",
        "Find funding opportunities for an agritech startup in Ghana seeking Series A funding",
        "How do I register a technology company in Ghana and what are the tax implications?",
        "I want to research the e-commerce market in Kumasi. What methodology should I use?",
        "Help me create an impact measurement framework for my social enterprise serving rural farmers"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: {query}")
        result = agent.process_query_sync(query, {"country": "Ghana", "sector": "technology"})
        print(f"✅ Response: {result['answer'][:200]}...")
        print(f"🔧 Tools Used: {', '.join(result['tools_used'])}")