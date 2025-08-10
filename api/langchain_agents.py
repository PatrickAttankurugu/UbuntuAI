from typing import Dict, List, Any, Optional
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

class SimpleGhanaBusinessAgent:
    """
    Simplified business agent that provides Ghana-focused business intelligence
    without complex LangChain dependencies
    """
    
    def __init__(self):
        self.scoring_engines = create_scoring_engine()
        
    def process_query_sync(self, query: str, user_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process user query and provide business intelligence response"""
        
        try:
            # Determine query type and route accordingly
            query_type = self._classify_query(query.lower())
            
            if query_type == "business_assessment":
                return self._handle_business_assessment(query, user_context)
            elif query_type == "funding_search":
                return self._handle_funding_search(query, user_context)
            elif query_type == "regulatory_guidance":
                return self._handle_regulatory_guidance(query, user_context)
            elif query_type == "market_research":
                return self._handle_market_research(query, user_context)
            else:
                return self._handle_general_query(query, user_context)
                
        except Exception as e:
            logger.error(f"Agent processing error: {str(e)}")
            return {
                "answer": f"I encountered an error processing your request: {str(e)}. Please try rephrasing your question.",
                "tools_used": [],
                "success": False
            }
    
    def _classify_query(self, query: str) -> str:
        """Simple query classification based on keywords"""
        
        if any(word in query for word in ['assess', 'score', 'readiness', 'evaluate', 'rating']):
            return "business_assessment"
        elif any(word in query for word in ['funding', 'investment', 'investor', 'capital', 'finance']):
            return "funding_search"
        elif any(word in query for word in ['register', 'legal', 'compliance', 'law', 'regulation']):
            return "regulatory_guidance"
        elif any(word in query for word in ['market', 'research', 'competition', 'analysis']):
            return "market_research"
        else:
            return "general"
    
    def _handle_business_assessment(self, query: str, user_context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle business assessment requests"""
        
        try:
            # Extract business information from context
            business_data = {
                'business_description': query,
                'sector': user_context.get('sector', 'general'),
                'team_size': user_context.get('team_size', 1),
                'product_stage': user_context.get('business_stage', 'idea'),
                'local_team_members': user_context.get('country') == 'Ghana',
                'mobile_first': True,  # Assume true for African markets
                'local_market_knowledge': True,
                'target_market': f"{user_context.get('sector', 'general')} customers in {user_context.get('country', 'Ghana')}"
            }
            
            # Get scoring assessment
            scorer = self.scoring_engines['startup_scorer']
            result = scorer.score_startup(business_data)
            
            # Format response
            response = f"""🏢 BUSINESS ASSESSMENT RESULTS

📊 OVERALL SCORE: {result.overall_score:.2f}/1.0

📈 COMPONENT BREAKDOWN:"""
            
            for component, score in result.component_scores.items():
                emoji = "🟢" if score > 0.7 else "🟡" if score > 0.5 else "🔴"
                response += f"\n{emoji} {component.replace('_', ' ').title()}: {score:.2f}/1.0"
            
            if result.risk_factors:
                response += "\n\n⚠️ KEY RISK FACTORS:"
                for risk in result.risk_factors[:3]:
                    response += f"\n• {risk}"
            
            if result.recommendations:
                response += "\n\n💡 RECOMMENDATIONS:"
                for rec in result.recommendations[:3]:
                    response += f"\n• {rec}"
            
            # Add Ghana-specific insights
            if user_context.get('country') == 'Ghana':
                response += f"\n\n🇬🇭 GHANA MARKET INSIGHTS:"
                response += f"\n• {user_context.get('sector', 'Your')} sector shows good potential in Ghana"
                response += f"\n• Consider mobile-first approach for Ghana market"
                response += f"\n• Local partnerships recommended for market entry"
            
            return {
                "answer": response,
                "tools_used": ["business_assessment"],
                "success": True
            }
            
        except Exception as e:
            return {
                "answer": f"Error in business assessment: {str(e)}",
                "tools_used": ["business_assessment"],
                "success": False
            }
    
    def _handle_funding_search(self, query: str, user_context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle funding opportunity searches"""
        
        try:
            # Search for funding opportunities
            search_filters = {
                'sector': user_context.get('sector'),
                'country': user_context.get('country', 'Ghana'),
                'stage': user_context.get('business_stage')
            }
            
            # Remove None values
            search_filters = {k: v for k, v in search_filters.items() if v}
            
            funding_opportunities = funding_db.search_funding(**search_filters)
            
            if not funding_opportunities:
                # Fallback to RAG search
                enhanced_query = f"funding opportunities for {user_context.get('sector', 'business')} startups in {user_context.get('country', 'Ghana')}"
                rag_response = rag_engine.query(enhanced_query, user_context=user_context)
                return {
                    "answer": rag_response.get('answer', 'No specific funding opportunities found.'),
                    "tools_used": ["funding_search", "rag_engine"],
                    "success": True
                }
            
            # Format funding results
            response = f"💰 FUNDING OPPORTUNITIES ({len(funding_opportunities)} found)\n\n"
            
            for i, opp in enumerate(funding_opportunities[:5], 1):
                response += f"{i}. **{opp['name']}** ({opp.get('type', 'N/A')})\n"
                response += f"   💵 Investment: {opp.get('typical_investment', 'N/A')}\n"
                response += f"   🎯 Focus: {', '.join(opp.get('focus_sectors', []))}\n"
                response += f"   📍 Location: {opp.get('country', 'N/A')}\n"
                
                if opp.get('application_process'):
                    response += f"   📝 Application: {opp['application_process']}\n"
                
                if opp.get('website'):
                    response += f"   🌐 Website: {opp['website']}\n"
                
                response += "\n"
            
            response += "🎯 NEXT STEPS:\n"
            response += "1. Research each funder's portfolio companies\n"
            response += "2. Prepare pitch deck and business plan\n"
            response += "3. Get warm introductions when possible\n"
            response += "4. Apply to 3-5 relevant opportunities"
            
            return {
                "answer": response,
                "tools_used": ["funding_search"],
                "success": True
            }
            
        except Exception as e:
            return {
                "answer": f"Error searching funding: {str(e)}",
                "tools_used": ["funding_search"],
                "success": False
            }
    
    def _handle_regulatory_guidance(self, query: str, user_context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle regulatory and legal guidance requests"""
        
        try:
            country = user_context.get('country', 'Ghana')
            
            # Get regulatory information
            reg_info = regulatory_db.get_business_registration_info(country)
            tax_info = regulatory_db.get_tax_information(country)
            
            if reg_info:
                response = f"📋 BUSINESS REGISTRATION - {country.upper()}\n\n"
                response += f"**Authority**: {reg_info.get('registration_authority', 'N/A')}\n"
                response += f"**Online Portal**: {reg_info.get('online_portal', 'N/A')}\n"
                response += f"**Processing Time**: {reg_info.get('processing_time', 'N/A')}\n"
                response += f"**Cost**: {reg_info.get('cost', 'N/A')}\n\n"
                
                if reg_info.get('required_documents'):
                    response += "**Required Documents:**\n"
                    for doc in reg_info['required_documents']:
                        response += f"• {doc}\n"
                
                if tax_info:
                    response += f"\n💰 TAX INFORMATION:\n"
                    response += f"**Corporate Tax**: {tax_info.get('corporate_tax_rate', 'N/A')}\n"
                    response += f"**VAT Rate**: {tax_info.get('vat_rate', 'N/A')}\n"
                    response += f"**Tax Authority**: {tax_info.get('tax_authority', 'N/A')}\n"
            else:
                # Fallback to RAG search
                enhanced_query = f"business registration and legal requirements in {country}"
                rag_response = rag_engine.query(enhanced_query, user_context=user_context)
                response = rag_response.get('answer', f'Regulatory information for {country} is not available.')
            
            return {
                "answer": response,
                "tools_used": ["regulatory_guidance"],
                "success": True
            }
            
        except Exception as e:
            return {
                "answer": f"Error getting regulatory guidance: {str(e)}",
                "tools_used": ["regulatory_guidance"],
                "success": False
            }
    
    def _handle_market_research(self, query: str, user_context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle market research requests"""
        
        try:
            # Use RAG for market insights
            enhanced_query = f"market research {user_context.get('sector', 'business')} {user_context.get('country', 'Ghana')} analysis"
            rag_response = rag_engine.query(enhanced_query, user_context=user_context)
            
            market_info = rag_response.get('answer', '')
            
            # Add structured research methodology
            research_guide = "\n\n📊 MARKET RESEARCH METHODOLOGY:\n\n"
            research_guide += "🎯 CUSTOMER RESEARCH:\n"
            research_guide += "1. Conduct 15-20 customer interviews\n"
            research_guide += "2. Create online surveys (Google Forms)\n"
            research_guide += "3. Observe customer behavior in natural settings\n"
            research_guide += "4. Join relevant WhatsApp/Facebook groups\n"
            research_guide += "5. Test willingness to pay with MVP\n\n"
            
            research_guide += "🏢 COMPETITIVE ANALYSIS:\n"
            research_guide += "1. Identify direct and indirect competitors\n"
            research_guide += "2. Analyze their pricing strategies\n"
            research_guide += "3. Study their marketing approaches\n"
            research_guide += "4. Assess their strengths and weaknesses\n"
            research_guide += "5. Find market gaps and opportunities\n\n"
            
            # Country-specific research tips
            if user_context.get('country') == 'Ghana':
                research_guide += "🇬🇭 GHANA-SPECIFIC TIPS:\n"
                research_guide += "• Visit local markets in major cities\n"
                research_guide += "• Connect with trade associations\n"
                research_guide += "• Use Ghana Statistical Service data\n"
                research_guide += "• Consider mobile money adoption rates\n"
                research_guide += "• Factor in seasonal spending patterns"
            
            return {
                "answer": market_info + research_guide,
                "tools_used": ["market_research", "rag_engine"],
                "success": True
            }
            
        except Exception as e:
            return {
                "answer": f"Error conducting market research: {str(e)}",
                "tools_used": ["market_research"],
                "success": False
            }
    
    def _handle_general_query(self, query: str, user_context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle general business queries using RAG"""
        
        try:
            rag_response = rag_engine.query(query, user_context=user_context)
            
            return {
                "answer": rag_response.get('answer', 'I apologize, but I could not find relevant information for your query.'),
                "tools_used": ["rag_engine"],
                "success": True
            }
            
        except Exception as e:
            return {
                "answer": f"Error processing query: {str(e)}",
                "tools_used": ["rag_engine"],
                "success": False
            }

def create_ghana_business_agent():
    """Create and return a simplified Ghana business agent"""
    return SimpleGhanaBusinessAgent()

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