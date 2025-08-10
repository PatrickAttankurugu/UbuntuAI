import google.generativeai as genai
import json
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from twilio.rest import Client
import logging
from config.settings import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WhatsAppBusinessAgent:
    def __init__(self):
        # Initialize Gemini
        if not settings.GOOGLE_API_KEY:
            raise ValueError("Google API key not configured for WhatsApp agent")
            
        genai.configure(api_key=settings.GOOGLE_API_KEY)
        
        # Use mobile-optimized Gemini model for faster responses
        self.gemini_model = genai.GenerativeModel(
            model_name='gemini-1.5-flash',  # Faster model for mobile
            generation_config=settings.get_mobile_gemini_config()
        )
        
        # Initialize Twilio client
        whatsapp_config = settings.get_whatsapp_config()
        if whatsapp_config:
            self.twilio_client = Client(
                whatsapp_config['account_sid'],
                whatsapp_config['auth_token']
            )
        else:
            self.twilio_client = None
            logger.warning("Twilio not configured - WhatsApp features will be limited")
        
        # Try to import scoring engines
        try:
            from api.scoring_engine import create_scoring_engine
            self.scoring_engines = create_scoring_engine()
        except ImportError:
            logger.warning("Scoring engines not available")
            self.scoring_engines = None
        
        # Language support for Ghana
        self.supported_languages = {
            'en': 'English',
            'tw': 'Twi (basic)',
            'ga': 'Ga (basic)'
        }
        
        # Session memory for each WhatsApp number
        self.user_sessions = {}
        
        # Conversation flows
        self.conversation_flows = {
            'business_assessment': self._business_assessment_flow,
            'funding_guidance': self._funding_guidance_flow,
            'regulatory_help': self._regulatory_help_flow,
            'market_research': self._market_research_flow,
            'general_advice': self._general_advice_flow
        }
        
        # Ghana-specific business context
        self.ghana_context = {
            'common_sectors': ['agriculture', 'retail', 'services', 'technology', 'manufacturing'],
            'funding_sources': ['banks', 'microfinance', 'mobile_money', 'family', 'investors'],
            'business_stages': ['idea', 'startup', 'growing', 'established'],
            'locations': ['accra', 'kumasi', 'tamale', 'cape_coast', 'other']
        }

    def handle_message(self, from_number: str, message_body: str) -> str:
        """Main message handler for WhatsApp messages"""
        try:
            # Initialize or get user session
            if from_number not in self.user_sessions:
                self.user_sessions[from_number] = self._initialize_session()
            
            session = self.user_sessions[from_number]
            
            # Process message based on current flow
            current_flow = session.get('current_flow', 'general_advice')
            flow_handler = self.conversation_flows.get(current_flow, self._general_advice_flow)
            
            response = flow_handler(from_number, message_body, session)
            
            # Update session
            session['last_activity'] = datetime.now()
            session['message_count'] = session.get('message_count', 0) + 1
            
            # Optimize for low bandwidth - keep responses concise
            optimized_response = self._optimize_for_low_bandwidth(response)
            
            return optimized_response
            
        except Exception as e:
            logger.error(f"Error handling message from {from_number}: {str(e)}")
            return self._get_error_response()

    def _initialize_session(self) -> Dict[str, Any]:
        """Initialize a new user session"""
        return {
            'created_at': datetime.now(),
            'last_activity': datetime.now(),
            'language': 'en',
            'current_flow': 'general_advice',
            'flow_state': {},
            'user_profile': {},
            'conversation_history': [],
            'business_data': {},
            'message_count': 0
        }

    def _business_assessment_flow(self, from_number: str, message: str, session: Dict[str, Any]) -> str:
        """Multi-step business assessment flow"""
        flow_state = session.get('flow_state', {})
        step = flow_state.get('step', 0)
        
        if step == 0:
            # Introduction and business description
            flow_state['step'] = 1
            session['flow_state'] = flow_state
            return """🏢 Business Assessment Started!
            
I'll help evaluate your business readiness. This takes 5-7 questions.

1/7: Tell me about your business in 2-3 sentences. What problem do you solve?

Type your answer or 'stop' to exit."""

        elif step == 1:
            # Store business description and ask about sector
            if message.lower() == 'stop':
                return self._reset_to_general_flow(session)
            
            session['business_data']['description'] = message
            flow_state['step'] = 2
            session['flow_state'] = flow_state
            
            sectors = ", ".join(self.ghana_context['common_sectors'])
            return f"""2/7: What sector is your business in?

Choose from: {sectors}

Or type your own sector."""

        elif step == 2:
            # Store sector and ask about team
            session['business_data']['sector'] = message.lower()
            flow_state['step'] = 3
            session['flow_state'] = flow_state
            
            return """3/7: How many people are in your team?

Include yourself and any co-founders, employees, or regular helpers.

Type a number (e.g., 1, 3, 5)"""

        elif step == 3:
            # Store team size and ask about stage
            try:
                team_size = int(re.findall(r'\d+', message)[0])
                session['business_data']['team_size'] = team_size
            except:
                session['business_data']['team_size'] = 1
            
            flow_state['step'] = 4
            session['flow_state'] = flow_state
            
            stages = ", ".join(self.ghana_context['business_stages'])
            return f"""4/7: What stage is your business?

Choose: {stages}

Type one of the options above."""

        elif step == 4:
            # Store stage and ask about customers
            session['business_data']['stage'] = message.lower()
            flow_state['step'] = 5
            session['flow_state'] = flow_state
            
            return """5/7: How many customers do you have?

If you haven't launched yet, type 0.
If you're not sure, give your best estimate.

Type a number."""

        elif step == 5:
            # Store customers and ask about revenue
            try:
                customers = int(re.findall(r'\d+', message)[0])
                session['business_data']['customers'] = customers
            except:
                session['business_data']['customers'] = 0
            
            flow_state['step'] = 6
            session['flow_state'] = flow_state
            
            return """6/7: Are you making money from your business?

Reply:
- YES if you're generating revenue
- NO if not yet
- PLANNING if you have a clear plan"""

        elif step == 6:
            # Store revenue status and ask about location
            revenue_status = message.upper()
            session['business_data']['revenue'] = revenue_status in ['YES', 'PLANNING']
            
            flow_state['step'] = 7
            session['flow_state'] = flow_state
            
            locations = ", ".join(self.ghana_context['locations'])
            return f"""7/7: Where is your business based?

Choose: {locations}

Type one of the locations above."""

        elif step == 7:
            # Final step - generate assessment
            session['business_data']['location'] = message.lower()
            
            # Reset flow
            session['current_flow'] = 'general_advice'
            session['flow_state'] = {}
            
            # Generate assessment using scoring engine or Gemini
            assessment = self._generate_business_assessment(session['business_data'])
            
            return assessment

    def _funding_guidance_flow(self, from_number: str, message: str, session: Dict[str, Any]) -> str:
        """Funding guidance conversation flow"""
        if 'funding_context' not in session:
            session['funding_context'] = {}
            return """💰 Funding Guidance
            
What type of funding help do you need?

1. FIND - Help finding funding sources
2. APPLY - Help with applications  
3. PREPARE - Get ready to apply
4. AMOUNT - How much to ask for

Reply with a number (1-4) or keyword."""

        context = session['funding_context']
        
        if message.upper() in ['1', 'FIND']:
            # Use Gemini to provide funding guidance
            business_info = session.get('business_data', {})
            sector = business_info.get('sector', 'general')
            
            funding_response = self._get_gemini_funding_advice(sector)
            
            # Reset context
            session.pop('funding_context', None)
            
            return funding_response

        elif message.upper() in ['2', 'APPLY']:
            session.pop('funding_context', None)
            return """📝 Application Tips:

1. Strong business plan (1-2 pages)
2. Clear financial projections  
3. Highlight social impact
4. Show market validation
5. Professional pitch deck

Need help with any of these? Ask specifically!"""

        elif message.upper() in ['3', 'PREPARE']:
            session.pop('funding_context', None)
            return """🎯 Get Funding-Ready:

Before applying:
✅ Register your business formally
✅ Get basic financial records
✅ Test your product/service
✅ Know your numbers (costs, revenue)
✅ Practice your pitch

Want to assess your readiness? Type: assess my business"""

        elif message.upper() in ['4', 'AMOUNT']:
            session.pop('funding_context', None)
            return """💭 How Much to Ask For:

Calculate:
• 6-12 months operating costs
• Product development costs  
• Marketing budget
• Emergency buffer (20%)

Example for small tech startup in Ghana:
- Operations: GHS 5,000/month × 12 = 60,000
- Development: GHS 15,000
- Marketing: GHS 10,000  
- Buffer: GHS 17,000
Total: ~GHS 100,000

Need help calculating? Share your monthly costs."""

        else:
            # Handle free-form funding questions
            session.pop('funding_context', None)
            return self._get_gemini_response(f"funding advice for Ghana business: {message}")

    def _regulatory_help_flow(self, from_number: str, message: str, session: Dict[str, Any]) -> str:
        """Business registration and regulatory guidance"""
        
        # Use Gemini to get regulatory information for Ghana
        regulatory_response = self._get_gemini_response(
            f"business registration and regulatory requirements in Ghana: {message}"
        )
        
        # Add Ghana-specific quick tips
        if 'register' in message.lower():
            quick_tips = """

🇬🇭 Quick Ghana Tips:
• Use rgd.gov.gh for registration
• Cost: GHS 500-2000 typically
• Takes 3-5 business days
• Get TIN from GRA after registration"""
            
            regulatory_response += quick_tips
        
        return self._summarize_for_whatsapp(regulatory_response)

    def _market_research_flow(self, from_number: str, message: str, session: Dict[str, Any]) -> str:
        """Market research guidance"""
        
        # Use Gemini for market insights
        market_response = self._get_gemini_response(
            f"market research guidance for Ghana business: {message}"
        )
        
        # Add actionable research steps
        research_tips = """

📊 Research Steps:
1. Talk to 10+ potential customers
2. Check competitors' prices
3. Visit local markets/shops
4. Use Ghana Statistical Service data
5. Join relevant WhatsApp/Facebook groups"""
        
        return self._summarize_for_whatsapp(market_response + research_tips)

    def _general_advice_flow(self, from_number: str, message: str, session: Dict[str, Any]) -> str:
        """Handle general business questions"""
        
        # Detect intent and route to appropriate flow
        message_lower = message.lower()
        
        # Flow routing keywords
        if any(word in message_lower for word in ['assess', 'score', 'evaluate', 'ready']):
            session['current_flow'] = 'business_assessment'
            return self._business_assessment_flow(from_number, message, session)
        
        elif any(word in message_lower for word in ['funding', 'investment', 'money', 'capital']):
            session['current_flow'] = 'funding_guidance'
            return self._funding_guidance_flow(from_number, message, session)
        
        elif any(word in message_lower for word in ['register', 'legal', 'license', 'permit']):
            session['current_flow'] = 'regulatory_help'
            return self._regulatory_help_flow(from_number, message, session)
        
        elif any(word in message_lower for word in ['market', 'research', 'customers', 'competition']):
            session['current_flow'] = 'market_research'
            return self._market_research_flow(from_number, message, session)
        
        # Handle greetings and help
        elif any(word in message_lower for word in ['hello', 'hi', 'hey', 'start']):
            return self._get_welcome_message()
        
        elif 'help' in message_lower:
            return self._get_help_message()
        
        # General Gemini query
        else:
            return self._get_gemini_response(f"Ghana business advice: {message}")

    def _get_gemini_response(self, prompt: str) -> str:
        """Get response from Gemini with context for Ghana business"""
        try:
            enhanced_prompt = f"""You are UbuntuAI, an expert business advisor for African entrepreneurs, especially in Ghana. 

Context: You're helping a Ghanaian entrepreneur via WhatsApp. Keep responses:
- Under 600 characters for mobile
- Practical and actionable
- Specific to Ghana/Africa when possible
- Encouraging but realistic

Question: {prompt}

Response:"""

            response = self.gemini_model.generate_content(enhanced_prompt)
            
            if response and response.text:
                return self._summarize_for_whatsapp(response.text)
            else:
                return "I'm having trouble generating a response right now. Please try rephrasing your question."
                
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return "Sorry, I'm experiencing technical difficulties. Please try again shortly."

    def _get_gemini_funding_advice(self, sector: str) -> str:
        """Get funding advice from Gemini"""
        prompt = f"""List 3-4 funding sources for a {sector} business in Ghana. For each, include:
- Name of funding source
- Type of funding
- Typical amount range in GHS
- Key requirement

Keep response under 500 characters, mobile-friendly format."""

        return self._get_gemini_response(prompt)

    def _generate_business_assessment(self, business_data: Dict[str, Any]) -> str:
        """Generate business assessment using Gemini or scoring engine"""
        
        if self.scoring_engines:
            # Use scoring engine if available
            try:
                scoring_data = {
                    'business_description': business_data.get('description', ''),
                    'sector': business_data.get('sector', ''),
                    'team_size': business_data.get('team_size', 1),
                    'product_stage': self._map_stage_to_product_stage(business_data.get('stage', 'idea')),
                    'customer_count': business_data.get('customers', 0),
                    'generating_revenue': business_data.get('revenue', False),
                    'local_team_members': True,
                    'mobile_first': True,
                    'local_market_knowledge': True
                }
                
                scorer = self.scoring_engines['startup_scorer']
                result = scorer.score_startup(scoring_data)
                
                return self._format_assessment_result(result)
                
            except Exception as e:
                logger.error(f"Scoring engine error: {e}")
                # Fall back to Gemini
        
        # Use Gemini for assessment
        prompt = f"""Assess this Ghana business and give a score out of 10 with brief advice:

Business: {business_data.get('description', 'N/A')}
Sector: {business_data.get('sector', 'N/A')}
Team: {business_data.get('team_size', 1)} people
Stage: {business_data.get('stage', 'idea')}
Customers: {business_data.get('customers', 0)}
Revenue: {'Yes' if business_data.get('revenue') else 'No'}
Location: {business_data.get('location', 'Ghana')}

Give:
1. Score /10
2. Top strength
3. Main weakness
4. Next step

Keep under 400 characters."""

        return self._get_gemini_response(prompt)

    def _format_assessment_result(self, result) -> str:
        """Format scoring engine result for WhatsApp"""
        score_emoji = "🟢" if result.overall_score > 0.7 else "🟡" if result.overall_score > 0.5 else "🔴"
        
        assessment = f"""📊 Your Business Assessment {score_emoji}

Overall Score: {result.overall_score:.1f}/1.0

🎯 Top Strength: {max(result.component_scores, key=result.component_scores.get).replace('_', ' ').title()}

⚠️ Area to Improve: {min(result.component_scores, key=result.component_scores.get).replace('_', ' ').title()}

💡 Next Step: {result.recommendations[0] if result.recommendations else 'Focus on customer validation'}

Want detailed advice? Just ask!"""
        
        return assessment

    def _map_stage_to_product_stage(self, stage: str) -> str:
        """Map WhatsApp stage to scoring stage"""
        stage_mapping = {
            'idea': 'idea',
            'startup': 'prototype',
            'growing': 'beta',
            'established': 'launched'
        }
        return stage_mapping.get(stage, 'idea')

    def _optimize_for_low_bandwidth(self, response: str) -> str:
        """Optimize response for low bandwidth/data usage"""
        
        # Keep responses under WhatsApp limit
        max_length = settings.WHATSAPP_MAX_MESSAGE_LENGTH
        if len(response) > max_length:
            response = response[:max_length-20] + "...\n\nType 'more' for details"
        
        # Remove excessive formatting
        response = re.sub(r'\*\*\*+', '**', response)
        response = re.sub(r'\n\n\n+', '\n\n', response)
        
        # Replace long URLs with placeholders
        response = re.sub(r'https?://[^\s]+', '[link]', response)
        
        return response

    def _summarize_for_whatsapp(self, long_text: str) -> str:
        """Summarize long responses for WhatsApp"""
        
        max_chars = 600
        if len(long_text) < max_chars:
            return long_text
        
        # Use Gemini to summarize
        try:
            summary_prompt = f"""Summarize this for WhatsApp (max {max_chars} chars, mobile-friendly, Ghana context):

{long_text[:2000]}

Keep key info and make actionable."""
            
            response = self.gemini_model.generate_content(summary_prompt)
            
            if response and response.text:
                return response.text.strip()[:max_chars]
            else:
                # Fallback: simple truncation
                return long_text[:max_chars] + "...\n\nType 'more info' for details"
                
        except Exception as e:
            logger.error(f"Summarization error: {e}")
            # Fallback: simple truncation
            return long_text[:max_chars] + "...\n\nType 'more info' for details"

    def _get_welcome_message(self) -> str:
        """Get welcome message for new users"""
        return """👋 Welcome to Ghana Business AI!

I'm here to help with:
🏢 Business advice
💰 Funding guidance  
📋 Registration help
📊 Market insights

What can I help you with today?

Try: "assess my business" or ask any business question!"""

    def _get_help_message(self) -> str:
        """Get help message"""
        return """🆘 How I Can Help:

Commands:
• "assess my business" - Get scored evaluation
• "funding options" - Find funding sources
• "register business" - Registration guide
• "market research" - Research tips

Or just ask any business question in plain language!

Example: "How do I price my product?" or "What permits do I need?"

🇬🇭 Specialized for Ghana"""

    def _get_error_response(self) -> str:
        """Get error response for failed messages"""
        return """Sorry, I'm having technical issues right now. 

Please try again in a few minutes.

For urgent help, you can also visit ghana.gov.gh for business registration or contact your nearest business development center."""

    def _reset_to_general_flow(self, session: Dict[str, Any]) -> str:
        """Reset session to general flow"""
        session['current_flow'] = 'general_advice'
        session['flow_state'] = {}
        return "Assessment stopped. How else can I help with your business?"

    def cleanup_old_sessions(self, hours: int = 24):
        """Clean up old sessions to manage memory"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        expired_sessions = [
            number for number, session in self.user_sessions.items()
            if session.get('last_activity', datetime.min) < cutoff_time
        ]
        
        for number in expired_sessions:
            del self.user_sessions[number]
        
        logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")


# Flask app for WhatsApp webhook
app = Flask(__name__)

# Initialize WhatsApp agent
try:
    whatsapp_agent = WhatsAppBusinessAgent()
    logger.info("WhatsApp agent initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize WhatsApp agent: {e}")
    whatsapp_agent = None

@app.route("/whatsapp", methods=['POST'])
def whatsapp_webhook():
    """Handle incoming WhatsApp messages"""
    if not whatsapp_agent:
        return jsonify({"error": "WhatsApp agent not available"}), 503
        
    try:
        from_number = request.values.get('From', '').replace('whatsapp:', '')
        message_body = request.values.get('Body', '').strip()
        
        if not from_number or not message_body:
            return jsonify({"error": "Missing required fields"}), 400
        
        # Get response from agent
        response = whatsapp_agent.handle_message(from_number, message_body)
        
        # Send response via Twilio if configured
        if whatsapp_agent.twilio_client:
            to_number = f"whatsapp:{from_number}"
            
            whatsapp_agent.twilio_client.messages.create(
                body=response,
                from_=f"whatsapp:{settings.TWILIO_WHATSAPP_NUMBER}",
                to=to_number
            )
        
        return jsonify({"status": "success"}), 200
        
    except Exception as e:
        logger.error(f"WhatsApp webhook error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "agent_available": whatsapp_agent is not None
    }), 200

@app.route("/cleanup", methods=['POST'])
def cleanup_sessions():
    """Manual session cleanup"""
    if whatsapp_agent:
        whatsapp_agent.cleanup_old_sessions()
        return jsonify({"status": "cleaned"}), 200
    else:
        return jsonify({"error": "Agent not available"}), 503

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=5000)