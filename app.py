import streamlit as st
import sys
import os
import logging
from datetime import datetime
import traceback
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ubuntuai.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import core components with error handling
try:
    from config.settings import settings, SettingsValidationError
    from config.prompts import prompt_templates
except ImportError as e:
    st.error(f"Configuration error: {e}")
    st.stop()

try:
    from api.rag_engine import rag_engine
    from api.langchain_agents import create_ghana_business_agent
except ImportError as e:
    st.error(f"API module error: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title=settings.APP_TITLE,
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern CSS styling
st.markdown("""
<style>
    /* Import modern fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styles */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Header styling */
    .main-header {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        padding: 3rem 2rem;
        border-radius: 20px;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
        text-align: center;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .main-header h1 {
        color: #2D3748;
        font-weight: 700;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        letter-spacing: -0.025em;
    }
    
    .main-header p {
        color: #4A5568;
        font-size: 1.125rem;
        font-weight: 400;
        margin: 0;
    }
    
    /* Chat interface */
    .chat-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin-bottom: 2rem;
    }
    
    .chat-message {
        background: linear-gradient(135deg, #f8f9ff 0%, #e8eeff 100%);
        padding: 1.5rem;
        border-radius: 16px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    }
    
    .response-header {
        color: #2D3748;
        font-weight: 600;
        font-size: 1.25rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
    }
    
    .sidebar-section {
        background: rgba(248, 250, 252, 0.8);
        padding: 1.5rem;
        border-radius: 16px;
        margin: 1rem 0;
        border: 1px solid rgba(226, 232, 240, 0.8);
    }
    
    /* Metrics and stats */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9ff 100%);
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.06);
        border: 1px solid rgba(226, 232, 240, 0.8);
        margin: 0.5rem 0;
        text-align: center;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
        margin-bottom: 0.25rem;
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: #64748B;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Status indicators */
    .status-success {
        background: linear-gradient(135deg, #10B981, #059669);
        color: white;
        padding: 0.75rem 1rem;
        border-radius: 12px;
        font-weight: 500;
        margin: 0.5rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .status-error {
        background: linear-gradient(135deg, #EF4444, #DC2626);
        color: white;
        padding: 0.75rem 1rem;
        border-radius: 12px;
        font-weight: 500;
        margin: 0.5rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .status-warning {
        background: linear-gradient(135deg, #F59E0B, #D97706);
        color: white;
        padding: 0.75rem 1rem;
        border-radius: 12px;
        font-weight: 500;
        margin: 0.5rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Action buttons */
    .action-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 1rem 2rem;
        border-radius: 12px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);
        width: 100%;
        margin: 0.5rem 0;
    }
    
    .action-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 25px rgba(102, 126, 234, 0.4);
    }
    
    /* Follow-up questions */
    .follow-up-question {
        background: rgba(102, 126, 234, 0.1);
        border: 1px solid rgba(102, 126, 234, 0.2);
        padding: 1rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .follow-up-question:hover {
        background: rgba(102, 126, 234, 0.15);
        transform: translateX(4px);
    }
    
    /* Sources section */
    .sources-container {
        background: rgba(248, 250, 252, 0.8);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(226, 232, 240, 0.8);
    }
    
    .source-item {
        background: white;
        padding: 1rem;
        border-radius: 12px;
        margin: 0.75rem 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        border-left: 4px solid #667eea;
    }
    
    /* Text input styling */
    .stTextArea textarea {
        border-radius: 16px;
        border: 2px solid rgba(102, 126, 234, 0.2);
        padding: 1rem;
        font-family: 'Inter', sans-serif;
        transition: all 0.3s ease;
    }
    
    .stTextArea textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Hide default streamlit styling */
    .css-1y4p8pa {
        padding-top: 0;
    }
    
    .css-1d391kg .css-1y4p8pa {
        padding-top: 1rem;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header {
            padding: 2rem 1rem;
        }
        
        .main-header h1 {
            font-size: 2rem;
        }
        
        .chat-container {
            padding: 1.5rem;
        }
    }
</style>
""", unsafe_allow_html=True)

def validate_user_input(query: str) -> tuple[bool, str]:
    """
    Validate user input for safety and basic requirements
    
    Args:
        query: User's input query
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not query or not query.strip():
        return False, "Please enter a question"
    
    if len(query.strip()) < 3:
        return False, "Question too short - please provide more details"
    
    if len(query) > 2000:
        return False, "Question too long - please keep it under 2000 characters"
    
    # Basic safety check for malicious content
    dangerous_patterns = ['<script', 'javascript:', 'eval(', 'exec(']
    query_lower = query.lower()
    if any(pattern in query_lower for pattern in dangerous_patterns):
        return False, "Invalid characters detected in input"
    
    return True, ""

def safe_execute(func, *args, **kwargs):
    """
    Safely execute a function with error handling and logging
    
    Args:
        func: Function to execute
        *args, **kwargs: Arguments for the function
        
    Returns:
        Function result or None if error occurred
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.error(f"Error in {func.__name__}: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

def initialize_session_state():
    """Initialize Streamlit session state with default values"""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'user_profile' not in st.session_state:
        st.session_state.user_profile = {
            'name': '',
            'country': 'Ghana',
            'sector': 'Agritech',
            'business_stage': 'Pre-seed',
            'team_size': 1,
            'experience_level': 'First-time entrepreneur'
        }
    
    if 'app_initialized' not in st.session_state:
        st.session_state.app_initialized = False

def display_system_status():
    """Display system status and configuration in sidebar"""
    with st.sidebar:
        st.markdown("### 🔧 System Status")
        
        # Settings validation
        try:
            is_configured = settings.validate_config()
            if is_configured:
                st.markdown('<div class="status-success">✅ Configuration Valid</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="status-error">❌ Configuration Issues</div>', unsafe_allow_html=True)
        except Exception as e:
            st.markdown(f'<div class="status-error">❌ Config Error: {str(e)[:50]}...</div>', unsafe_allow_html=True)
        
        # API status
        if settings.GOOGLE_API_KEY:
            st.markdown('<div class="status-success">✅ Gemini API Configured</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-error">❌ Gemini API Key Missing</div>', unsafe_allow_html=True)
        
        # WhatsApp status
        whatsapp_config = settings.get_whatsapp_config()
        if whatsapp_config:
            st.markdown('<div class="status-success">✅ WhatsApp Integration Ready</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-warning">⚠️ WhatsApp Not Configured</div>', unsafe_allow_html=True)
        
        # System info
        st.markdown("### 📊 System Info")
        config_info = settings.to_dict()
        
        # Metrics cards
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-value">{config_info["supported_countries"]}</div>
            <div class="metric-label">Countries</div>
        </div>
        ''', unsafe_allow_html=True)
        
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-value">{config_info["supported_sectors"]}</div>
            <div class="metric-label">Sectors</div>
        </div>
        ''', unsafe_allow_html=True)
        
        st.write(f"**Model**: {config_info['embedding_model']}")

def display_user_profile():
    """Display and manage user profile in sidebar"""
    with st.sidebar:
        st.markdown("### 👤 Your Profile")
        
        with st.form("user_profile_form"):
            st.session_state.user_profile['name'] = st.text_input(
                "Name", 
                value=st.session_state.user_profile.get('name', ''),
                help="Your name for personalized responses"
            )
            
            st.session_state.user_profile['country'] = st.selectbox(
                "Country",
                options=settings.AFRICAN_COUNTRIES,
                index=settings.AFRICAN_COUNTRIES.index(st.session_state.user_profile.get('country', 'Ghana')),
                help="Your business location"
            )
            
            st.session_state.user_profile['sector'] = st.selectbox(
                "Business Sector",
                options=settings.BUSINESS_SECTORS,
                index=settings.BUSINESS_SECTORS.index(st.session_state.user_profile.get('sector', 'Agritech')),
                help="Your industry focus"
            )
            
            # Fix the business stage selection
            current_stage = st.session_state.user_profile.get('business_stage', 'Pre-seed')
            # Ensure the current stage exists in the list, if not use Pre-seed
            if current_stage not in settings.FUNDING_STAGES:
                current_stage = 'Pre-seed'
                st.session_state.user_profile['business_stage'] = current_stage
            
            st.session_state.user_profile['business_stage'] = st.selectbox(
                "Business Stage",
                options=settings.FUNDING_STAGES,
                index=settings.FUNDING_STAGES.index(current_stage),
                help="Current stage of your business"
            )
            
            st.session_state.user_profile['team_size'] = st.number_input(
                "Team Size",
                min_value=1,
                max_value=100,
                value=st.session_state.user_profile.get('team_size', 1),
                help="Number of team members"
            )
            
            if st.form_submit_button("💾 Update Profile"):
                st.success("Profile updated!")
                logger.info(f"User profile updated: {st.session_state.user_profile}")

def process_user_query(query: str) -> Optional[Dict[str, Any]]:
    """
    Process user query through the AI system
    
    Args:
        query: User's question
        
    Returns:
        Response dictionary or None if error
    """
    try:
        # Create user context from profile
        user_context = st.session_state.user_profile.copy()
        
        # Add conversation history context
        if st.session_state.chat_history:
            recent_history = st.session_state.chat_history[-5:]  # Last 5 exchanges
            user_context['recent_conversation'] = recent_history
        
        # Process through RAG engine
        with st.spinner("🤖 Analyzing your question..."):
            response = rag_engine.query(
                question=query,
                conversation_history=st.session_state.chat_history,
                user_context=user_context
            )
        
        if response and response.get('answer'):
            # Log successful query
            logger.info(f"Query processed successfully for user: {user_context.get('name', 'Anonymous')}")
            return response
        else:
            st.error("No response generated. Please try rephrasing your question.")
            return None
            
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        st.error(f"Sorry, I encountered an error: {str(e)}")
        return None

def display_response(response: Dict[str, Any]):
    """Display the AI response with proper formatting"""
    if not response:
        return
    
    # Main answer
    st.markdown(f'''
    <div class="chat-container">
        <div class="response-header">🤖 UbuntuAI Response</div>
        <div class="chat-message">{response["answer"]}</div>
    </div>
    ''', unsafe_allow_html=True)
    
    # Confidence and sources
    col1, col2 = st.columns(2)
    
    with col1:
        confidence = response.get('confidence', 0)
        if confidence > 0:
            st.markdown(f'''
            <div class="metric-card">
                <div class="metric-value">{confidence:.1%}</div>
                <div class="metric-label">Confidence</div>
            </div>
            ''', unsafe_allow_html=True)
    
    with col2:
        sources_count = len(response.get('sources', []))
        if sources_count > 0:
            st.markdown(f'''
            <div class="metric-card">
                <div class="metric-value">{sources_count}</div>
                <div class="metric-label">Sources Used</div>
            </div>
            ''', unsafe_allow_html=True)
    
    # Follow-up questions
    if response.get('follow_up_questions'):
        st.markdown("### 💡 Follow-up Questions")
        for i, question in enumerate(response['follow_up_questions'], 1):
            if st.button(f"{i}. {question}", key=f"followup_{i}", help="Click to explore this topic"):
                st.session_state['followup_query'] = question
                st.rerun()
    
    # Sources (collapsible)
    if response.get('sources'):
        with st.expander("📚 Sources & References", expanded=False):
            for i, source in enumerate(response['sources'], 1):
                st.markdown(f'''
                <div class="source-item">
                    <strong>Source {i}:</strong><br>
                    {source.get('content_preview', '')}
                </div>
                ''', unsafe_allow_html=True)
                
                if source.get('metadata'):
                    st.json(source['metadata'])

def main():
    """Main application function"""
    try:
        # Initialize session state
        initialize_session_state()
        
        # Header
        st.markdown(f'''
        <div class="main-header">
            <h1>🚀 {settings.APP_TITLE}</h1>
            <p>{settings.APP_DESCRIPTION}</p>
        </div>
        ''', unsafe_allow_html=True)
        
        # Sidebar
        display_system_status()
        display_user_profile()
        
        # Main content area
        st.markdown('''
        <div class="chat-container">
            <h3>💬 Ask UbuntuAI</h3>
        </div>
        ''', unsafe_allow_html=True)
        
        # Handle follow-up questions
        if 'followup_query' in st.session_state:
            query = st.session_state['followup_query']
            del st.session_state['followup_query']
        else:
            # Text input
            query = st.text_area(
                "",
                placeholder=settings.CHAT_PLACEHOLDER,
                height=120,
                help="Ask anything about African business, funding, regulations, or market insights",
                label_visibility="collapsed"
            )
        
        # Process query
        if query:
            # Validate input
            is_valid, error_msg = validate_user_input(query)
            if not is_valid:
                st.error(error_msg)
                return
            
            # Add to chat history
            st.session_state.chat_history.append({
                'role': 'user',
                'content': query,
                'timestamp': datetime.now().isoformat()
            })
            
            # Process and display response
            response = process_user_query(query)
            if response:
                display_response(response)
                
                # Add assistant response to history
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': response['answer'],
                    'timestamp': datetime.now().isoformat()
                })
        
        # Quick Actions
        st.markdown("---")
        st.markdown("### 🚀 Quick Actions")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🏢 Business Assessment", help="Get a comprehensive business readiness assessment"):
                st.session_state['followup_query'] = "Please assess my business readiness and provide recommendations"
                st.rerun()
        
        with col2:
            if st.button("💰 Find Funding", help="Discover funding opportunities for your business"):
                sector = st.session_state.user_profile.get('sector', 'tech')
                country = st.session_state.user_profile.get('country', 'Africa')
                st.session_state['followup_query'] = f"What funding opportunities are available for {sector} startups in {country}?"
                st.rerun()
        
        with col3:
            if st.button("📋 Regulatory Help", help="Get guidance on business registration and compliance"):
                country = st.session_state.user_profile.get('country', 'Ghana')
                st.session_state['followup_query'] = f"What are the business registration requirements in {country}?"
                st.rerun()
        
        # Chat history (collapsible)
        if st.session_state.chat_history:
            with st.expander("💬 Conversation History", expanded=False):
                for i, msg in enumerate(reversed(st.session_state.chat_history[-10:])):  # Show last 10
                    role_icon = "👤" if msg['role'] == 'user' else "🤖"
                    st.markdown(f"**{role_icon} {msg['role'].title()}:**")
                    st.write(msg['content'][:200] + "..." if len(msg['content']) > 200 else msg['content'])
                    st.markdown("---")
        
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        st.error("An unexpected error occurred. Please refresh the page and try again.")
        st.code(str(e))

if __name__ == "__main__":
    main()