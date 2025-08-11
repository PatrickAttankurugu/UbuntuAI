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

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        background: #f8f9ff;
    }
    
    .error-message {
        background: #fee;
        border-left: 4px solid #f66;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    
    .success-message {
        background: #efe;
        border-left: 4px solid #6f6;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    
    .sidebar-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
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
            'sector': 'Fintech',
            'business_stage': 'Pre-seed',  # Changed from 'Idea' to 'Pre-seed'
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
            st.success("✅ Configuration Valid") if is_configured else st.error("❌ Configuration Issues")
        except Exception as e:
            st.error(f"❌ Config Error: {str(e)[:50]}...")
        
        # API status
        if settings.GOOGLE_API_KEY:
            st.success("✅ Gemini API Configured")
        else:
            st.error("❌ Gemini API Key Missing")
        
        # WhatsApp status
        whatsapp_config = settings.get_whatsapp_config()
        if whatsapp_config:
            st.success("✅ WhatsApp Integration Ready")
        else:
            st.warning("⚠️ WhatsApp Not Configured")
        
        # System info
        st.markdown("### 📊 System Info")
        config_info = settings.to_dict()
        st.write(f"**Countries**: {config_info['supported_countries']}")
        st.write(f"**Sectors**: {config_info['supported_sectors']}")
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
                index=settings.BUSINESS_SECTORS.index(st.session_state.user_profile.get('sector', 'Fintech')),
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
        with st.spinner("🤖 Thinking..."):
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
    st.markdown("### 🤖 UbuntuAI Response")
    st.markdown(f'<div class="chat-message">{response["answer"]}</div>', unsafe_allow_html=True)
    
    # Confidence and sources
    col1, col2 = st.columns(2)
    
    with col1:
        confidence = response.get('confidence', 0)
        if confidence > 0:
            st.metric("Confidence", f"{confidence:.1%}")
    
    with col2:
        sources_count = len(response.get('sources', []))
        if sources_count > 0:
            st.metric("Sources Used", sources_count)
    
    # Follow-up questions
    if response.get('follow_up_questions'):
        st.markdown("### 💡 Follow-up Questions")
        for i, question in enumerate(response['follow_up_questions'], 1):
            if st.button(f"{i}. {question}", key=f"followup_{i}"):
                st.session_state['followup_query'] = question
                st.rerun()
    
    # Sources (collapsible)
    if response.get('sources'):
        with st.expander("📚 Sources & References"):
            for i, source in enumerate(response['sources'], 1):
                st.markdown(f"**Source {i}:**")
                st.write(source.get('content_preview', ''))
                if source.get('metadata'):
                    st.json(source['metadata'])

def main():
    """Main application function"""
    try:
        # Initialize session state
        initialize_session_state()
        
        # Header
        st.markdown(f'<div class="main-header"><h1>🚀 {settings.APP_TITLE}</h1><p>{settings.APP_DESCRIPTION}</p></div>', unsafe_allow_html=True)
        
        # Sidebar
        display_system_status()
        display_user_profile()
        
        # Main content area
        st.markdown("### 💬 Ask UbuntuAI")
        
        # Handle follow-up questions
        if 'followup_query' in st.session_state:
            query = st.session_state['followup_query']
            del st.session_state['followup_query']
        else:
            # Example questions
            st.markdown("**Try these example questions:**")
            examples = prompt_templates.get_conversation_starter_prompts()
            
            # Display examples as clickable buttons
            cols = st.columns(2)
            for i, example in enumerate(examples[:6]):  # Show first 6 examples
                with cols[i % 2]:
                    if st.button(example, key=f"example_{i}", help="Click to use this question"):
                        query = example
                        break
            else:
                query = None
            
            # Text input
            if not query:
                query = st.text_area(
                    "Your Question:",
                    placeholder=settings.CHAT_PLACEHOLDER,
                    height=100,
                    help="Ask anything about African business, funding, regulations, or market insights"
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
        
        # Chat history
        if st.session_state.chat_history:
            with st.expander("💬 Conversation History"):
                for i, msg in enumerate(reversed(st.session_state.chat_history[-10:])):  # Show last 10
                    role_icon = "👤" if msg['role'] == 'user' else "🤖"
                    st.markdown(f"**{role_icon} {msg['role'].title()}:**")
                    st.write(msg['content'][:200] + "..." if len(msg['content']) > 200 else msg['content'])
                    st.markdown("---")
        
        # Footer
        st.markdown("---")
        st.markdown("### 🔗 Quick Links")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🏢 Business Assessment"):
                st.session_state['followup_query'] = "Please assess my business readiness and provide recommendations"
                st.rerun()
        
        with col2:
            if st.button("💰 Find Funding"):
                sector = st.session_state.user_profile.get('sector', 'tech')
                country = st.session_state.user_profile.get('country', 'Africa')
                st.session_state['followup_query'] = f"What funding opportunities are available for {sector} startups in {country}?"
                st.rerun()
        
        with col3:
            if st.button("📋 Regulatory Help"):
                country = st.session_state.user_profile.get('country', 'Ghana')
                st.session_state['followup_query'] = f"What are the business registration requirements in {country}?"
                st.rerun()
        
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        st.error("An unexpected error occurred. Please refresh the page and try again.")
        st.code(str(e))

if __name__ == "__main__":
    main()