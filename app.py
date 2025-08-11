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
    page_icon="🇬🇭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern CSS styling with Ghanaian theme
st.markdown("""
<style>
    /* Import modern fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styles */
    .stApp {
        background: linear-gradient(135deg, #CE1126 0%, #FCD116 50%, #006B3F 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Header styling with Ghanaian colors */
    .main-header {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        padding: 3rem 2rem;
        border-radius: 20px;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
        text-align: center;
        margin-bottom: 2rem;
        border: 3px solid #CE1126;
    }
    
    .main-header h1 {
        color: #CE1126;
        font-weight: 700;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        letter-spacing: -0.025em;
    }
    
    .main-header p {
        color: #006B3F;
        font-size: 1.125rem;
        font-weight: 500;
        margin: 0;
    }
    
    /* Chat interface */
    .chat-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
        border: 2px solid #FCD116;
        margin-bottom: 2rem;
    }
    
    /* Sector badges */
    .sector-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        margin: 0.25rem;
        border-radius: 25px;
        font-size: 0.875rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .sector-fintech {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .sector-agritech {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
    }
    
    .sector-healthtech {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
    }
    
    /* Ghana flag colors */
    .ghana-red { color: #CE1126; }
    .ghana-yellow { color: #FCD116; }
    .ghana-green { color: #006B3F; }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #CE1126 0%, #FCD116 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(206, 17, 38, 0.3);
    }
    
    /* Chat message styling */
    .chat-message {
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 15px;
        border-left: 4px solid #FCD116;
    }
    
    .user-message {
        background: rgba(206, 17, 38, 0.1);
        border-left-color: #CE1126;
    }
    
    .assistant-message {
        background: rgba(0, 107, 63, 0.1);
        border-left-color: #006B3F;
    }
</style>
""")

def validate_user_input(query: str) -> tuple[bool, str]:
    """Validate user input for Ghanaian startup ecosystem focus"""
    
    if not query or not query.strip():
        return False, "Please enter a question about the Ghanaian startup ecosystem."
    
    query_lower = query.lower().strip()
    
    # Check if query is relevant to Ghanaian startup ecosystem
    ghana_keywords = [
        "ghana", "ghanian", "accra", "kumasi", "tamale", "kumasi", "tema",
        "fintech", "agritech", "healthtech", "startup", "entrepreneur",
        "business", "investment", "funding", "regulation", "compliance"
    ]
    
    if not any(keyword in query_lower for keyword in ghana_keywords):
        return False, "Please ask a question related to the Ghanaian startup ecosystem (fintech, agritech, or healthtech)."
    
    if len(query.strip()) < 10:
        return False, "Please provide a more detailed question (at least 10 characters)."
    
    if len(query.strip()) > 1000:
        return False, "Question is too long. Please keep it under 1000 characters."
    
    return True, ""

def safe_execute(func, *args, **kwargs):
    """Safely execute functions with error handling"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.error(f"Error executing {func.__name__}: {e}")
        return None

def initialize_session_state():
    """Initialize Streamlit session state"""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "current_provider" not in st.session_state:
        st.session_state.current_provider = settings.PRIMARY_LLM_PROVIDER
    
    if "user_profile" not in st.session_state:
        st.session_state.user_profile = {
            "sector": "general",
            "region": "Greater Accra",
            "experience_level": "beginner"
        }

def display_system_status():
    """Display system status and configuration"""
    
    st.sidebar.markdown("## 🇬🇭 System Status")
    
    # Provider status
    available_providers = settings.get_available_llm_providers()
    current_provider = st.session_state.current_provider
    
    st.sidebar.markdown("### 🤖 LLM Providers")
    for provider in available_providers:
        status = "🟢" if provider == current_provider else "⚪"
        st.sidebar.markdown(f"{status} {provider.title()}")
    
    # Vector store status
    st.sidebar.markdown("### 📚 Knowledge Base")
    try:
        if rag_engine and rag_engine.vector_store:
            st.sidebar.markdown("🟢 Vector store connected")
        else:
            st.sidebar.markdown("🔴 Vector store not connected")
    except:
        st.sidebar.markdown("🔴 Vector store status unknown")
    
    # Ghanaian focus indicators
    st.sidebar.markdown("### 🇬🇭 Ghanaian Focus")
    st.sidebar.markdown(f"🎯 **Sectors:** {', '.join(settings.GHANA_STARTUP_SECTORS)}")
    st.sidebar.markdown(f"📍 **Regions:** {len(settings.GHANA_REGIONS)} regions")
    
    # Performance metrics
    st.sidebar.markdown("### 📊 Performance")
    st.sidebar.markdown(f"⚡ **Strategy:** {settings.RETRIEVAL_STRATEGY}")
    st.sidebar.markdown(f"🔍 **Chunking:** {settings.CHUNKING_STRATEGY}")
    st.sidebar.markdown(f"📈 **Reranking:** {'Enabled' if settings.USE_RERANKING else 'Disabled'}")

def display_user_profile():
    """Display and allow editing of user profile"""
    
    st.sidebar.markdown("## 👤 Your Profile")
    
    # Sector selection
    sector = st.sidebar.selectbox(
        "🎯 Primary Sector",
        options=settings.GHANA_STARTUP_SECTORS,
        index=settings.GHANA_STARTUP_SECTORS.index(st.session_state.user_profile["sector"])
        if st.session_state.user_profile["sector"] in settings.GHANA_STARTUP_SECTORS else 0
    )
    
    # Region selection
    region = st.sidebar.selectbox(
        "📍 Primary Region",
        options=settings.GHANA_REGIONS,
        index=settings.GHANA_REGIONS.index(st.session_state.user_profile["region"])
        if st.session_state.user_profile["region"] in settings.GHANA_REGIONS else 0
    )
    
    # Experience level
    experience_levels = ["beginner", "intermediate", "advanced", "expert"]
    experience = st.sidebar.selectbox(
        "🚀 Experience Level",
        options=experience_levels,
        index=experience_levels.index(st.session_state.user_profile["experience_level"])
    )
    
    # Update profile
    if st.sidebar.button("💾 Update Profile"):
        st.session_state.user_profile.update({
            "sector": sector,
            "region": region,
            "experience_level": experience
        })
        st.sidebar.success("Profile updated!")
    
    # Display current profile
    st.sidebar.markdown("### Current Profile:")
    st.sidebar.markdown(f"**Sector:** {sector}")
    st.sidebar.markdown(f"**Region:** {region}")
    st.sidebar.markdown(f"**Experience:** {experience.title()}")

def process_user_query(query: str) -> Optional[Dict[str, Any]]:
    """Process user query using the RAG engine"""
    
    try:
        # Get user context
        user_context = {
            "sector": st.session_state.user_profile["sector"],
            "region": st.session_state.user_profile["region"],
            "experience_level": st.session_state.user_profile["experience_level"],
            "current_provider": st.session_state.current_provider
        }
        
        # Process query
        if rag_engine:
            response = rag_engine.query(
                question=query,
                conversation_history=st.session_state.chat_history,
                user_context=user_context,
                provider=st.session_state.current_provider
            )
            return response
        else:
            st.error("RAG engine not available")
            return None
            
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        st.error(f"Error processing your question: {str(e)}")
        return None

def display_response(response: Dict[str, Any]):
    """Display the RAG response with Ghanaian styling"""
    
    if not response:
        return
    
    # Display main answer
    st.markdown("### 🤖 UbuntuAI Response")
    
    answer_container = st.container()
    with answer_container:
        st.markdown(response.get("answer", "No answer provided"))
    
    # Display confidence score
    confidence = response.get("confidence", 0)
    if confidence > 0:
        st.markdown(f"**Confidence:** {confidence:.1%}")
    
    # Display sources
    sources = response.get("sources", [])
    if sources:
        st.markdown("### 📚 Sources")
        for i, source in enumerate(sources[:5]):  # Limit to top 5 sources
            with st.expander(f"Source {i+1}: {source.get('title', 'Unknown')}"):
                st.markdown(f"**Content:** {source.get('content', 'No content')[:200]}...")
                st.markdown(f"**Relevance:** {source.get('relevance', 'Unknown')}")
    
    # Display follow-up questions
    follow_ups = response.get("follow_up_questions", [])
    if follow_ups:
        st.markdown("### 💡 Suggested Follow-up Questions")
        for i, question in enumerate(follow_ups):
            if st.button(f"❓ {question}", key=f"followup_{i}"):
                st.session_state.user_input = question
                st.rerun()
    
    # Display processing metadata
    with st.expander("🔍 Processing Details"):
        st.markdown(f"**Provider:** {response.get('provider', 'Unknown')}")
        st.markdown(f"**Processing Time:** {response.get('processing_time', 'Unknown')}")
        st.markdown(f"**Chunks Retrieved:** {response.get('chunks_retrieved', 'Unknown')}")

def main():
    """Main application function"""
    
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🇬🇭 UbuntuAI</h1>
        <p>Your AI Assistant for the Ghanaian Startup Ecosystem</p>
        <div style="margin-top: 1rem;">
            <span class="sector-badge sector-fintech">Fintech</span>
            <span class="sector-badge sector-agritech">Agritech</span>
            <span class="sector-badge sector-healthtech">Healthtech</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        display_system_status()
        st.markdown("---")
        display_user_profile()
        
        # Provider selection
        st.markdown("## 🔄 Switch Provider")
        available_providers = settings.get_available_llm_providers()
        if available_providers:
            new_provider = st.selectbox(
                "Select LLM Provider",
                options=available_providers,
                index=available_providers.index(st.session_state.current_provider)
                if st.session_state.current_provider in available_providers else 0
            )
            
            if new_provider != st.session_state.current_provider:
                st.session_state.current_provider = new_provider
                st.success(f"Switched to {new_provider.title()}")
    
    # Main chat interface
    st.markdown("## 💬 Ask About Ghanaian Startups")
    
    # Chat input
    user_input = st.text_area(
        "Ask me anything about fintech, agritech, or healthtech in Ghana...",
        height=100,
        placeholder="e.g., What are the key regulations for fintech startups in Ghana?",
        key="user_input"
    )
    
    # Process button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        process_button = st.button("🚀 Ask UbuntuAI", use_container_width=True)
    
    # Process query
    if process_button and user_input:
        # Validate input
        is_valid, validation_message = validate_user_input(user_input)
        
        if not is_valid:
            st.warning(validation_message)
            return
        
        # Add user message to chat history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now().isoformat()
        })
        
        # Show processing message
        with st.spinner("🤔 Thinking about your Ghanaian startup question..."):
            # Process query
            response = process_user_query(user_input)
            
            if response:
                # Add assistant response to chat history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response.get("answer", ""),
                    "timestamp": datetime.now().isoformat()
                })
                
                # Display response
                display_response(response)
                
                # Clear input
                st.session_state.user_input = ""
            else:
                st.error("Sorry, I couldn't process your question. Please try again.")
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("## 📝 Chat History")
        
        for i, message in enumerate(st.session_state.chat_history):
            if message["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>👤 You:</strong> {message["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <strong>🤖 UbuntuAI:</strong> {message["content"]}
                </div>
                """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>🇬🇭 <strong>UbuntuAI</strong> - Empowering Ghanaian Entrepreneurs</p>
        <p>Focusing on Fintech, Agritech, and Healthtech</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()