import streamlit as st
import sys
import os

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from api.rag_engine import rag_engine
from knowledge_base.funding_database import funding_db
from knowledge_base.regulatory_info import regulatory_db
from config.settings import settings
from config.prompts import prompt_templates
import time
from typing import List, Dict, Any

# Page configuration
st.set_page_config(
    page_title=settings.APP_TITLE,
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    
    .assistant-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    
    .source-box {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #4caf50;
        margin: 0.5rem 0;
    }
    
    .follow-up-button {
        background-color: #ffffff;
        border: 1px solid #ddd;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        margin: 0.25rem;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .follow-up-button:hover {
        background-color: #f0f0f0;
        border-color: #2196f3;
    }
    
    .stats-container {
        display: flex;
        justify-content: space-around;
        margin: 1rem 0;
    }
    
    .stat-box {
        text-align: center;
        padding: 1rem;
        background-color: #f8f9fa;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

if "user_context" not in st.session_state:
    st.session_state.user_context = {}

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🌍 UbuntuAI</h1>
        <p>Your AI-powered guide to African entrepreneurship and business opportunities</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Settings")
        
        # User context settings
        st.subheader("Your Context")
        user_country = st.selectbox(
            "Your Country/Region of Interest:",
            [""] + settings.AFRICAN_COUNTRIES,
            help="This helps personalize responses to your region"
        )
        
        user_sector = st.selectbox(
            "Your Business Sector:",
            [""] + settings.BUSINESS_SECTORS,
            help="Your area of business focus"
        )
        
        business_stage = st.selectbox(
            "Business Stage:",
            ["", "Idea Stage", "Early Stage", "Growth Stage", "Established"],
            help="Your current business development stage"
        )
        
        # Update user context
        st.session_state.user_context = {
            "country": user_country if user_country else None,
            "sector": user_sector if user_sector else None,
            "business_stage": business_stage if business_stage else None
        }
        
        st.divider()
        
        # Quick actions
        st.subheader("🚀 Quick Access")
        
        if st.button("💰 Browse Funding Database", use_container_width=True):
            st.session_state.current_page = "funding"
            st.rerun()
        
        if st.button("📋 Regulatory Guide", use_container_width=True):
            st.session_state.current_page = "regulatory"
            st.rerun()
        
        if st.button("💬 Reset Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.conversation_history = []
            st.rerun()
        
        st.divider()
        
        # System stats
        st.subheader("📊 System Info")
        try:
            from api.vector_store import vector_store
            stats = vector_store.get_collection_stats()
            st.metric("Knowledge Base Size", f"{stats.get('total_documents', 0)} documents")
        except:
            st.metric("Knowledge Base Size", "Loading...")
    
    # Main content area
    current_page = getattr(st.session_state, 'current_page', 'chat')
    
    if current_page == 'funding':
        show_funding_page()
    elif current_page == 'regulatory':
        show_regulatory_page()
    else:
        show_chat_page()

def show_chat_page():
    st.header("💬 Ask UbuntuAI Anything")
    
    # Quick start suggestions
    if not st.session_state.messages:
        st.subheader("🌟 Popular Questions")
        
        cols = st.columns(2)
        suggestions = prompt_templates.get_conversation_starter_prompts()
        
        for i, suggestion in enumerate(suggestions[:6]):
            col = cols[i % 2]
            with col:
                if st.button(suggestion, key=f"suggestion_{i}", use_container_width=True):
                    process_user_input(suggestion)
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                st.markdown(message["content"])
                
                # Show sources if available
                if "sources" in message and message["sources"]:
                    with st.expander(f"📚 Sources ({len(message['sources'])})"):
                        for i, source in enumerate(message["sources"]):
                            st.markdown(f"""
                            <div class="source-box">
                                <strong>Source {i+1}</strong> (Similarity: {source.get('similarity', 'N/A')})<br>
                                {source['content_preview']}
                                <br><small>{source.get('metadata', {})}</small>
                            </div>
                            """, unsafe_allow_html=True)
                
                # Show follow-up questions
                if "follow_ups" in message and message["follow_ups"]:
                    st.markdown("**💡 You might also ask:**")
                    for j, follow_up in enumerate(message["follow_ups"]):
                        if st.button(follow_up, key=f"followup_{len(st.session_state.messages)}_{j}"):
                            process_user_input(follow_up)
            else:
                st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input(settings.CHAT_PLACEHOLDER):
        process_user_input(prompt)

def process_user_input(user_input: str):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.conversation_history.append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            try:
                response = rag_engine.query(
                    question=user_input,
                    conversation_history=st.session_state.conversation_history,
                    user_context=st.session_state.user_context
                )
                
                answer = response.get("answer", "I'm sorry, I couldn't generate a response.")
                sources = response.get("sources", [])
                follow_ups = response.get("follow_up_questions", [])
                confidence = response.get("confidence", 0.0)
                
                # Display answer
                st.markdown(answer)
                
                # Display confidence if low
                if confidence < 0.5:
                    st.warning(f"⚠️ Low confidence response ({confidence:.2f}). Please verify information from additional sources.")
                
                # Store assistant message with metadata
                assistant_message = {
                    "role": "assistant", 
                    "content": answer,
                    "sources": sources,
                    "follow_ups": follow_ups,
                    "confidence": confidence
                }
                
                st.session_state.messages.append(assistant_message)
                st.session_state.conversation_history.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                error_msg = f"I apologize, but I encountered an error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    st.rerun()

def show_funding_page():
    st.header("💰 African Funding Database")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🔍 Search Filters")
        
        search_country = st.selectbox(
            "Country:",
            ["All"] + settings.AFRICAN_COUNTRIES
        )
        
        search_sector = st.selectbox(
            "Sector:",
            ["All"] + settings.BUSINESS_SECTORS
        )
        
        search_stage = st.selectbox(
            "Funding Stage:",
            ["All"] + settings.FUNDING_STAGES
        )
        
        funding_type = st.selectbox(
            "Type:",
            ["All", "VC Firm", "Accelerator", "Grant", "Government Fund"]
        )
        
        if st.button("🔎 Search", use_container_width=True):
            filters = {}
            if search_country != "All":
                filters["country"] = search_country
            if search_sector != "All":
                filters["sector"] = search_sector
            if search_stage != "All":
                filters["stage"] = search_stage
            if funding_type != "All":
                filters["funding_type"] = funding_type
            
            st.session_state.funding_results = funding_db.search_funding(**filters)
    
    with col2:
        st.subheader("📊 Results")
        
        if hasattr(st.session_state, 'funding_results'):
            results = st.session_state.funding_results
            
            if results:
                st.success(f"Found {len(results)} funding opportunities")
                
                for result in results:
                    with st.expander(f"**{result['name']}** - {result.get('type', 'N/A')}"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**Country:** {result.get('country', 'N/A')}")
                            st.write(f"**Focus Sectors:** {', '.join(result.get('focus_sectors', []))}")
                            st.write(f"**Stages:** {', '.join(result.get('stage', []))}")
                        
                        with col2:
                            st.write(f"**Investment Range:** {result.get('typical_investment', 'N/A')}")
                            if result.get('website'):
                                st.write(f"**Website:** [{result['website']}]({result['website']})")
                        
                        st.write(f"**Description:** {result.get('description', '')}")
                        
                        if result.get('application_process'):
                            st.info(f"**Application:** {result['application_process']}")
            else:
                st.info("No funding opportunities match your search criteria.")
        else:
            # Show default overview
            st.info("Use the filters on the left to search for funding opportunities, or browse the overview below.")
            
            # Show funding statistics
            total_opportunities = len(funding_db.funding_opportunities)
            by_type = {}
            for opp in funding_db.funding_opportunities:
                opp_type = opp.get('type', 'Other')
                by_type[opp_type] = by_type.get(opp_type, 0) + 1
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Opportunities", total_opportunities)
            
            with col2:
                st.metric("VC Firms", by_type.get('VC Firm', 0))
            
            with col3:
                st.metric("Accelerators", by_type.get('Accelerator', 0))
    
    if st.button("🏠 Back to Chat", use_container_width=True):
        st.session_state.current_page = "chat"
        st.rerun()

def show_regulatory_page():
    st.header("📋 Business Regulatory Guide")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🌍 Select Country")
        
        available_countries = list(regulatory_db.business_registration.keys())
        selected_country = st.selectbox(
            "Country:",
            available_countries
        )
        
        if st.button("📖 Generate Guide", use_container_width=True):
            st.session_state.selected_country = selected_country
    
    with col2:
        if hasattr(st.session_state, 'selected_country'):
            country = st.session_state.selected_country
            guide = regulatory_db.generate_country_guide(country)
            
            st.markdown(guide)
            
            # Additional resources
            st.subheader("📚 Additional Resources")
            
            reg_info = regulatory_db.get_business_registration_info(country)
            if reg_info.get('online_portal'):
                st.write(f"🌐 **Official Portal:** [{reg_info['online_portal']}]({reg_info['online_portal']})")
        else:
            st.info("Select a country from the left panel to view the business registration guide.")
            
            # Show overview
            st.subheader("🌍 Available Countries")
            
            countries = list(regulatory_db.business_registration.keys())
            cols = st.columns(2)
            
            for i, country in enumerate(countries):
                col = cols[i % 2]
                with col:
                    if st.button(f"📍 {country}", key=f"country_{i}", use_container_width=True):
                        st.session_state.selected_country = country
                        st.rerun()
    
    if st.button("🏠 Back to Chat", use_container_width=True):
        st.session_state.current_page = "chat"
        st.rerun()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"""
        ## ⚠️ Application Error
        
        There was an error running UbuntuAI: {str(e)}
        
        **Common solutions:**
        1. Make sure your OpenAI API key is set in the `.env` file
        2. Install all required dependencies: `pip install -r requirements.txt`
        3. Check that all files are in the correct directory structure
        
        **For support:** Please check the documentation or contact support.
        """)
        
        if st.button("🔄 Retry"):
            st.rerun()