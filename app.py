#!/usr/bin/env python3
"""
UbuntuAI - African Business Intelligence Platform
Streamlit application for Ghanaian startup ecosystem insights
"""

import streamlit as st
import os
import sys
import logging
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="UbuntuAI - African Business Intelligence",
    page_icon="🇬🇭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .ghana-flag {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    .sector-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .confidence-high { color: #28a745; }
    .confidence-medium { color: #ffc107; }
    .confidence-low { color: #dc3545; }
</style>
""", unsafe_allow_html=True)

def initialize_app():
    """Initialize the application and check dependencies"""
    try:
        # Import required modules
        from config.settings import settings
        from api.vector_store import vector_store
        from api.rag_engine import rag_engine
        from utils.embeddings import embedding_service
        
        # Check if services are available
        services_status = {
            "settings": settings is not None,
            "vector_store": vector_store is not None,
            "rag_engine": rag_engine is not None,
            "embedding_service": embedding_service is not None
        }
        
        # Check if all required services are available
        all_services_available = all(services_status.values())
        
        if not all_services_available:
            st.error("⚠️ Some services failed to initialize. Please check your configuration.")
            st.write("Services status:", services_status)
            return False
        
        st.success("✅ All services initialized successfully!")
        return True
        
    except Exception as e:
        st.error(f"❌ Failed to initialize application: {e}")
        logger.error(f"Application initialization failed: {e}")
        return False

def display_header():
    """Display the application header"""
    st.markdown("""
    <div class="main-header">
        <div class="ghana-flag">🇬🇭</div>
        <h1>UbuntuAI - African Business Intelligence Platform</h1>
        <p>Empowering Ghanaian entrepreneurs with AI-driven insights in Fintech, Agritech, and Healthtech</p>
    </div>
    """, unsafe_allow_html=True)

def display_sidebar():
    """Display the sidebar with navigation and information"""
    st.sidebar.title("🇬🇭 UbuntuAI")
    st.sidebar.markdown("---")
    
    # Navigation
    page = st.sidebar.selectbox(
        "Navigation",
        ["🏠 Home", "🤖 AI Chat", "📚 Knowledge Base", "📊 Analytics", "⚙️ Settings"]
    )
    
    st.sidebar.markdown("---")
    
    # Ghanaian sectors
    st.sidebar.subheader("🎯 Focus Sectors")
    sectors = ["Fintech", "Agritech", "Healthtech", "Edtech", "Logistics", "E-commerce"]
    for sector in sectors:
        st.sidebar.markdown(f"• {sector}")
    
    # Ghanaian regions
    st.sidebar.subheader("📍 Coverage")
    st.sidebar.markdown("All 16 regions of Ghana")
    
    st.sidebar.markdown("---")
    
    # System status
    st.sidebar.subheader("🔧 System Status")
    try:
        from api.vector_store import vector_store
        if vector_store:
            stats = vector_store.get_collection_stats()
            st.sidebar.success(f"Vector Store: ✅")
            st.sidebar.info(f"Documents: {stats.get('total_documents', 0)}")
        else:
            st.sidebar.error("Vector Store: ❌")
    except:
        st.sidebar.error("Vector Store: ❌")
    
    return page

def display_home_page():
    """Display the home page"""
    st.header("🏠 Welcome to UbuntuAI")
    
    # Introduction
    st.markdown("""
    UbuntuAI is your intelligent companion for navigating Ghana's vibrant startup ecosystem. 
    We focus on three key sectors that are driving innovation and economic growth in Ghana:
    """)
    
    # Sector cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="sector-card">
            <h3>🏦 Fintech</h3>
            <p>Digital payments, mobile money, banking solutions, and financial inclusion technologies</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="sector-card">
            <h3>🌾 Agritech</h3>
            <p>Agricultural technology, smart farming, crop management, and food security solutions</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="sector-card">
            <h3>🏥 Healthtech</h3>
            <p>Healthcare technology, telemedicine, medical devices, and health information systems</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Features
    st.subheader("🚀 Key Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        - **AI-Powered Insights**: Get intelligent answers about Ghanaian business opportunities
        - **Document Analysis**: Upload and analyze business documents, reports, and regulations
        - **Sector Expertise**: Deep knowledge of fintech, agritech, and healthtech in Ghana
        - **Regional Coverage**: Insights covering all 16 regions of Ghana
        """)
    
    with col2:
        st.markdown("""
        - **Real-time Updates**: Stay current with the latest business trends and regulations
        - **Practical Guidance**: Actionable advice for entrepreneurs and investors
        - **WhatsApp Integration**: Access insights on the go via WhatsApp
        - **Multi-language Support**: English and local language support
        """)
    
    # Quick start
    st.subheader("🚀 Quick Start")
    st.markdown("""
    1. **Ask Questions**: Use the AI Chat to ask about business opportunities in Ghana
    2. **Upload Documents**: Add business documents to expand your knowledge base
    3. **Explore Sectors**: Discover insights in fintech, agritech, and healthtech
    4. **Get Analytics**: View trends and patterns in Ghana's startup ecosystem
    """)

def display_chat_page():
    """Display the AI chat page"""
    st.header("🤖 AI Chat - Ask About Ghanaian Business")
    
    # Check if RAG engine is available
    try:
        from api.rag_engine import rag_engine
        if not rag_engine or not rag_engine.is_initialized():
            st.error("❌ RAG engine not available. Please check your configuration.")
            return
    except Exception as e:
        st.error(f"❌ Failed to load RAG engine: {e}")
        return
    
    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about business opportunities in Ghana..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                try:
                    response = rag_engine.query(prompt)
                    
                    if response and 'answer' in response:
                        answer = response['answer']
                        confidence = response.get('confidence', 0.0)
                        sources = response.get('sources', [])
                        
                        # Display answer
                        st.markdown(answer)
                        
                        # Display confidence
                        if confidence >= 0.7:
                            confidence_class = "confidence-high"
                        elif confidence >= 0.4:
                            confidence_class = "confidence-medium"
                        else:
                            confidence_class = "confidence-low"
                        
                        st.markdown(f"<p class='{confidence_class}'><strong>Confidence: {confidence:.1%}</strong></p>", 
                                  unsafe_allow_html=True)
                        
                        # Display sources if available
                        if sources:
                            with st.expander("📚 Sources"):
                                for i, source in enumerate(sources, 1):
                                    st.markdown(f"**Source {i}:** {source.get('source', 'Unknown')}")
                                    st.markdown(f"**Type:** {source.get('document_type', 'Document')}")
                                    st.markdown(f"**Relevance:** {source.get('relevance_score', 'N/A')}")
                                    st.markdown("---")
                        
                        # Add assistant message to chat history
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                        
                    else:
                        st.error("Failed to generate response. Please try again.")
                        
                except Exception as e:
                    st.error(f"Error generating response: {e}")
                    logger.error(f"Chat error: {e}")

def display_knowledge_base_page():
    """Display the knowledge base page"""
    st.header("📚 Knowledge Base")
    
    # Check if vector store is available
    try:
        from api.vector_store import vector_store
        if not vector_store:
            st.error("❌ Vector store not available. Please check your configuration.")
            return
    except Exception as e:
        st.error(f"❌ Failed to load vector store: {e}")
        return
    
    # Get collection stats
    try:
        stats = vector_store.get_collection_stats()
        st.info(f"📊 **Collection:** {stats.get('collection_name', 'N/A')}")
        st.info(f"📄 **Total Documents:** {stats.get('total_documents', 0)}")
        st.info(f"🗂️ **Storage Location:** {stats.get('persist_directory', 'N/A')}")
    except Exception as e:
        st.error(f"Failed to get collection stats: {e}")
    
    # Document upload
    st.subheader("📤 Upload Documents")
    
    uploaded_file = st.file_uploader(
        "Choose a document to add to the knowledge base",
        type=['pdf', 'txt', 'docx', 'xlsx'],
        help="Supported formats: PDF, TXT, DOCX, XLSX"
    )
    
    if uploaded_file is not None:
        if st.button("Add to Knowledge Base"):
            try:
                # Process the uploaded file
                from data.processor import data_processor
                
                # Read file content
                file_content = uploaded_file.read()
                file_name = uploaded_file.name
                
                # Process the document
                document = {
                    'text': file_content.decode('utf-8') if isinstance(file_content, bytes) else str(file_content),
                    'metadata': {
                        'source': file_name,
                        'document_type': file_name.split('.')[-1].upper(),
                        'uploaded_at': datetime.now().isoformat()
                    }
                }
                
                # Add to vector store
                success = vector_store.add_documents([document])
                
                if success:
                    st.success(f"✅ Document '{file_name}' added successfully!")
                    st.rerun()
                else:
                    st.error("❌ Failed to add document to knowledge base.")
                    
            except Exception as e:
                st.error(f"❌ Error processing document: {e}")
                logger.error(f"Document processing error: {e}")
    
    # Search documents
    st.subheader("🔍 Search Knowledge Base")
    
    search_query = st.text_input("Enter search query:")
    if search_query and st.button("Search"):
        try:
            results = vector_store.search(search_query, max_results=5)
            
            if results:
                st.success(f"Found {len(results)} relevant documents:")
                
                for i, result in enumerate(results, 1):
                    with st.expander(f"Document {i}: {result.get('metadata', {}).get('source', 'Unknown')}"):
                        st.markdown(f"**Content:** {result.get('document', '')[:500]}...")
                        st.markdown(f"**Source:** {result.get('metadata', {}).get('source', 'Unknown')}")
                        st.markdown(f"**Type:** {result.get('metadata', {}).get('document_type', 'Document')}")
                        st.markdown(f"**Relevance Score:** {result.get('distance', 'N/A')}")
            else:
                st.warning("No relevant documents found.")
                
        except Exception as e:
            st.error(f"Search failed: {e}")

def display_analytics_page():
    """Display the analytics page"""
    st.header("📊 Analytics & Insights")
    
    # Placeholder for analytics
    st.info("📈 Analytics features are coming soon!")
    
    # Mock data for demonstration
    st.subheader("🎯 Sector Distribution")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Fintech", "35%", "↑ 12%")
    
    with col2:
        st.metric("Agritech", "28%", "↑ 8%")
    
    with col3:
        st.metric("Healthtech", "22%", "↑ 15%")
    
    st.subheader("📍 Regional Activity")
    st.markdown("""
    - **Greater Accra**: 45% of startups
    - **Ashanti**: 18% of startups  
    - **Western**: 12% of startups
    - **Other Regions**: 25% of startups
    """)
    
    st.subheader("📅 Recent Trends")
    st.markdown("""
    - **Mobile Money Adoption**: Growing rapidly across all regions
    - **Agricultural Technology**: Increasing investment in smart farming solutions
    - **Healthcare Innovation**: Telemedicine platforms gaining traction
    - **Digital Payments**: Fintech solutions expanding beyond urban areas
    """)

def display_settings_page():
    """Display the settings page"""
    st.header("⚙️ Settings & Configuration")
    
    # Configuration status
    st.subheader("🔧 System Configuration")
    
    try:
        from config.settings import settings
        
        config_info = settings.to_dict()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"**Primary LLM Provider:** {config_info.get('primary_llm_provider', 'N/A')}")
            st.info(f"**Vector Store Type:** {config_info.get('vector_store_type', 'N/A')}")
            st.info(f"**Retrieval Strategy:** {config_info.get('retrieval_strategy', 'N/A')}")
        
        with col2:
            st.info(f"**Available LLM Providers:** {', '.join(config_info.get('available_llm_providers', []))}")
            st.info(f"**Supported Sectors:** {len(config_info.get('supported_sectors', []))}")
            st.info(f"**Ghana Regions:** {len(config_info.get('ghana_regions', []))}")
        
    except Exception as e:
        st.error(f"Failed to load configuration: {e}")
    
    # Service status
    st.subheader("🔍 Service Status")
    
    services = [
        ("Vector Store", "api.vector_store", "vector_store"),
        ("RAG Engine", "api.rag_engine", "rag_engine"),
        ("Embedding Service", "utils.embeddings", "embedding_service"),
        ("LLM Manager", "api.llm_providers", "llm_manager")
    ]
    
    for service_name, module_path, var_name in services:
        try:
            module = __import__(module_path, fromlist=[var_name])
            service = getattr(module, var_name)
            
            if service:
                st.success(f"✅ {service_name}: Available")
            else:
                st.error(f"❌ {service_name}: Not available")
        except Exception as e:
            st.error(f"❌ {service_name}: Error - {e}")
    
    # Environment variables
    st.subheader("🌍 Environment Variables")
    
    env_vars = [
        "GOOGLE_API_KEY",
        "OPENAI_API_KEY", 
        "ANTHROPIC_API_KEY",
        "TWILIO_ACCOUNT_SID",
        "TWILIO_AUTH_TOKEN"
    ]
    
    for var in env_vars:
        value = os.getenv(var, "Not set")
        if value != "Not set":
            # Mask sensitive values
            display_value = value[:8] + "..." if len(value) > 8 else value
            st.success(f"✅ {var}: {display_value}")
        else:
            st.warning(f"⚠️ {var}: Not set")

def main():
    """Main application function"""
    try:
        # Initialize the application
        if not initialize_app():
            st.error("Application initialization failed. Please check the logs and try again.")
            return
        
        # Display header
        display_header()
        
        # Display sidebar and get selected page
        selected_page = display_sidebar()
        
        # Route to appropriate page
        if selected_page == "🏠 Home":
            display_home_page()
        elif selected_page == "🤖 AI Chat":
            display_chat_page()
        elif selected_page == "📚 Knowledge Base":
            display_knowledge_base_page()
        elif selected_page == "📊 Analytics":
            display_analytics_page()
        elif selected_page == "⚙️ Settings":
            display_settings_page()
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #666;'>
            🇬🇭 UbuntuAI - Empowering Ghanaian Entrepreneurs with AI Intelligence
            <br>Built with ❤️ for Africa's startup ecosystem
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"❌ Application error: {e}")
        logger.error(f"Application error: {e}")
        st.info("Please check the logs for more details and try refreshing the page.")

if __name__ == "__main__":
    main()