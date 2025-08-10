import streamlit as st
import sys
import os
import asyncio
from datetime import datetime
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import SIGMA platform components
from api.rag_engine import rag_engine
from api.langchain_agents import create_ghana_business_agent
from api.scoring_engine import create_scoring_engine
from api.business_model_copilot import create_business_model_copilot
from api.growth_recommender import create_growth_recommender
from api.credit_underwriting import create_credit_underwriting_engine
from api.merl_analytics import create_merl_analytics_engine
from api.mcp_patterns import create_task_orchestrator
from api.streaming_interfaces import create_streaming_interface_manager
from knowledge_base.funding_database import funding_db
from knowledge_base.regulatory_info import regulatory_db
from config.settings import settings
from config.prompts import prompt_templates
import time
from typing import List, Dict, Any

# Page configuration
st.set_page_config(
    page_title=settings.APP_TITLE,
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS for SIGMA Platform
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
    }
    
    .sigma-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
    }
    
    .feature-card {
        background: linear-gradient(145deg, #f0f2f6, #ffffff);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #667eea;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea, #764ba2);
        border-radius: 10px;
        padding: 1rem;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
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
    
    .tool-badge {
        display: inline-block;
        background: #667eea;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        margin: 0.2rem;
    }
    
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-active { background-color: #4CAF50; }
    .status-inactive { background-color: #f44336; }
    .status-warning { background-color: #ff9800; }
    
    .sidebar-section {
        background: rgba(255, 255, 255, 0.05);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for SIGMA platform
if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

if "user_context" not in st.session_state:
    st.session_state.user_context = {}

if "sigma_mode" not in st.session_state:
    st.session_state.sigma_mode = "comprehensive"

if "active_workflow" not in st.session_state:
    st.session_state.active_workflow = None

# Initialize SIGMA platform components
@st.cache_resource
def initialize_sigma_platform():
    """Initialize all SIGMA platform components"""
    
    components = {}
    
    try:
        # Core components
        components['rag_engine'] = rag_engine
        components['scoring_engines'] = create_scoring_engine()
        components['ghana_agent'] = create_ghana_business_agent()
        
        # SIGMA-specific components
        if settings.ENABLE_BUSINESS_MODEL_COPILOT:
            components['business_model_copilot'] = create_business_model_copilot()
        
        if settings.ENABLE_GROWTH_RECOMMENDER:
            components['growth_recommender'] = create_growth_recommender()
        
        if settings.ENABLE_CREDIT_UNDERWRITING:
            components['credit_engine'] = create_credit_underwriting_engine()
        
        if settings.ENABLE_MERL_ANALYTICS:
            components['merl_engine'] = create_merl_analytics_engine()
        
        if settings.ENABLE_MCP_PATTERNS:
            components['task_orchestrator'] = create_task_orchestrator()
        
        components['streaming_manager'] = create_streaming_interface_manager()
        
        return components
        
    except Exception as e:
        st.error(f"Error initializing SIGMA platform: {e}")
        return {}

# Load SIGMA components
sigma_components = initialize_sigma_platform()

def main():
    # SIGMA Platform Header
    st.markdown(f"""
    <div class="main-header">
        <h1>🚀 {settings.APP_TITLE}</h1>
        <p>{settings.APP_DESCRIPTION}</p>
        <p><strong>Powered by AI • Built for Africa • Focused on Impact</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.header("🎛️ SIGMA Control Panel")
        
        # Platform status indicators
        st.subheader("Platform Status")
        display_platform_status()
        
        st.divider()
        
        # SIGMA mode selection
        st.subheader("AI Mode Selection")
        sigma_mode = st.selectbox(
            "Choose AI Mode:",
            ["comprehensive", "business_model", "credit_assessment", "growth_strategy", "impact_measurement"],
            index=0,
            help="Select the SIGMA AI mode for specialized assistance"
        )
        st.session_state.sigma_mode = sigma_mode
        
        st.divider()
        
        # User context settings
        st.subheader("Your Profile")
        user_profile = collect_user_profile()
        st.session_state.user_context.update(user_profile)
        
        st.divider()
        
        # Quick actions
        st.subheader("Quick Actions")
        display_quick_actions()
        
        st.divider()
        
        # Platform analytics
        st.subheader("Platform Analytics")
        display_platform_metrics()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main content area - tabbed interface
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "💬 AI Assistant", 
        "🏢 Business Tools", 
        "💰 Financing", 
        "📊 Impact Dashboard", 
        "🔧 Platform Tools"
    ])
    
    with tab1:
        show_ai_assistant()
    
    with tab2:
        show_business_tools()
    
    with tab3:
        show_financing_tools()
    
    with tab4:
        show_impact_dashboard()
    
    with tab5:
        show_platform_tools()

def display_platform_status():
    """Display real-time platform status"""
    
    # Check component status
    components_status = {
        "RAG Engine": "rag_engine" in sigma_components,
        "Business Model Copilot": settings.ENABLE_BUSINESS_MODEL_COPILOT and "business_model_copilot" in sigma_components,
        "Growth Recommender": settings.ENABLE_GROWTH_RECOMMENDER and "growth_recommender" in sigma_components,
        "Credit Engine": settings.ENABLE_CREDIT_UNDERWRITING and "credit_engine" in sigma_components,
        "MERL Analytics": settings.ENABLE_MERL_ANALYTICS and "merl_engine" in sigma_components,
        "Streaming Interfaces": "streaming_manager" in sigma_components
    }
    
    for component, status in components_status.items():
        status_class = "status-active" if status else "status-inactive"
        st.markdown(
            f'<span class="status-indicator {status_class}"></span>{component}',
            unsafe_allow_html=True
        )

def collect_user_profile():
    """Collect user profile information"""
    
    profile = {}
    
    # Country selection with SIGMA priority
    country_options = settings.SIGMA_PRIORITY_COUNTRIES + [
        c for c in settings.AFRICAN_COUNTRIES if c not in settings.SIGMA_PRIORITY_COUNTRIES
    ]
    
    profile["country"] = st.selectbox(
        "Country/Region:",
        [""] + country_options,
        help="Your primary business location"
    )
    
    # Enhanced sector selection
    profile["sector"] = st.selectbox(
        "Business Sector:",
        [""] + settings.BUSINESS_SECTORS,
        help="Your primary business sector"
    )
    
    # Business stage with SIGMA focus
    profile["business_stage"] = st.selectbox(
        "Business Stage:",
        ["", "Idea Stage", "Early Stage (MVP)", "Growth Stage", "Scaling", "Established"],
        help="Current stage of your business"
    )
    
    # SIGMA target user identification
    profile["user_type"] = st.multiselect(
        "Business Characteristics:",
        ["Women-led", "Informal Enterprise", "Small Agribusiness", "Tech-enabled", "Export-oriented"],
        help="Select characteristics that apply to your business"
    )
    
    # Funding status
    profile["funding_stage"] = st.selectbox(
        "Funding Status:",
        [""] + settings.FUNDING_STAGES,
        help="Current funding stage or type needed"
    )
    
    return {k: v for k, v in profile.items() if v}

def display_quick_actions():
    """Display quick action buttons"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🎯 Business Assessment", use_container_width=True):
            st.session_state.active_page = "business_assessment"
            st.rerun()
        
        if st.button("💰 Credit Check", use_container_width=True):
            st.session_state.active_page = "credit_assessment"
            st.rerun()
    
    with col2:
        if st.button("📈 Growth Plan", use_container_width=True):
            st.session_state.active_page = "growth_planning"
            st.rerun()
        
        if st.button("📊 Impact Report", use_container_width=True):
            st.session_state.active_page = "impact_measurement"
            st.rerun()

def display_platform_metrics():
    """Display key platform metrics"""
    
    try:
        # Simulated metrics (in production, these would come from actual usage data)
        metrics = {
            "Active Users": 1247,
            "Businesses Assessed": 892,
            "Credit Applications": 234,
            "Impact Score Avg": 7.8
        }
        
        for metric, value in metrics.items():
            st.metric(metric, value)
            
    except Exception as e:
        st.error(f"Error loading metrics: {e}")

def show_ai_assistant():
    """Main AI assistant interface"""
    
    st.header("🤖 SIGMA AI Assistant")
    
    # Mode indicator
    mode_descriptions = {
        "comprehensive": "Full SIGMA platform capabilities with all tools",
        "business_model": "Specialized business model design and optimization",
        "credit_assessment": "Credit scoring and loan recommendation focus",
        "growth_strategy": "Growth planning and strategy development",
        "impact_measurement": "Impact measurement and MERL analytics"
    }
    
    st.info(f"**Current Mode:** {st.session_state.sigma_mode.title()} - {mode_descriptions[st.session_state.sigma_mode]}")
    
    # Quick start suggestions based on mode
    if not st.session_state.messages:
        show_mode_specific_suggestions()
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                st.markdown(message["content"])
                
                # Show SIGMA-specific enhancements
                if message.get("sigma_enhanced"):
                    show_sigma_message_enhancements(message)
                
            else:
                st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input(settings.CHAT_PLACEHOLDER):
        process_sigma_input(prompt)

def show_mode_specific_suggestions():
    """Show suggestions based on current SIGMA mode"""
    
    st.subheader("💡 Mode-Specific Suggestions")
    
    suggestions = {
        "comprehensive": [
            "Assess my agritech startup readiness and recommend next steps",
            "Design a business model for rural fintech services in Ghana",
            "Create a complete growth strategy with financing recommendations",
            "Build an impact measurement framework for my social enterprise"
        ],
        "business_model": [
            "Design a business model for mobile money services in rural areas",
            "Optimize my e-commerce platform for African markets",
            "Create a subscription model for agricultural advisory services",
            "Develop a marketplace model for women entrepreneurs"
        ],
        "credit_assessment": [
            "Assess my creditworthiness for a $50,000 business loan",
            "Evaluate loan risk for a women-led agribusiness",
            "Recommend optimal loan terms for seasonal farming business",
            "Analyze alternative data for informal enterprise credit scoring"
        ],
        "growth_strategy": [
            "Create a growth plan for scaling my fintech startup",
            "Recommend expansion strategies for my agritech platform",
            "Develop customer acquisition strategy for rural markets",
            "Design operational scaling plan for my business"
        ],
        "impact_measurement": [
            "Create impact measurement framework for farmer income improvement",
            "Design MERL system for women empowerment program",
            "Calculate social return on investment for my social enterprise",
            "Build beneficiary feedback collection system"
        ]
    }
    
    mode_suggestions = suggestions.get(st.session_state.sigma_mode, suggestions["comprehensive"])
    
    cols = st.columns(2)
    for i, suggestion in enumerate(mode_suggestions):
        col = cols[i % 2]
        with col:
            if st.button(suggestion, key=f"suggestion_{i}", use_container_width=True):
                process_sigma_input(suggestion)

def process_sigma_input(user_input: str):
    """Process user input through SIGMA platform"""
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.conversation_history.append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Generate SIGMA response
    with st.chat_message("assistant"):
        with st.spinner("SIGMA is thinking..."):
            try:
                response = generate_sigma_response(user_input)
                
                # Display response
                st.markdown(response["content"])
                
                # Store response
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response["content"],
                    "sigma_enhanced": True,
                    "mode": st.session_state.sigma_mode,
                    "tools_used": response.get("tools_used", []),
                    "confidence": response.get("confidence", 0.0),
                    "processing_time": response.get("processing_time", 0.0)
                })
                
                st.session_state.conversation_history.append({
                    "role": "assistant", 
                    "content": response["content"]
                })
                
                # Show SIGMA enhancements
                show_sigma_response_enhancements(response)
                
            except Exception as e:
                error_msg = f"I apologize, but I encountered an error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    st.rerun()

def generate_sigma_response(user_input: str) -> Dict[str, Any]:
    """Generate response using SIGMA platform capabilities"""
    
    start_time = time.time()
    
    try:
        mode = st.session_state.sigma_mode
        user_context = st.session_state.user_context
        
        if mode == "business_model" and "business_model_copilot" in sigma_components:
            return generate_business_model_response(user_input, user_context)
        
        elif mode == "credit_assessment" and "credit_engine" in sigma_components:
            return generate_credit_response(user_input, user_context)
        
        elif mode == "growth_strategy" and "growth_recommender" in sigma_components:
            return generate_growth_response(user_input, user_context)
        
        elif mode == "impact_measurement" and "merl_engine" in sigma_components:
            return generate_impact_response(user_input, user_context)
        
        else:
            # Comprehensive mode using agent orchestration
            return generate_comprehensive_response(user_input, user_context)
    
    except Exception as e:
        # Fallback to basic RAG
        rag_response = rag_engine.query(
            question=user_input,
            conversation_history=st.session_state.conversation_history,
            user_context=user_context
        )
        
        return {
            "content": rag_response.get("answer", "I apologize, but I couldn't generate a response."),
            "tools_used": ["rag_engine"],
            "confidence": rag_response.get("confidence", 0.5),
            "processing_time": time.time() - start_time,
            "fallback": True
        }

def generate_business_model_response(user_input: str, user_context: Dict[str, Any]) -> Dict[str, Any]:
    """Generate response using business model copilot"""
    
    copilot = sigma_components["business_model_copilot"]
    
    # Extract business data from context and input
    entrepreneur_data = {
        "sector": user_context.get("sector", "general"),
        "location": user_context.get("country", "Ghana"),
        "business_stage": user_context.get("business_stage", "early"),
        "user_type": user_context.get("user_type", []),
        "query": user_input
    }
    
    # Design business model
    recommendation = copilot.design_business_model(
        entrepreneur_data, 
        st.session_state.conversation_history
    )
    
    # Format response
    response_content = f"""## Business Model Recommendation

**Model Type:** {recommendation.model_type}

**Revenue Streams:**
{chr(10).join(f'• {stream}' for stream in recommendation.revenue_streams)}

**Key Value Propositions:**
{chr(10).join(f'• {prop}' for prop in recommendation.value_propositions)}

**Target Customer Segments:**
{chr(10).join(f'• {segment}' for segment in recommendation.customer_segments)}

**Implementation Steps:**
{chr(10).join(f'{i+1}. {step}' for i, step in enumerate(recommendation.implementation_steps[:5]))}

**Confidence Level:** {recommendation.confidence_score:.2f}/1.0
"""
    
    return {
        "content": response_content,
        "tools_used": ["business_model_copilot"],
        "confidence": recommendation.confidence_score,
        "processing_time": time.time() - time.time(),
        "business_model_data": recommendation
    }

def generate_comprehensive_response(user_input: str, user_context: Dict[str, Any]) -> Dict[str, Any]:
    """Generate comprehensive response using agent orchestration"""
    
    if "task_orchestrator" in sigma_components:
        orchestrator = sigma_components["task_orchestrator"]
        
        # Use workflow orchestration for complex queries
        workflow_result = orchestrator.execute_workflow(
            "entrepreneur_onboarding",
            user_context.get("user_id", "default_user"),
            {"query": user_input, "context": user_context}
        )
        
        if workflow_result["status"] == "completed":
            return format_workflow_response(workflow_result)
    
    # Fallback to enhanced agent
    agent_result = sigma_components["ghana_agent"].process_query_sync(
        user_input, user_context
    )
    
    return {
        "content": agent_result.get("answer", ""),
        "tools_used": agent_result.get("tools_used", []),
        "confidence": 0.8,
        "processing_time": time.time() - time.time()
    }

def show_business_tools():
    """Show business tools interface"""
    
    st.header("🏢 Business Development Tools")
    
    tool_tab1, tool_tab2, tool_tab3, tool_tab4 = st.tabs([
        "Business Model Designer", 
        "Readiness Assessment", 
        "Growth Planner", 
        "Regulatory Guide"
    ])
    
    with tool_tab1:
        show_business_model_designer()
    
    with tool_tab2:
        show_readiness_assessment()
    
    with tool_tab3:
        show_growth_planner()
    
    with tool_tab4:
        show_regulatory_guide()

def show_financing_tools():
    """Show financing tools interface"""
    
    st.header("💰 Financing & Credit Tools")
    
    fin_tab1, fin_tab2, fin_tab3 = st.tabs([
        "Credit Assessment", 
        "Funding Database", 
        "Loan Structuring"
    ])
    
    with fin_tab1:
        show_credit_assessment_tool()
    
    with fin_tab2:
        show_funding_database()
    
    with fin_tab3:
        show_loan_structuring_tool()

def show_impact_dashboard():
    """Show impact measurement dashboard"""
    
    st.header("📊 Impact Measurement Dashboard")
    
    impact_tab1, impact_tab2, impact_tab3 = st.tabs([
        "Impact Framework", 
        "MERL Analytics", 
        "SDG Mapping"
    ])
    
    with impact_tab1:
        show_impact_framework_builder()
    
    with impact_tab2:
        show_merl_analytics()
    
    with impact_tab3:
        show_sdg_mapping()

def show_platform_tools():
    """Show platform administration tools"""
    
    st.header("🔧 Platform Administration")
    
    if st.checkbox("Show Advanced Tools", help="Enable advanced platform tools"):
        
        admin_tab1, admin_tab2, admin_tab3 = st.tabs([
            "System Status", 
            "Configuration", 
            "Analytics"
        ])
        
        with admin_tab1:
            show_system_status()
        
        with admin_tab2:
            show_configuration_panel()
        
        with admin_tab3:
            show_platform_analytics()

def show_business_model_designer():
    """Business model designer interface"""
    
    st.subheader("🎨 Business Model Designer")
    
    if "business_model_copilot" not in sigma_components:
        st.warning("Business Model Copilot not available. Please check configuration.")
        return
    
    with st.form("business_model_form"):
        st.write("**Design your business model with AI assistance**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            business_desc = st.text_area(
                "Business Description",
                placeholder="Describe your business idea or current model...",
                height=100
            )
            
            target_market = st.text_input(
                "Target Market",
                placeholder="Who are your customers?"
            )
            
            value_prop = st.text_area(
                "Value Proposition",
                placeholder="What value do you provide?",
                height=80
            )
        
        with col2:
            revenue_model = st.selectbox(
                "Preferred Revenue Model",
                ["Subscription", "Transaction-based", "Commission", "Advertising", "Freemium", "Marketplace"]
            )
            
            market_size = st.selectbox(
                "Market Size",
                ["Local", "Regional", "National", "Continental", "Global"]
            )
            
            business_complexity = st.slider(
                "Business Complexity",
                1, 5, 3,
                help="1=Simple, 5=Complex"
            )
        
        submitted = st.form_submit_button("Design Business Model", use_container_width=True)
        
        if submitted and business_desc:
            with st.spinner("Designing your business model..."):
                
                entrepreneur_data = {
                    "business_description": business_desc,
                    "target_market": target_market,
                    "value_proposition": value_prop,
                    "revenue_model": revenue_model,
                    "market_size": market_size,
                    "complexity": business_complexity,
                    **st.session_state.user_context
                }
                
                copilot = sigma_components["business_model_copilot"]
                recommendation = copilot.design_business_model(entrepreneur_data)
                
                # Display comprehensive business model
                display_business_model_recommendation(recommendation)

def show_credit_assessment_tool():
    """Credit assessment tool interface"""
    
    st.subheader("💳 Credit Assessment Tool")
    
    if "credit_engine" not in sigma_components:
        st.warning("Credit Underwriting Engine not available. Please check configuration.")
        return
    
    with st.form("credit_assessment_form"):
        st.write("**Comprehensive credit assessment for African markets**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Personal Information**")
            age = st.slider("Age", 18, 70, 35)
            gender = st.selectbox("Gender", ["Female", "Male", "Other"])
            education = st.selectbox("Education Level", 
                ["Primary", "Secondary", "Tertiary", "Vocational", "Graduate"])
        
        with col2:
            st.write("**Business Information**")
            business_age = st.slider("Business Age (months)", 0, 120, 12)
            monthly_revenue = st.number_input("Monthly Revenue (USD)", 0, 50000, 1000)
            employees = st.slider("Number of Employees", 0, 50, 2)
        
        with col3:
            st.write("**Financial Information**")
            loan_amount = st.number_input("Requested Loan Amount (USD)", 1000, 500000, 25000)
            loan_term = st.slider("Loan Term (months)", 6, 60, 12)
            collateral = st.checkbox("Has Collateral Available")
        
        # Alternative data section
        st.write("**Alternative Data (Optional)**")
        col4, col5 = st.columns(2)
        
        with col4:
            mobile_money = st.checkbox("Regular Mobile Money Usage")
            community_leader = st.checkbox("Community Leader/Member")
            group_member = st.checkbox("Member of Business Group/Cooperative")
        
        with col5:
            social_refs = st.slider("Social References Available", 0, 10, 3)
            bank_account = st.checkbox("Has Bank Account")
            credit_history = st.checkbox("Has Formal Credit History")
        
        submitted = st.form_submit_button("Assess Credit", use_container_width=True)
        
        if submitted:
            with st.spinner("Analyzing creditworthiness..."):
                
                # Prepare assessment data
                applicant_data = {
                    "age": age,
                    "gender": gender.lower(),
                    "education_level": education,
                    "business_age_months": business_age,
                    "monthly_revenue": monthly_revenue,
                    "team_size": employees,
                    "has_collateral": collateral,
                    "location": st.session_state.user_context.get("country", "Ghana"),
                    "sector": st.session_state.user_context.get("sector", "general")
                }
                
                loan_request = {
                    "amount": loan_amount,
                    "term_months": loan_term,
                    "purpose": "business_expansion"
                }
                
                alternative_data = {
                    "mobile_money_data": {
                        "account_age_months": 24 if mobile_money else 0,
                        "transaction_frequency": 15 if mobile_money else 0
                    },
                    "social_network_data": {
                        "community_endorsements": 5 if community_leader else 0,
                        "network_quality_score": 0.8 if group_member else 0.3
                    }
                }
                
                # Perform assessment
                credit_engine = sigma_components["credit_engine"]
                assessment = credit_engine.assess_credit_application(
                    applicant_data, loan_request, alternative_data
                )
                
                # Display results
                display_credit_assessment_results(assessment)

def display_credit_assessment_results(assessment):
    """Display credit assessment results"""
    
    st.markdown("### 📊 Credit Assessment Results")
    
    # Overall score and recommendation
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        score_color = "green" if assessment.credit_score > 70 else "orange" if assessment.credit_score > 50 else "red"
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: white;">Credit Score</h3>
            <h2 style="color: white;">{assessment.credit_score:.0f}/100</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: white;">Risk Category</h3>
            <h2 style="color: white;">{assessment.risk_category}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        approval_prob = assessment.approval_probability * 100
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: white;">Approval Probability</h3>
            <h2 style="color: white;">{approval_prob:.0f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: white;">Recommended Amount</h3>
            <h2 style="color: white;">${assessment.recommended_amount:,.0f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed breakdown
    col5, col6 = st.columns(2)
    
    with col5:
        st.markdown("#### Risk Factors")
        for risk in assessment.risk_factors:
            st.warning(f"• {risk}")
    
    with col6:
        st.markdown("#### Mitigating Factors")
        for factor in assessment.mitigating_factors:
            st.success(f"• {factor}")
    
    # Loan recommendation details
    st.markdown("#### Loan Recommendation")
    loan_rec = assessment.loan_recommendation
    
    rec_col1, rec_col2, rec_col3 = st.columns(3)
    
    with rec_col1:
        st.metric("Recommended Term", f"{assessment.recommended_term} months")
        st.metric("Interest Rate", f"{assessment.interest_rate_suggestion:.1%}")
    
    with rec_col2:
        st.metric("Monthly Payment", f"${(assessment.recommended_amount * (assessment.interest_rate_suggestion/12)) / (1 - (1 + assessment.interest_rate_suggestion/12)**(-assessment.recommended_term)):,.0f}")
    
    with rec_col3:
        if assessment.alternative_products:
            st.write("**Alternative Products:**")
            for alt in assessment.alternative_products[:2]:
                st.info(f"• {alt.get('name', 'Alternative Product')}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"""
        ## SIGMA Platform Error
        
        There was an error running the SIGMA platform: {str(e)}
        
        **Common solutions:**
        1. Ensure all environment variables are properly set
        2. Check that all dependencies are installed: `pip install -r requirements.txt`
        3. Verify that the SIGMA platform components are properly initialized
        
        **For technical support:** Check the troubleshooting guide or contact support.
        """)
        
        if st.button("Retry"):
            st.rerun()