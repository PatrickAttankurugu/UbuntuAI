import streamlit as st
import sys
import os
import asyncio
from datetime import datetime

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from api.rag_engine import rag_engine
from api.langchain_agents import create_ghana_business_agent
from api.scoring_engine import create_scoring_engine
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

if "agent_mode" not in st.session_state:
    st.session_state.agent_mode = False

if "scoring_engines" not in st.session_state:
    st.session_state.scoring_engines = create_scoring_engine()

if "ghana_agent" not in st.session_state:
    st.session_state.ghana_agent = create_ghana_business_agent()

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>UbuntuAI</h1>
        <p>Your AI-powered guide to African entrepreneurship and business opportunities</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        
        # Agent mode toggle
        st.subheader("AI Mode")
        agent_mode = st.toggle(
            "Advanced Agent Mode",
            value=st.session_state.agent_mode,
            help="Enable multi-step agent workflows with tools"
        )
        st.session_state.agent_mode = agent_mode
        
        if agent_mode:
            st.info("Agent mode: Uses advanced workflows with multiple tools")
        else:
            st.info("Chat mode: Direct Q&A with knowledge base")
        
        st.divider()
        
        # User context settings
        st.subheader("Your Context")
        user_country = st.selectbox(
            "Your Country/Region of Interest:",
            [""] + settings.AFRICAN_COUNTRIES,
            help="This helps personalize responses to your region"
        )
        
        # Enhanced sector selection
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
        
        # Ghana-specific location if Ghana is selected
        ghana_location = None
        if user_country == "Ghana":
            ghana_location = st.selectbox(
                "Ghana Location:",
                [""] + settings.GHANA_MAJOR_CITIES,
                help="Your specific location in Ghana"
            )
        
        # Update user context
        st.session_state.user_context = {
            "country": user_country if user_country else None,
            "sector": user_sector if user_sector else None,
            "business_stage": business_stage if business_stage else None,
            "ghana_location": ghana_location if ghana_location else None
        }
        
        st.divider()
        
        # Quick actions
        st.subheader("Quick Access")
        
        if st.button("Business Assessment", use_container_width=True):
            st.session_state.current_page = "assessment"
            st.rerun()
        
        if st.button("Browse Funding Database", use_container_width=True):
            st.session_state.current_page = "funding"
            st.rerun()
        
        if st.button("Regulatory Guide", use_container_width=True):
            st.session_state.current_page = "regulatory"
            st.rerun()
        
        if st.button("Agent Workflows", use_container_width=True):
            st.session_state.current_page = "workflows"
            st.rerun()
        
        if st.button("Reset Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.conversation_history = []
            st.rerun()
        
        st.divider()
        
        # System stats
        st.subheader("System Info")
        try:
            from api.vector_store import vector_store
            stats = vector_store.get_collection_stats()
            st.metric("Knowledge Base Size", f"{stats.get('total_documents', 0)} documents")
            
            if agent_mode:
                st.metric("Agent Tools", "5 active")
                st.metric("Scoring Models", "2 available")
        except:
            st.metric("Knowledge Base Size", "Loading...")
    
    # Main content area
    current_page = getattr(st.session_state, 'current_page', 'chat')
    
    if current_page == 'assessment':
        show_assessment_page()
    elif current_page == 'funding':
        show_funding_page()
    elif current_page == 'regulatory':
        show_regulatory_page()
    elif current_page == 'workflows':
        show_workflows_page()
    else:
        show_chat_page()

def show_assessment_page():
    st.header("Business Assessment Tool")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Business Information")
        
        with st.form("business_assessment_form"):
            business_description = st.text_area(
                "Business Description",
                placeholder="Describe your business in 2-3 sentences...",
                height=100
            )
            
            sector = st.selectbox(
                "Business Sector",
                settings.BUSINESS_SECTORS,
                help="Select your primary business sector"
            )
            
            team_size = st.slider(
                "Team Size",
                min_value=1,
                max_value=20,
                value=1,
                help="Total number of team members including yourself"
            )
            
            stage = st.selectbox(
                "Business Stage",
                ["Idea", "Early", "Growing", "Established"],
                help="Current stage of your business"
            )
            
            location = st.selectbox(
                "Business Location",
                settings.GHANA_MAJOR_CITIES if st.session_state.user_context.get('country') == 'Ghana' else settings.AFRICAN_COUNTRIES,
                help="Where is your business based?"
            )
            
            # Additional assessment fields
            st.subheader("Business Details")
            
            col1_inner, col2_inner = st.columns(2)
            
            with col1_inner:
                generating_revenue = st.checkbox("Currently generating revenue")
                has_customers = st.checkbox("Have paying customers")
                mobile_first = st.checkbox("Mobile-first approach")
                
            with col2_inner:
                local_team = st.checkbox("Local team members")
                market_research = st.checkbox("Conducted market research")
                has_funding = st.checkbox("Previously received funding")
            
            submitted = st.form_submit_button("Assess My Business", use_container_width=True)
    
    with col2:
        st.subheader("Assessment Results")
        
        if submitted and business_description and sector:
            with st.spinner("Analyzing your business..."):
                # Prepare data for scoring
                assessment_data = {
                    'business_description': business_description,
                    'sector': sector.lower(),
                    'team_size': team_size,
                    'product_stage': map_stage_to_product_stage(stage),
                    'generating_revenue': generating_revenue,
                    'customer_count': 100 if has_customers else 0,
                    'mobile_first': mobile_first,
                    'local_team_members': local_team,
                    'market_research_done': market_research,
                    'funding_status': 'pre_seed' if has_funding else 'none',
                    'local_market_knowledge': True,  # Assume true for local businesses
                }
                
                # Get assessment from scoring engine
                scorer = st.session_state.scoring_engines['startup_scorer']
                result = scorer.score_startup(assessment_data)
                
                # Display results
                display_assessment_results(result)
        
        elif submitted:
            st.warning("Please fill in the business description and sector to get an assessment.")
        else:
            st.info("Fill out the form on the left to get a comprehensive business assessment.")
    
    if st.button("Back to Chat", use_container_width=True):
        st.session_state.current_page = "chat"
        st.rerun()

def display_assessment_results(result):
    """Display comprehensive assessment results"""
    
    # Overall score with color coding
    score_color = "HIGH" if result.overall_score > 0.7 else "MEDIUM" if result.overall_score > 0.5 else "LOW"
    
    st.metric(
        f"{score_color} Overall Readiness Score",
        f"{result.overall_score:.2f}/1.0",
        delta=f"Confidence: {result.confidence:.2f}"
    )
    
    # Component scores
    st.subheader("Detailed Breakdown")
    
    for component, score in result.component_scores.items():
        component_name = component.replace('_', ' ').title()
        status = "HIGH" if score > 0.7 else "MEDIUM" if score > 0.5 else "LOW"
        
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            st.write(f"**{component_name}** ({status})")
        with col2:
            st.write(f"{score:.2f}")
        with col3:
            st.progress(score)
    
    # Risk factors
    if result.risk_factors:
        st.subheader("Risk Factors")
        for risk in result.risk_factors:
            st.warning(f"• {risk}")
    
    # Recommendations
    if result.recommendations:
        st.subheader("Recommendations")
        for i, rec in enumerate(result.recommendations, 1):
            st.success(f"{i}. {rec}")
    
    # Ghana-specific insights if applicable
    st.subheader("Ghana Market Insights")
    st.info("""
    **Key Considerations for Ghana:**
    • Focus on mobile-first solutions (95%+ mobile penetration)
    • Consider Mobile Money integration for payments
    • Build trust through local partnerships
    • Adapt to seasonal business cycles
    • Leverage strong community networks
    """)

def show_workflows_page():
    st.header("Agent Workflows")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Available Workflows")
        
        workflows = st.session_state.ghana_agent.get_available_workflows()
        
        selected_workflow = st.selectbox(
            "Choose a workflow:",
            options=[w['name'] for w in workflows],
            help="Select an agent workflow to run"
        )
        
        # Show workflow details
        selected_details = next(w for w in workflows if w['name'] == selected_workflow)
        
        st.info(f"**Description:** {selected_details['description']}")
        st.caption(f"**Triggers:** {selected_details['trigger']}")
        
        # Workflow input
        workflow_query = st.text_area(
            "Workflow Input",
            placeholder="Describe what you need help with...",
            help="Provide context for the agent workflow"
        )
        
        run_workflow = st.button("Run Workflow", use_container_width=True)
        
        st.divider()
        
        # Quick workflow buttons
        st.subheader("Quick Actions")
        
        if st.button("Assess Fintech Startup", use_container_width=True):
            st.session_state.workflow_query = "I have a fintech startup in Accra with mobile payments. Please assess my business readiness and provide recommendations."
            st.session_state.auto_run = True
            st.rerun()
        
        if st.button("Find Agritech Funding", use_container_width=True):
            st.session_state.workflow_query = "Find funding opportunities for my agritech startup in Ghana that helps farmers with crop monitoring."
            st.session_state.auto_run = True
            st.rerun()
        
        if st.button("Ghana Business Registration", use_container_width=True):
            st.session_state.workflow_query = "Help me understand the complete process for registering a technology company in Ghana including taxes."
            st.session_state.auto_run = True
            st.rerun()
    
    with col2:
        st.subheader("Workflow Results")
        
        # Check for auto-run workflows
        auto_run = getattr(st.session_state, 'auto_run', False)
        auto_query = getattr(st.session_state, 'workflow_query', '')
        
        if auto_run:
            workflow_query = auto_query
            run_workflow = True
            st.session_state.auto_run = False
            st.session_state.workflow_query = ''
        
        if run_workflow and workflow_query:
            with st.spinner("Running agent workflow..."):
                try:
                    # Add user context to the query
                    user_context = st.session_state.user_context
                    
                    # Run the agent workflow
                    result = st.session_state.ghana_agent.process_query_sync(
                        workflow_query, 
                        user_context
                    )
                    
                    if result['success']:
                        # Display the response
                        st.markdown("### Agent Response")
                        st.markdown(result['answer'])
                        
                        # Show tools used
                        if result['tools_used']:
                            st.markdown("### Tools Used")
                            for tool in result['tools_used']:
                                st.badge(tool.replace('_', ' ').title())
                        
                        # Show actions taken
                        if result['actions_taken']:
                            with st.expander("Detailed Actions"):
                                for i, action in enumerate(result['actions_taken'], 1):
                                    st.markdown(f"**{i}. {action['tool']}**")
                                    st.code(str(action['input']), language="json")
                                    st.caption(f"Timestamp: {action['timestamp']}")
                    else:
                        st.error(f"Workflow failed: {result['answer']}")
                        
                except Exception as e:
                    st.error(f"Error running workflow: {str(e)}")
        
        elif run_workflow:
            st.warning("Please provide input for the workflow.")
        
        else:
            st.info("Select a workflow and provide input to see results here.")
            
            # Show example workflows
            st.markdown("### Example Workflows")
            
            examples = [
                {
                    "title": "Business Assessment",
                    "query": "Assess my edtech startup in Kumasi with 3 team members focusing on rural education"
                },
                {
                    "title": "Funding Search", 
                    "query": "Find Series A funding for my healthtech startup serving rural communities"
                },
                {
                    "title": "Market Research",
                    "query": "Research the e-commerce market in Northern Ghana for agricultural products"
                },
                {
                    "title": "Impact Measurement",
                    "query": "Create impact tracking framework for my social enterprise helping women farmers"
                }
            ]
            
            for example in examples:
                with st.expander(f"{example['title']}"):
                    st.code(example['query'])
    
    if st.button("Back to Chat", use_container_width=True):
        st.session_state.current_page = "chat"
        st.rerun()

def show_chat_page():
    st.header("Ask UbuntuAI Anything")
    
    # Mode indicator
    mode_color = "ACTIVE" if st.session_state.agent_mode else "STANDARD"
    mode_text = "Advanced Agent Mode" if st.session_state.agent_mode else "Chat Mode"
    st.info(f"**{mode_text}** ({mode_color}) - {get_mode_description()}")
    
    # Quick start suggestions
    if not st.session_state.messages:
        st.subheader("Popular Questions")
        
        # Enhanced suggestions based on mode
        if st.session_state.agent_mode:
            suggestions = [
                "Assess my fintech startup readiness in Ghana",
                "Find funding for my agritech business serving farmers",
                "Create impact measurement framework for my social enterprise",
                "Research the healthtech market in West Africa",
                "Help me register a tech company in Ghana with tax optimization",
                "Design a business model for rural e-commerce in Ghana"
            ]
        else:
            suggestions = prompt_templates.get_conversation_starter_prompts()
        
        cols = st.columns(2)
        
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
                
                # Enhanced assistant message display
                if message.get("agent_mode"):
                    # Agent mode specific display
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if message.get("tools_used"):
                            st.metric("Tools Used", len(message["tools_used"]))
                    
                    with col2:
                        if message.get("actions_taken"):
                            st.metric("Actions", message["actions_taken"])
                    
                    with col3:
                        st.metric("Mode", "Agent")
                    
                    # Show tools used
                    if message.get("tools_used"):
                        with st.expander("Agent Tools"):
                            for tool in message["tools_used"]:
                                st.badge(tool.replace('_', ' ').title())
                
                else:
                    # Regular mode display
                    # Show sources if available
                    if "sources" in message and message["sources"]:
                        with st.expander(f"Sources ({len(message['sources'])})"):
                            for i, source in enumerate(message["sources"]):
                                st.markdown(f"""
                                <div class="source-box">
                                    <strong>Source {i+1}</strong> (Similarity: {source.get('similarity', 'N/A')})<br>
                                    {source['content_preview']}
                                    <br><small>{source.get('metadata', {})}</small>
                                </div>
                                """, unsafe_allow_html=True)
                    
                    # Show confidence and follow-ups
                    if "confidence" in message:
                        confidence = message["confidence"]
                        if confidence < 0.6:
                            st.warning(f"Confidence: {confidence:.2f} - Consider enabling Agent Mode for better results")
                    
                    # Show follow-up questions
                    if "follow_ups" in message and message["follow_ups"]:
                        st.markdown("**You might also ask:**")
                        for j, follow_up in enumerate(message["follow_ups"]):
                            if st.button(follow_up, key=f"followup_{len(st.session_state.messages)}_{j}"):
                                process_user_input(follow_up)
            else:
                st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input(settings.CHAT_PLACEHOLDER):
        process_user_input(prompt)

def get_mode_description():
    """Get description of current mode"""
    if st.session_state.agent_mode:
        return "Uses multi-step workflows with business assessment, funding search, and regulatory tools"
    else:
        return "Direct Q&A with African business knowledge base"

def process_user_input(user_input: str):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.conversation_history.append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Generate response based on mode
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                if st.session_state.agent_mode:
                    # Use advanced agent workflow
                    agent_result = st.session_state.ghana_agent.process_query_sync(
                        user_input,
                        user_context=st.session_state.user_context
                    )
                    
                    if agent_result['success']:
                        answer = agent_result['answer']
                        
                        # Display answer
                        st.markdown(answer)
                        
                        # Show agent insights
                        if agent_result['tools_used']:
                            with st.expander(f"Agent Tools Used ({len(agent_result['tools_used'])})"):
                                for tool in agent_result['tools_used']:
                                    st.badge(tool.replace('_', ' ').title())
                        
                        # Store agent response
                        assistant_message = {
                            "role": "assistant",
                            "content": answer,
                            "agent_mode": True,
                            "tools_used": agent_result['tools_used'],
                            "actions_taken": len(agent_result['actions_taken'])
                        }
                        
                    else:
                        # Fallback to regular RAG if agent fails
                        st.warning("Agent workflow failed, falling back to regular mode...")
                        answer = agent_result['answer']
                        st.markdown(answer)
                        
                        assistant_message = {
                            "role": "assistant",
                            "content": answer,
                            "agent_mode": False,
                            "error": True
                        }
                
                else:
                    # Use regular RAG engine
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
                        st.warning(f"Low confidence response ({confidence:.2f}). Consider using Agent Mode for better results.")
                    
                    # Store regular response
                    assistant_message = {
                        "role": "assistant", 
                        "content": answer,
                        "sources": sources,
                        "follow_ups": follow_ups,
                        "confidence": confidence,
                        "agent_mode": False
                    }
                
                st.session_state.messages.append(assistant_message)
                st.session_state.conversation_history.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                error_msg = f"I apologize, but I encountered an error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    st.rerun()

def show_funding_page():
    st.header("African Funding Database")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Search Filters")
        
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
        
        if st.button("Search", use_container_width=True):
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
        st.subheader("Results")
        
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
    
    if st.button("Back to Chat", use_container_width=True):
        st.session_state.current_page = "chat"
        st.rerun()

def show_regulatory_page():
    st.header("Business Regulatory Guide")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Select Country")
        
        available_countries = list(regulatory_db.business_registration.keys())
        selected_country = st.selectbox(
            "Country:",
            available_countries
        )
        
        if st.button("Generate Guide", use_container_width=True):
            st.session_state.selected_country = selected_country
    
    with col2:
        if hasattr(st.session_state, 'selected_country'):
            country = st.session_state.selected_country
            guide = regulatory_db.generate_country_guide(country)
            
            st.markdown(guide)
            
            # Additional resources
            st.subheader("Additional Resources")
            
            reg_info = regulatory_db.get_business_registration_info(country)
            if reg_info.get('online_portal'):
                st.write(f"**Official Portal:** [{reg_info['online_portal']}]({reg_info['online_portal']})")
        else:
            st.info("Select a country from the left panel to view the business registration guide.")
            
            # Show overview
            st.subheader("Available Countries")
            
            countries = list(regulatory_db.business_registration.keys())
            cols = st.columns(2)
            
            for i, country in enumerate(countries):
                col = cols[i % 2]
                with col:
                    if st.button(f"{country}", key=f"country_{i}", use_container_width=True):
                        st.session_state.selected_country = country
                        st.rerun()
    
    if st.button("Back to Chat", use_container_width=True):
        st.session_state.current_page = "chat"
        st.rerun()

def map_stage_to_product_stage(stage: str) -> str:
    """Map assessment stage to scoring stage"""
    stage_mapping = {
        'Idea': 'idea',
        'Early': 'prototype',
        'Growing': 'beta',
        'Established': 'launched'
    }
    return stage_mapping.get(stage, 'idea')

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"""
        ## Application Error
        
        There was an error running UbuntuAI: {str(e)}
        
        **Common solutions:**
        1. Make sure your OpenAI API key is set in the `.env` file
        2. Install all required dependencies: `pip install -r requirements.txt`
        3. Check that all files are in the correct directory structure
        
        **For support:** Please check the documentation or contact support.
        """)
        
        if st.button("Retry"):
            st.rerun()