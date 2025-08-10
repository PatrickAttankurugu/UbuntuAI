#!/usr/bin/env python3
"""
Initialize the UbuntuAI Enhanced Knowledge Base

This script sets up the vector database with initial data from:
- Internal funding database
- Regulatory information
- Enhanced sample documents with scoring examples
- Agent workflow testing
- Ghana-specific business contexts

Run this script once after setting up the environment.
"""

import sys
import os
from typing import List, Dict, Any

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def initialize_knowledge_base():
    try:
        print("Initializing UbuntuAI Enhanced Knowledge Base...")
        
        # Import modules
        from api.vector_store import vector_store
        from data.processor import data_processor
        from config.settings import settings
        
        print("Processing internal data sources...")
        
        # Process funding data
        print("  Processing funding opportunities...")
        funding_chunks = data_processor.process_funding_data()
        print(f"     Generated {len(funding_chunks)} funding chunks")
        
        # Process regulatory data
        print("  Processing regulatory information...")
        regulatory_chunks = data_processor.process_regulatory_data()
        print(f"     Generated {len(regulatory_chunks)} regulatory chunks")
        
        # Generate enhanced sample documents (including scoring examples)
        print("  Generating enhanced sample documents...")
        sample_chunks = create_enhanced_sample_documents()
        print(f"     Generated {len(sample_chunks)} sample chunks")
        
        # Combine all chunks
        all_chunks = funding_chunks + regulatory_chunks + sample_chunks
        print(f"Total chunks to process: {len(all_chunks)}")
        
        # Prepare for vector store
        print("Preparing documents for vector storage...")
        vector_data = data_processor.prepare_documents_for_vectorstore(all_chunks)
        
        # Add to vector store
        print("Adding documents to vector database...")
        success = vector_store.add_documents(
            documents=vector_data["documents"],
            metadatas=vector_data["metadatas"],
            ids=vector_data["ids"]
        )
        
        if success:
            print("SUCCESS: Knowledge base initialized successfully!")
            
            # Get stats
            stats = vector_store.get_collection_stats()
            print(f"Database Statistics:")
            print(f"   - Total documents: {stats.get('total_documents', 0)}")
            print(f"   - Collection name: {stats.get('collection_name', 'N/A')}")
            print(f"   - Persist directory: {stats.get('persist_directory', 'N/A')}")
            
        else:
            print("ERROR: Failed to initialize knowledge base")
            return False
        
        # Test all systems
        print("\n" + "="*60)
        print("TESTING ENHANCED SYSTEMS")
        print("="*60)
        
        test_rag_system()
        test_ghana_specific_features()
        
        return True
        
    except ImportError as e:
        print(f"ERROR - Import error: {e}")
        print("Please ensure all dependencies are installed:")
        print("pip install -r requirements.txt")
        return False
        
    except Exception as e:
        print(f"ERROR - Error during initialization: {e}")
        return False

def test_rag_system():
    try:
        from api.rag_engine import rag_engine
        from api.langchain_agents import create_ghana_business_agent
        from api.scoring_engine import create_scoring_engine
        
        # Test queries
        test_queries = [
            "What funding opportunities are available for fintech startups in Nigeria?",
            "How do I register a business in Kenya?",
            "Tell me about successful African startups",
        ]
        
        print("\nRunning RAG system tests...")
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{i}. Testing RAG: {query[:50]}...")
            
            try:
                response = rag_engine.query(query)
                
                if response and response.get("answer"):
                    print(f"   SUCCESS: Response generated ({len(response['answer'])} chars)")
                    print(f"   Sources found: {len(response.get('sources', []))}")
                    print(f"   Confidence: {response.get('confidence', 0.0):.2f}")
                else:
                    print("   WARNING: No response generated")
                    
            except Exception as e:
                print(f"   ERROR: Query failed: {e}")
        
        # Test scoring engines
        print("\nTesting scoring engines...")
        try:
            scoring_engines = create_scoring_engine()
            
            # Test startup scorer
            test_business_data = {
                'business_description': 'A mobile payment app for rural farmers in Ghana',
                'sector': 'fintech',
                'team_size': 3,
                'product_stage': 'prototype',
                'generating_revenue': False,
                'customer_count': 50,
                'mobile_first': True,
                'local_team_members': True,
                'local_market_knowledge': True
            }
            
            startup_result = scoring_engines['startup_scorer'].score_startup(test_business_data)
            print(f"   Startup Scorer: {startup_result.overall_score:.2f} score")
            print(f"   Risk factors: {len(startup_result.risk_factors)}")
            print(f"   Recommendations: {len(startup_result.recommendations)}")
            
            # Test loan scorer
            test_loan_data = {
                'income_type': 'business_regular',
                'business_age_months': 18,
                'location_type': 'rural',
                'gender': 'female',
                'age': 35,
                'mobile_money_history': True,
                'established_customer_base': True
            }
            
            loan_result = scoring_engines['loan_scorer'].score_loan_risk(test_loan_data)
            print(f"   Loan Scorer: {loan_result.overall_score:.2f} score")
            print(f"   Risk factors: {len(loan_result.risk_factors)}")
            
        except Exception as e:
            print(f"   ERROR: Scoring test failed: {e}")
        
        # Test agent workflows (if enabled)
        if settings.USE_LANGCHAIN_AGENTS:
            print("\nTesting agent workflows...")
            try:
                agent = create_ghana_business_agent()
                
                test_agent_query = "Assess my fintech startup in Accra with 3 team members"
                agent_result = agent.process_query_sync(
                    test_agent_query, 
                    {'country': 'Ghana', 'sector': 'fintech'}
                )
                
                if agent_result['success']:
                    print(f"   Agent SUCCESS: Response generated")
                    print(f"   Tools used: {', '.join(agent_result['tools_used'])}")
                    print(f"   Actions taken: {len(agent_result['actions_taken'])}")
                else:
                    print(f"   Agent WARNING: {agent_result['answer'][:100]}...")
                    
            except Exception as e:
                print(f"   ERROR: Agent test failed: {e}")
        else:
            print("   SKIPPED: Agent workflows disabled")
        
        print("\nRAG system test completed!")
        
    except Exception as e:
        print(f"ERROR: RAG test failed: {e}")

def test_ghana_specific_features():
    """Test Ghana-specific enhancements"""
    print("\nTesting Ghana-specific features...")
    
    try:
        # Test Ghana context
        ghana_context = settings.get_ghana_context()
        print(f"   Ghana regions loaded: {len(ghana_context['regions'])}")
        print(f"   Major cities: {len(ghana_context['major_cities'])}")
        print(f"   Local sectors: {len(ghana_context['local_sectors'])}")
        
        # Test Ghana-specific scoring
        scoring_engines = create_scoring_engine()
        
        ghana_business = {
            'business_description': 'Cassava processing plant in Kumasi',
            'sector': 'cassava processing',
            'team_size': 5,
            'product_stage': 'launched',
            'generating_revenue': True,
            'customer_count': 200,
            'mobile_first': True,
            'local_team_members': True,
            'local_market_knowledge': True,
            'serves_rural_market': True,
            'women_led': True
        }
        
        result = scoring_engines['startup_scorer'].score_startup(ghana_business)
        print(f"   Ghana business score: {result.overall_score:.2f}")
        print(f"   SUCCESS: Ghana-specific features working")
        
    except Exception as e:
        print(f"   ERROR: Ghana features test failed: {e}")

def create_enhanced_sample_documents():
    """Create enhanced sample documents with scoring examples"""
    print("Generating enhanced sample documents...")
    
    from data.processor import data_processor
    
    # Get original samples
    original_samples = data_processor.generate_sample_documents()
    
    # Add scoring and agent workflow examples
    enhanced_samples = []
    
    # Business assessment examples
    assessment_examples = [
        {
            "content": """Business Assessment Example: TechGhana Solutions
            
            Company: TechGhana Solutions (Fintech Startup)
            Location: Accra, Ghana
            Team Size: 4 members
            Stage: Early growth (6 months operational)
            
            Business Model: Mobile money integration platform for small businesses
            Target Market: SMEs in Greater Accra region
            Revenue Model: Transaction fees (2.5% per transaction)
            Current Revenue: GHS 15,000/month (growing 20% monthly)
            Customer Base: 150 active merchants
            
            Assessment Results:
            - Overall Readiness Score: 0.72/1.0
            - Business Model Strength: 0.8/1.0
            - Market Readiness: 0.75/1.0  
            - Team Strength: 0.7/1.0
            - Financial Health: 0.65/1.0
            - Traction: 0.7/1.0
            
            Key Strengths:
            • Strong local market knowledge
            • Mobile-first approach aligned with Ghana market
            • Experienced technical team
            • Clear revenue model with proven traction
            
            Risk Factors:
            • Limited funding runway (6 months)
            • Dependency on mobile money providers
            • Competition from established players
            
            Recommendations:
            1. Secure Series A funding within 3 months
            2. Diversify payment integration options
            3. Expand to 2 additional regions
            4. Build stronger partnerships with banks""",
            "metadata": {
                "source": "Business Assessment Database",
                "type": "assessment_example",
                "country": "Ghana",
                "sector": "Fintech",
                "score": 0.72,
                "stage": "early_growth"
            }
        },
        {
            "content": """Loan Risk Assessment: Adwoa's Agribusiness
            
            Applicant: Adwoa Mensah
            Business: Organic Vegetable Farming & Distribution
            Location: Brong-Ahafo Region, Ghana
            Loan Amount Requested: GHS 50,000
            Purpose: Expand greenhouse operations and cold storage
            
            Risk Assessment Results:
            - Overall Risk Score: 0.25 (Low Risk)
            - Credit History Score: 0.6/1.0
            - Income Stability: 0.8/1.0
            - Business Viability: 0.85/1.0
            - Collateral Security: 0.7/1.0
            - Social Capital: 0.9/1.0
            
            Risk Factors:
            • Limited formal credit history
            • Seasonal income fluctuations
            • Dependency on weather conditions
            
            Mitigating Factors:
            • Strong community standing (church leader)
            • Consistent mobile money transaction history
            • Growing customer base in Kumasi markets
            • Property ownership (family land)
            • Previous successful microfinance history
            
            Recommendations:
            - Approve loan with 18-month term
            - Structure repayments around harvest cycles
            - Provide agricultural insurance option
            - Monitor through mobile money transactions""",
            "metadata": {
                "source": "Loan Assessment Database", 
                "type": "loan_assessment",
                "country": "Ghana",
                "sector": "Agriculture",
                "risk_score": 0.25,
                "gender": "female"
            }
        },
        {
            "content": """Agent Workflow Example: Market Research for E-commerce
            
            Query: "Research the e-commerce market in Northern Ghana for agricultural products"
            
            Agent Workflow Results:
            
            Tool 1: Market Research Tool
            - Northern Ghana population: 2.5M people
            - Internet penetration: 45% (growing 15% annually)
            - Mobile money usage: 78% of adults
            - Agricultural production: 60% of regional economy
            
            Tool 2: Competitive Analysis
            - Major players: AgroGhana, FarmConnect, Local cooperatives
            - Market gaps: Cold storage, direct farmer-to-consumer
            - Price transparency issues identified
            
            Tool 3: Business Model Recommendations
            - B2B2C model through cooperatives
            - Mobile-first platform essential
            - SMS notifications for low-smartphone areas
            - Integration with existing transport networks
            
            Tool 4: Funding Assessment
            - Estimated startup costs: GHS 250,000
            - Recommended funding: Seed stage (GHS 100K-500K)
            - Potential funders: TLcom, Savannah Fund, USAID grants
            
            Tool 5: Impact Measurement Framework
            - Primary KPIs: Farmer income increase, market access
            - Social metrics: Women farmers reached, youth employment
            - Environmental: Reduced food waste, sustainable practices
            
            Final Recommendation: Proceed with MVP development focusing on 3 major cooperatives in Tamale region""",
            "metadata": {
                "source": "Agent Workflow Database",
                "type": "workflow_example", 
                "country": "Ghana",
                "sector": "E-commerce",
                "workflow_type": "market_research",
                "tools_used": 5
            }
        },
        {
            "content": """WhatsApp Conversation Example: Business Assessment Flow
            
            User: "I want to start a cassava processing business in Kumasi"
            
            UbuntuAI: "Great! Cassava processing has strong potential in Ghana. I'll help assess your readiness through 7 quick questions.
            
            1/7: How many people are in your team including yourself?"
            
            User: "Just me for now, but my brother wants to join"
            
            UbuntuAI: "2/7: What's your target market? Who will buy your processed cassava?"
            
            User: "Local restaurants and food vendors in Kumasi"
            
            UbuntuAI: "3/7: Do you have experience in food processing or business?"
            
            User: "I worked at a flour mill for 3 years"
            
            UbuntuAI: "4/7: How much startup capital do you have access to?"
            
            User: "About GHS 25,000 saved plus family support"
            
            UbuntuAI: "5/7: Have you researched the local market and competition?"
            
            User: "Yes, I know the major suppliers and their prices"
            
            UbuntuAI: "6/7: Do you have a location identified for processing?"
            
            User: "My family has land in Ejisu, 30 minutes from Kumasi"
            
            UbuntuAI: "7/7: Are you planning to start small or at commercial scale?"
            
            User: "Start small, grow gradually"
            
            UbuntuAI: "Assessment complete! 
            
            Your Readiness Score: 7.2/10
            
            Strengths:
            • Relevant industry experience
            • Local market knowledge 
            • Access to land and capital
            • Realistic growth approach
            
            Next Steps:
            1. Get food processing license (FDA Ghana)
            2. Start with 2-3 restaurant customers
            3. Consider NBSSI business training
            4. Look into cassava cooperatives for supply
            
            Want help with any of these steps?"
            
            Optimizations:
            - Mobile-friendly format (under 1600 characters per message)
            - Local context (FDA Ghana, NBSSI, Ejisu location)
            - Practical next steps
            - Culturally appropriate (family land, gradual growth)""",
            "metadata": {
                "source": "WhatsApp Conversation Database",
                "type": "conversation_example",
                "country": "Ghana", 
                "sector": "Cassava Processing",
                "conversation_type": "business_assessment",
                "optimization": "mobile_first"
            }
        }
    ]
    
    # Process enhanced samples
    for sample in assessment_examples:
        chunks = data_processor.chunker.chunk_document(
            text=sample["content"],
            metadata=sample["metadata"]
        )
        enhanced_samples.extend(chunks)
    
    print(f"   Generated {len(enhanced_samples)} enhanced sample chunks")
    
    return original_samples + enhanced_samples

def check_environment():
    """Check if the environment is properly set up"""
    print("Checking environment setup...")
    
    required_env_vars = ["OPENAI_API_KEY"]
    missing_vars = []
    
    for var in required_env_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"ERROR: Missing required environment variables: {', '.join(missing_vars)}")
        print("Please create a .env file with the required variables.")
        print("See .env.example for reference.")
        return False
    
    # Check optional environment variables
    optional_env_vars = [
        "TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_WHATSAPP_NUMBER"
    ]
    missing_optional = []
    
    for var in optional_env_vars:
        if not os.getenv(var):
            missing_optional.append(var)
    
    if missing_optional:
        print(f"WARNING: Optional environment variables missing: {', '.join(missing_optional)}")
        print("WhatsApp integration will be disabled. Add these for full functionality.")
    
    # Check if directories exist
    required_dirs = ["data", "api", "config", "utils", "knowledge_base"]
    missing_dirs = []
    
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            missing_dirs.append(dir_name)
    
    if missing_dirs:
        print(f"ERROR: Missing directories: {', '.join(missing_dirs)}")
        return False
    
    # Create vector_db directory if it doesn't exist
    if not os.path.exists("vector_db"):
        os.makedirs("vector_db")
        print("Created vector_db directory")
    
    print("SUCCESS: Environment setup looks good!")
    return True

def check_dependencies():
    """Check if required packages are installed"""
    print("Checking Python dependencies...")
    
    required_packages = [
        "streamlit", "langchain", "openai", "chromadb", 
        "tiktoken", "pandas", "numpy", "python-dotenv"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"ERROR: Missing required packages: {', '.join(missing_packages)}")
        print("Please install them with: pip install -r requirements.txt")
        return False
    
    # Check optional packages
    optional_packages = ["twilio", "flask", "scikit-learn", "transformers"]
    missing_optional_packages = []
    
    for package in optional_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing_optional_packages.append(package)
    
    if missing_optional_packages:
        print(f"WARNING: Optional packages missing: {', '.join(missing_optional_packages)}")
        print("Some advanced features may be disabled.")
    
    print("SUCCESS: Core dependencies are installed!")
    return True

def main():
    print("=" * 60)
    print("UbuntuAI Enhanced Knowledge Base Initialization")
    print("=" * 60)
    
    # Check dependencies first
    if not check_dependencies():
        print("\nERROR: Dependency check failed. Please install required packages.")
        sys.exit(1)
    
    # Check environment
    if not check_environment():
        print("\nERROR: Environment check failed. Please fix the issues above.")
        sys.exit(1)
    
    # Initialize knowledge base
    success = initialize_knowledge_base()
    
    if success:
        print("\n" + "=" * 60)
        print("SUCCESS: UbuntuAI Enhanced System is ready!")
        print("")
        print("Available features:")
        print("✓ RAG Knowledge Base with African business context")
        print("✓ Business Assessment Scoring Engine")
        print("✓ LangChain Agent Workflows")
        print("✓ Ghana Market Specialization")
        print("✓ Funding Database with 500+ opportunities")
        print("✓ Regulatory Compliance Guidelines")
        print("")
        print("Next steps:")
        print("1. Launch main app: streamlit run app.py")
        print("2. For WhatsApp integration: python whatsapp_webhook.py")
        print("3. Test agent workflows in the UI")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("ERROR: Initialization failed. Please check the errors above.")
        print("=" * 60)
        sys.exit(1)

if __name__ == "__main__":
    main()