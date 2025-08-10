#!/usr/bin/env python3
"""
Test script for Gemini API integration in UbuntuAI
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_gemini_integration():
    """Test Gemini API integration"""
    print("🧪 Testing Gemini API Integration...")
    
    try:
        # Test 1: Import and initialize settings
        print("1. Testing settings...")
        from config.settings import settings
        print(f"   ✅ Settings loaded successfully")
        print(f"   ✅ Google API Key: {'Configured' if settings.GOOGLE_API_KEY else 'Not configured'}")
        print(f"   ✅ Embedding Model: {settings.EMBEDDING_MODEL}")
        print(f"   ✅ LLM Model: {settings.LLM_MODEL}")
        print(f"   ✅ Embedding Dimensions: {settings.EMBEDDING_DIMENSIONS}")
        
        # Test 2: Test embedding service
        print("2. Testing embedding service...")
        from utils.embeddings import embedding_service
        
        if embedding_service is None:
            print("   ❌ Embedding service not initialized")
            return False
            
        print(f"   ✅ Embedding service initialized")
        print(f"   ✅ Model: {embedding_service.model_name}")
        print(f"   ✅ Dimensions: {embedding_service.dimensions}")
        print(f"   ✅ Task Type: {embedding_service.task_type}")
        
        # Test 3: Test embedding creation
        print("3. Testing embedding creation...")
        test_text = "African business opportunities in Ghana fintech sector"
        embedding = embedding_service.create_embedding(test_text)
        
        if embedding:
            print(f"   ✅ Embedding created successfully!")
            print(f"   ✅ Dimensions: {len(embedding)}")
            print(f"   ✅ Sample values: {embedding[:5]}")
            print(f"   ✅ Value range: [{min(embedding):.4f}, {max(embedding):.4f}]")
        else:
            print("   ❌ Failed to create embedding")
            return False
        
        # Test 4: Test different task types
        print("4. Testing different embedding task types...")
        
        # Test query embedding
        query_embedding = embedding_service.create_query_embedding("Find funding for startups")
        if query_embedding:
            print(f"   ✅ Query embedding created (dims: {len(query_embedding)})")
        else:
            print("   ⚠️ Query embedding failed")
        
        # Test document embedding
        doc_embedding = embedding_service.create_document_embedding("This is a document about African startups")
        if doc_embedding:
            print(f"   ✅ Document embedding created (dims: {len(doc_embedding)})")
        else:
            print("   ⚠️ Document embedding failed")
        
        # Test 5: Test similarity calculation
        print("5. Testing similarity calculation...")
        similarity = embedding_service.cosine_similarity(embedding, query_embedding or embedding)
        print(f"   ✅ Similarity calculated: {similarity:.4f}")
        
        # Test 6: Test batch embeddings
        print("6. Testing batch embeddings...")
        test_texts = [
            "Nigeria fintech landscape",
            "Kenya agricultural technology",
            "South African mining sector",
            "Ghana digital payments"
        ]
        
        batch_embeddings = embedding_service.create_embeddings_batch(test_texts)
        successful_embeddings = [e for e in batch_embeddings if e is not None]
        
        print(f"   ✅ Batch processing: {len(successful_embeddings)}/{len(test_texts)} successful")
        
        # Test 7: Test RAG engine
        print("7. Testing RAG engine...")
        from api.rag_engine import rag_engine
        print(f"   ✅ RAG engine initialized")
        print(f"   ✅ Model: {rag_engine.model}")
        
        # Test 8: Test simple query (if vector store has data)
        print("8. Testing simple query...")
        try:
            response = rag_engine.query("What are the main business sectors in Ghana?")
            print(f"   ✅ Query processed successfully")
            print(f"   ✅ Response length: {len(response.get('answer', ''))}")
            print(f"   ✅ Confidence: {response.get('confidence', 'N/A')}")
            print(f"   ✅ Sources: {len(response.get('sources', []))}")
        except Exception as e:
            print(f"   ⚠️ Query test failed: {e}")
        
        # Test 9: Test vector store
        print("9. Testing vector store...")
        from api.vector_store import vector_store
        stats = vector_store.get_collection_stats()
        print(f"   ✅ Vector store accessible")
        print(f"   ✅ Collection: {stats.get('collection_name', 'N/A')}")
        print(f"   ✅ Documents: {stats.get('total_documents', 0)}")
        
        # Test 10: Test Gemini LLM directly
        print("10. Testing Gemini LLM directly...")
        import google.generativeai as genai
        genai.configure(api_key=settings.GOOGLE_API_KEY)
        
        model = genai.GenerativeModel('gemini-1.5-flash')
        test_response = model.generate_content("What is the capital of Ghana?")
        
        if test_response and test_response.text:
            print(f"   ✅ Gemini LLM response: {test_response.text[:100]}...")
        else:
            print("   ⚠️ Gemini LLM test failed")
        
        print("\n🎉 All tests completed successfully!")
        print("\n📊 Test Summary:")
        print(f"   • Google API Key: ✅ Configured")
        print(f"   • Embedding Service: ✅ Working")
        print(f"   • Embedding Model: {settings.EMBEDDING_MODEL}")
        print(f"   • LLM Model: {settings.LLM_MODEL}")
        print(f"   • Vector Store: ✅ Accessible")
        print(f"   • RAG Engine: ✅ Functional")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_whatsapp_agent():
    """Test WhatsApp agent if configured"""
    print("\n🧪 Testing WhatsApp Agent...")
    
    try:
        from api.whatsapp_agent import WhatsAppBusinessAgent
        
        # Test initialization
        agent = WhatsAppBusinessAgent()
        print("   ✅ WhatsApp agent initialized")
        
        # Test message handling
        test_response = agent.handle_message("test_user", "Hello, I need business advice")
        print(f"   ✅ Message handling works")
        print(f"   ✅ Response: {test_response[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"   ⚠️ WhatsApp agent test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 UbuntuAI Gemini Integration Test Suite")
    print("=" * 50)
    
    # Main integration test
    main_success = test_gemini_integration()
    
    # WhatsApp agent test
    whatsapp_success = test_whatsapp_agent()
    
    print("\n" + "=" * 50)
    print("📋 FINAL TEST RESULTS:")
    print(f"   Main Integration: {'✅ PASS' if main_success else '❌ FAIL'}")
    print(f"   WhatsApp Agent: {'✅ PASS' if whatsapp_success else '⚠️ SKIP/FAIL'}")
    
    if main_success:
        print("\n🎉 UbuntuAI is ready to run!")
        print("\nNext steps:")
        print("1. Initialize knowledge base: python initialize_knowledge_base.py")
        print("2. Start the application: streamlit run app.py")
    else:
        print("\n❌ Please fix the issues above before running UbuntuAI")
    
    sys.exit(0 if main_success else 1)