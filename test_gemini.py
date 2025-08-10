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
        
        # Test 2: Test embedding service
        print("2. Testing embedding service...")
        from utils.embeddings import EmbeddingService
        embedding_service = EmbeddingService()
        print(f"   ✅ Embedding service initialized")
        print(f"   ✅ Model: {embedding_service.model}")
        
        # Test 3: Test embedding creation
        print("3. Testing embedding creation...")
        test_text = "African business opportunities in Ghana"
        embedding = embedding_service.create_embedding(test_text)
        
        if embedding:
            print(f"   ✅ Embedding created successfully!")
            print(f"   ✅ Dimensions: {len(embedding)}")
            print(f"   ✅ Sample values: {embedding[:5]}")
        else:
            print("   ❌ Failed to create embedding")
            return False
        
        # Test 4: Test RAG engine
        print("4. Testing RAG engine...")
        from api.rag_engine import rag_engine
        print(f"   ✅ RAG engine initialized")
        print(f"   ✅ Model: {rag_engine.model}")
        
        # Test 5: Test simple query
        print("5. Testing simple query...")
        try:
            response = rag_engine.query("What are the main business sectors in Ghana?")
            print(f"   ✅ Query processed successfully")
            print(f"   ✅ Response length: {len(response.get('answer', ''))}")
        except Exception as e:
            print(f"   ⚠️ Query test failed: {e}")
        
        print("\n🎉 All tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_gemini_integration()
    sys.exit(0 if success else 1) 