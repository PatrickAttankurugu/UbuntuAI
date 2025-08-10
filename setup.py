#!/usr/bin/env python3
"""
UbuntuAI Setup Script

This script helps set up the UbuntuAI application with proper dependencies
and configuration validation.

Usage:
    python setup.py [--check] [--install] [--test]
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path

def print_header():
    print("🚀 UbuntuAI Setup Script")
    print("=" * 50)
    print("Setting up your African Business Intelligence Platform")
    print()

def check_python_version():
    """Check if Python version is 3.11+"""
    print("📋 Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} detected")
        print("   UbuntuAI requires Python 3.8 or higher")
        print("   Please upgrade Python and try again")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def check_api_key():
    """Check if Google API key is configured"""
    print("🔑 Checking API key configuration...")
    
    env_file = Path(".env")
    
    if not env_file.exists():
        print("⚠️  .env file not found")
        print("   Creating .env file from template...")
        
        # Copy from .env.example if it exists
        example_file = Path(".env.example")
        if example_file.exists():
            import shutil
            shutil.copy(example_file, env_file)
            print("✅ .env file created from template")
            print("   Please edit .env and add your GOOGLE_API_KEY")
        else:
            print("❌ .env.example not found")
            print("   Please create .env file manually with GOOGLE_API_KEY")
        
        return False
    
    # Check if API key is in .env
    with open(env_file, 'r') as f:
        content = f.read()
    
    if "GOOGLE_API_KEY=" in content and "your_google_api_key_here" not in content:
        print("✅ Google API key configured in .env")
        return True
    else:
        print("⚠️  Google API key not properly configured")
        print("   Please edit .env and add your GOOGLE_API_KEY")
        print("   Get your API key from: https://makersuite.google.com/app/apikey")
        return False

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    
    try:
        # Install requirements
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("📦 Checking dependencies...")
    
    required_packages = [
        "streamlit",
        "google-generativeai", 
        "chromadb",
        "pandas",
        "numpy",
        "python-dotenv"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} (missing)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("   Run: python setup.py --install")
        return False
    
    print("✅ All dependencies available")
    return True

def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    
    directories = [
        "vector_db",
        "logs",
        "data/processed"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ {directory}/")
    
    return True

def test_configuration():
    """Test the configuration"""
    print("🧪 Testing configuration...")
    
    try:
        # Add current directory to path
        sys.path.insert(0, os.getcwd())
        
        # Test settings import
        from config.settings import settings
        print("✅ Settings module loaded")
        
        # Test API key
        if settings.GOOGLE_API_KEY:
            print("✅ Google API key configured")
        else:
            print("❌ Google API key not configured")
            return False
        
        # Test embedding service
        from utils.embeddings import embedding_service
        if embedding_service:
            print("✅ Embedding service initialized")
        else:
            print("❌ Embedding service failed to initialize")
            return False
        
        # Test vector store
        from api.vector_store import vector_store
        if vector_store:
            print("✅ Vector store initialized")
        else:
            print("❌ Vector store failed to initialize")
            return False
        
        print("✅ Configuration test passed")
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def initialize_knowledge_base():
    """Initialize the knowledge base"""
    print("🧠 Initializing knowledge base...")
    
    try:
        subprocess.check_call([
            sys.executable, "initialize_knowledge_base.py", "--verbose"
        ])
        print("✅ Knowledge base initialized")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Knowledge base initialization failed: {e}")
        return False

def run_tests():
    """Run the test suite"""
    print("🧪 Running tests...")
    
    try:
        subprocess.check_call([
            sys.executable, "test_gemini.py"
        ])
        print("✅ All tests passed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Tests failed: {e}")
        return False

def print_next_steps():
    """Print next steps for the user"""
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Start the application:")
    print("   streamlit run app.py")
    print()
    print("2. Optional: Test WhatsApp integration:")
    print("   python whatsapp_webhook.py")
    print()
    print("3. Access the application at:")
    print("   http://localhost:8501")
    print()
    print("📚 Documentation: README.md")
    print("🐛 Issues: Check logs/ directory")

def main():
    parser = argparse.ArgumentParser(description="UbuntuAI Setup Script")
    parser.add_argument("--check", action="store_true", help="Check configuration only")
    parser.add_argument("--install", action="store_true", help="Install dependencies")
    parser.add_argument("--test", action="store_true", help="Run tests only")
    parser.add_argument("--full", action="store_true", help="Full setup including knowledge base")
    
    args = parser.parse_args()
    
    print_header()
    
    # Basic checks
    if not check_python_version():
        return 1
    
    success = True
    
    if args.install:
        success &= install_dependencies()
    elif args.check:
        success &= check_dependencies()
        success &= check_api_key()
        success &= test_configuration()
    elif args.test:
        success &= run_tests()
    elif args.full:
        # Full setup
        success &= check_dependencies() or install_dependencies()
        success &= check_api_key()
        success &= create_directories()
        success &= test_configuration()
        success &= initialize_knowledge_base()
        success &= run_tests()
    else:
        # Default: basic setup
        success &= check_dependencies() or install_dependencies()
        success &= check_api_key()
        success &= create_directories()
        success &= test_configuration()
    
    print("\n" + "=" * 50)
    
    if success:
        print("✅ Setup completed successfully!")
        if not args.check and not args.test:
            print_next_steps()
    else:
        print("Setup failed!")
        print("Please check the errors above and try again.")
        print("\nCommon issues:")
        print("- Missing Google API key in .env file")
        print("- Python version too old (need 3.8+)")
        print("- Network issues preventing package installation")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())