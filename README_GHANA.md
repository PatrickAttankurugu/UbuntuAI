# 🇬🇭 UbuntuAI - Ghanaian Startup Ecosystem RAG System

A modern, modular RAG (Retrieval-Augmented Generation) architecture built specifically for the Ghanaian startup ecosystem, focusing exclusively on **fintech**, **agritech**, and **healthtech** sectors.

## 🎯 Mission

Empower Ghanaian entrepreneurs with AI-powered insights and knowledge about:
- **Fintech**: Mobile money, digital payments, banking regulations, financial inclusion
- **Agritech**: Agricultural technology, farming innovations, food security, rural development
- **Healthtech**: Digital health solutions, medical technology, healthcare accessibility

## 🏗️ Architecture Overview

### Core Components

```
UbuntuAI/
├── 🇬🇭 Ghanaian Focus
│   ├── 16 regions of Ghana
│   ├── 3 target sectors (fintech, agritech, healthtech)
│   └── Ghana-specific business context
├── 🤖 Modular LLM Support
│   ├── OpenAI (GPT models)
│   ├── Anthropic (Claude models)
│   ├── Google (Gemini models)
│   └── Ollama (local models)
├── 📚 Dynamic Document Processing
│   ├── Automatic loading from data/documents/
│   ├── No hardcoded content
│   └── Support for PDF, TXT, CSV, JSON, MD, HTML
├── 🔍 Advanced RAG Engine
│   ├── Self-reflective responses
│   ├── Corrective RAG (CRAG)
│   ├── Context-aware retrieval
│   └── Ghanaian relevance filtering
└── 🎨 Modern UI
    ├── Ghanaian-themed interface
    ├── Sector-specific badges
    ├── Regional focus options
    └── Real-time provider switching
```

### Key Features

- **🇬🇭 Ghana-Exclusive Focus**: No other countries, only Ghanaian startup ecosystem
- **🔄 Easy LLM Provider Switching**: Switch between providers without code changes
- **📚 Dynamic Document Loading**: Automatically processes documents from data/documents/
- **🎯 Sector-Specific Intelligence**: Specialized knowledge for fintech, agritech, healthtech
- **📍 Regional Context**: 16 Ghanaian regions with specific business insights
- **🤖 Self-Reflective AI**: AI evaluates and improves its own responses
- **📊 Confidence Scoring**: Transparent confidence levels for all responses
- **🔍 Source Attribution**: Clear citations and source information

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.8+
- pip or conda
- At least one LLM API key (OpenAI, Anthropic, Google, or Ollama)

### 2. Installation

```bash
# Clone the repository
git clone <repository-url>
cd UbuntuAI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp env_example.txt .env
```

### 3. Configuration

Edit `.env` file with your API keys:

```env
# Primary LLM Provider (choose one)
PRIMARY_LLM_PROVIDER=openai
# or: anthropic, google, ollama

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-3.5-turbo

# Anthropic Configuration
ANTHROPIC_API_KEY=your_anthropic_api_key
ANTHROPIC_MODEL=claude-3-haiku-20240307

# Google Configuration
GOOGLE_API_KEY=your_google_api_key
GOOGLE_MODEL=gemini-pro

# Ollama Configuration (local)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama2

# Vector Store Configuration
VECTOR_STORE_TYPE=chroma
CHROMA_PERSIST_DIRECTORY=./vector_db
```

### 4. Add Your Documents

Place your Ghanaian startup ecosystem documents in `data/documents/`:

```
data/documents/
├── fintech_regulations.pdf
├── agritech_opportunities.pdf
├── healthtech_guidelines.pdf
├── ghana_business_guide.pdf
└── startup_resources.pdf
```

### 5. Initialize the System

```bash
# Initialize the Ghanaian RAG system
python initialize_ghana_rag.py
```

### 6. Run the Application

```bash
# Start the Streamlit application
streamlit run app.py
```

## 🎯 Usage Examples

### Fintech Queries
- "What are the regulatory requirements for fintech startups in Ghana?"
- "How can I get funding for my mobile money platform in Accra?"
- "What are the key challenges for fintech in rural Ghana?"

### Agritech Queries
- "What government programs support agritech innovation in Ghana?"
- "How can I implement precision farming technology in Ashanti region?"
- "What are the export opportunities for Ghanaian agritech products?"

### Healthtech Queries
- "What FDA approvals do I need for my health app in Ghana?"
- "How can I partner with Ghana Health Service for my telemedicine platform?"
- "What are the data protection requirements for healthtech in Ghana?"

## 🔧 System Architecture

### 1. Configuration Layer (`config/`)
- **`settings.py`**: Ghanaian business context, LLM configurations, vector store settings
- **`prompts.py`**: Ghana-focused prompt templates for all AI interactions

### 2. Data Processing Layer (`data/`)
- **`processor.py`**: Dynamic document loading and processing from data/documents/
- **Automatic chunking and metadata enhancement**
- **Ghanaian sector relevance detection**

### 3. API Layer (`api/`)
- **`llm_providers.py`**: Multi-provider LLM abstraction with automatic fallback
- **`rag_engine.py`**: Self-reflective RAG engine with Ghanaian focus
- **`vector_store.py`**: Vector database management and retrieval
- **`langchain_agents.py`**: Specialized agents for Ghanaian business operations

### 4. Utilities (`utils/`)
- **`chunking.py`**: Advanced document chunking strategies
- **`embeddings.py`**: Multi-provider embedding support
- **`context_enhancer.py`**: Context enhancement for better retrieval

### 5. User Interface (`app.py`)
- **Ghanaian-themed Streamlit interface**
- **Sector-specific user profiles**
- **Real-time LLM provider switching**
- **Interactive chat with source attribution**

## 🔄 LLM Provider Switching

The system automatically detects available providers and allows easy switching:

```python
# In the UI sidebar, you can switch between:
- OpenAI (GPT models)
- Anthropic (Claude models)  
- Google (Gemini models)
- Ollama (local models)

# The system automatically:
- Detects available API keys
- Provides fallback options
- Maintains conversation context
- Preserves user preferences
```

## 📚 Document Processing

### Supported Formats
- **PDF**: PDFPlumberLoader for text extraction
- **TXT**: Plain text files
- **CSV**: Tabular data
- **JSON**: Structured data
- **MD**: Markdown files
- **HTML**: Web content

### Automatic Processing
```python
# Documents are automatically:
1. Loaded from data/documents/
2. Chunked using advanced strategies
3. Enhanced with Ghanaian context
4. Indexed in the vector store
5. Made available for RAG queries
```

## 🎨 Customization

### Adding New Sectors
```python
# In config/settings.py
self.GHANA_STARTUP_SECTORS = [
    "Fintech", "Agritech", "Healthtech", "YourNewSector"
]
```

### Adding New Regions
```python
# In config/settings.py
self.GHANA_REGIONS = [
    "Greater Accra", "Ashanti", "YourNewRegion"
]
```

### Custom Prompts
```python
# In config/prompts.py
class PromptTemplates:
    YOUR_CUSTOM_PROMPT = """Your custom prompt template"""
```

## 🔍 Advanced Features

### Self-Reflection
The AI evaluates its own responses and improves them:
- **Accuracy scoring** (0-100)
- **Ghanaian relevance assessment**
- **Automatic correction** for low-quality responses
- **Continuous learning** from user interactions

### Context-Aware Retrieval
- **User profile integration** (sector, region, experience)
- **Conversation history** consideration
- **Ghanaian relevance filtering**
- **Sector-specific document prioritization**

### Confidence Scoring
- **Document relevance** assessment
- **Self-reflection** integration
- **Transparent scoring** for users
- **Quality indicators** for responses

## 🚨 Troubleshooting

### Common Issues

1. **No LLM Providers Available**
   - Check your API keys in `.env`
   - Verify API key validity
   - Check internet connectivity

2. **Documents Not Loading**
   - Ensure documents are in `data/documents/`
   - Check file format support
   - Verify file permissions

3. **Vector Store Errors**
   - Check disk space
   - Verify write permissions
   - Restart initialization script

4. **Import Errors**
   - Activate virtual environment
   - Install missing dependencies
   - Check Python version compatibility

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python initialize_ghana_rag.py
```

## 📊 Performance Monitoring

### Built-in Metrics
- **Processing time** for each query
- **Document retrieval** statistics
- **LLM provider** performance
- **User interaction** patterns

### LangFuse Integration
```env
# Enable LangFuse monitoring
USE_LANGFUSE=true
LANGFUSE_SECRET_KEY=your_secret_key
LANGFUSE_PUBLIC_KEY=your_public_key
LANGFUSE_HOST=https://cloud.langfuse.com
```

## 🔒 Security & Privacy

- **No hardcoded** sensitive information
- **Environment variable** configuration
- **Local document** processing
- **User data** privacy protection
- **API key** security best practices

## 🤝 Contributing

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Code formatting
black .
isort .
```

### Contribution Guidelines
1. **Ghanaian Focus**: All contributions must maintain Ghanaian startup ecosystem focus
2. **No Hardcoding**: Avoid hardcoded content, use configuration files
3. **Modular Design**: Maintain the modular architecture
4. **Testing**: Include tests for new features
5. **Documentation**: Update relevant documentation

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Ghanaian Startup Ecosystem** for inspiration and context
- **LangChain** for the powerful RAG framework
- **Streamlit** for the beautiful UI framework
- **Open Source Community** for various dependencies

## 📞 Support

For support and questions:
- **Issues**: Create GitHub issues
- **Documentation**: Check this README and inline code comments
- **Community**: Join our Ghanaian startup ecosystem discussions

---

**🇬🇭 Built with ❤️ for Ghanaian Entrepreneurs**

*Empowering the future of African innovation through AI-powered knowledge and insights.* 