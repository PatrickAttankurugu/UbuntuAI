# UbuntuAI - African Business Intelligence Platform

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28.0-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A comprehensive AI-powered business intelligence platform designed specifically for African entrepreneurs and startups. UbuntuAI provides personalized guidance on funding opportunities, regulatory compliance, market insights, and business development strategies across 54 African countries.

## 🌟 Key Features

### 🤖 **AI-Powered Business Intelligence**
- **Advanced RAG Engine**: Retrieval-Augmented Generation for accurate, context-aware responses
- **Business Assessment**: Automated startup readiness scoring and recommendations
- **Intelligent Query Classification**: Automatically routes questions to specialized handlers
- **Multi-modal Support**: Text-based interface with mobile-optimized responses

### 🌍 **African Market Focus**
- **54 Country Coverage**: Comprehensive business information across all African nations
- **Local Context Awareness**: Culturally sensitive and regionally specific advice
- **Sector Specialization**: Deep knowledge in Fintech, Agritech, Healthtech, and more
- **Funding Landscape**: Extensive database of African VCs, accelerators, and funding opportunities

### 📱 **Multi-Channel Access**
- **Web Interface**: Professional Streamlit-based dashboard
- **WhatsApp Integration**: Mobile-first access for low-bandwidth environments
- **API-Ready**: RESTful endpoints for third-party integrations
- **Offline Capabilities**: Local vector database for reduced API dependencies

### 🔧 **Professional Architecture**
- **Modular Design**: Clean separation of concerns with pluggable components
- **Error Handling**: Comprehensive error handling and logging throughout
- **Scalable Vector Store**: ChromaDB for efficient similarity search
- **Configuration Management**: Environment-based configuration with validation

## 🚀 Quick Start

### Prerequisites

- **Python 3.11 or higher**
- **OpenAI API Key** (required for AI functionality)
- **Optional**: Twilio account for WhatsApp integration

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/UbuntuAI.git
cd UbuntuAI

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file from the example:

```bash
# Copy the example configuration
cp env_example.txt .env

# Edit with your API keys
# Minimum required: OPENAI_API_KEY
```

**Required Environment Variables:**
```env
# OpenAI API Key (Required)
OPENAI_API_KEY=your_openai_api_key_here

# Optional: WhatsApp Integration
TWILIO_ACCOUNT_SID=your_twilio_account_sid
TWILIO_AUTH_TOKEN=your_twilio_auth_token
TWILIO_WHATSAPP_NUMBER=whatsapp:+1415xxxxxxx
```

### 3. Initialize Knowledge Base

```bash
# Initialize with sample data
python initialize_knowledge_base.py --verbose

# Force reinitialize if needed
python initialize_knowledge_base.py --force
```

### 4. Run the Application

```bash
# Start the Streamlit web interface
streamlit run app.py

# The app will be available at http://localhost:8501
```

## 📖 Usage Guide

### Web Interface

1. **Profile Setup**: Configure your business profile in the sidebar
2. **Ask Questions**: Use natural language to ask about:
   - Funding opportunities for your sector and country
   - Business registration and regulatory requirements
   - Market research and competitive analysis
   - Success stories and case studies

3. **Example Queries**:
   ```
   "What funding opportunities are available for fintech startups in Nigeria?"
   "How do I register a business in Kenya?"
   "Tell me about successful agritech companies in East Africa"
   "What are the tax incentives for startups in Rwanda?"
   ```

### Business Assessment

Get a comprehensive evaluation of your startup readiness:

```
"Please assess my business readiness and provide recommendations"
```

The system will analyze:
- Team composition and experience
- Market opportunity and fit
- Product development stage
- Business model viability
- Traction and growth indicators

### WhatsApp Integration

If configured, the system supports WhatsApp for mobile access:

1. **Setup Webhook**: Configure Twilio webhook to point to your deployment
2. **Mobile Access**: Users can interact via WhatsApp messages
3. **Optimized Responses**: Automatically formatted for mobile consumption

## 🏗️ Architecture

### Core Components

```
UbuntuAI/
├── api/                    # Core API modules
│   ├── rag_engine.py      # Retrieval-Augmented Generation
│   ├── langchain_agents.py # Business intelligence agents  
│   ├── scoring_engine.py   # Business assessment scoring
│   ├── vector_store.py     # Vector database interface
│   └── whatsapp_agent.py   # WhatsApp integration
├── config/                 # Configuration management
│   ├── settings.py         # Environment-based settings
│   └── prompts.py          # AI prompt templates
├── data/                   # Data processing
│   └── processor.py        # Document processing pipeline
├── knowledge_base/         # Domain knowledge
│   ├── funding_database.py # Funding opportunities
│   └── regulatory_info.py  # Regulatory information
├── utils/                  # Utilities
│   ├── embeddings.py       # Text embedding service
│   ├── chunking.py         # Document chunking
│   └── context_enhancer.py # Context enhancement
├── app.py                  # Main Streamlit application
└── initialize_knowledge_base.py # Setup script
```

### Data Flow

1. **User Input** → Input validation and safety checks
2. **Query Processing** → Context enhancement and classification
3. **Retrieval** → Vector similarity search for relevant documents
4. **Generation** → AI-powered response generation with context
5. **Response** → Formatted output with sources and follow-ups

### Vector Database

- **Technology**: ChromaDB for local vector storage
- **Embeddings**: OpenAI text-embedding-ada-002
- **Content**: Business documents, funding data, regulatory information
- **Metadata**: Country, sector, funding stage, source type

## 🔧 Configuration

### Environment Variables

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `OPENAI_API_KEY` | Yes | OpenAI API key for AI functionality | None |
| `CHROMA_PERSIST_DIRECTORY` | No | Vector database directory | `./vector_db` |
| `COLLECTION_NAME` | No | Vector collection name | `african_business_knowledge` |
| `EMBEDDING_MODEL` | No | OpenAI embedding model | `text-embedding-ada-002` |
| `SIMILARITY_THRESHOLD` | No | Minimum similarity for retrieval | `0.3` |
| `MAX_RETRIEVED_CHUNKS` | No | Maximum documents per query | `10` |
| `CONTEXT_WINDOW` | No | Maximum context tokens | `4000` |

### Advanced Configuration

```python
# Custom settings in config/settings.py
class Settings:
    # Model parameters
    TEMPERATURE = 0.3
    MAX_TOKENS = 1000
    
    # Vector search
    SIMILARITY_THRESHOLD = 0.3
    MAX_RETRIEVED_CHUNKS = 10
    
    # Performance
    CACHE_TTL = 3600
    MAX_CONCURRENT_REQUESTS = 10
```

## 📊 Monitoring and Logging

### Application Logs

- **Location**: `ubuntuai.log` (main app), `knowledge_base_init.log` (initialization)
- **Levels**: DEBUG, INFO, WARNING, ERROR
- **Format**: Timestamp, module, level, message

### System Status

The web interface provides real-time system status:
- Configuration validation
- API connectivity
- Vector database health
- Performance metrics

### Error Handling

- **Graceful Degradation**: Fallback responses when services are unavailable
- **User-Friendly Messages**: Clear error messages for end users
- **Detailed Logging**: Comprehensive error tracking for debugging

## 🧪 Testing

### Manual Testing

1. **Basic Functionality**:
   ```bash
   python -c "
   from api.rag_engine import rag_engine
   response = rag_engine.query('Tell me about fintech in Nigeria')
   print(response['answer'])
   "
   ```

2. **Vector Store**:
   ```bash
   python -c "
   from api.vector_store import vector_store
   stats = vector_store.get_collection_stats()
   print(f'Documents: {stats[\"total_documents\"]}')
   "
   ```

3. **Settings Validation**:
   ```bash
   python -c "
   from config.settings import settings
   print('Valid config:', settings.validate_config())
   "
   ```

### Performance Testing

- **Query Response Time**: Target < 3 seconds for typical queries
- **Embedding Generation**: Batch processing for efficiency
- **Memory Usage**: Monitor vector database size and RAM usage

## 🚀 Deployment

### Local Development

```bash
# Development mode with auto-reload
streamlit run app.py --server.runOnSave true
```

### Production Deployment

**Docker (Recommended)**:
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN python initialize_knowledge_base.py

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.headless", "true"]
```

**Cloud Platforms**:
- **Streamlit Cloud**: Direct GitHub integration
- **Heroku**: Web dyno with PostgreSQL addon for persistence
- **AWS/GCP**: Container deployment with managed databases

### Environment Considerations

- **API Keys**: Use secure environment variable management
- **Vector Database**: Consider external vector databases for production scale
- **Caching**: Implement Redis for response caching in production
- **Monitoring**: Set up application performance monitoring

## 🤝 Contributing

### Development Setup

1. **Fork and Clone**:
   ```bash
   git clone https://github.com/yourusername/UbuntuAI.git
   cd UbuntuAI
   ```

2. **Virtual Environment**:
   ```bash
   python -m venv ubuntu_ai_env
   source ubuntu_ai_env/bin/activate  # Linux/Mac
   # or
   ubuntu_ai_env\Scripts\activate     # Windows
   ```

3. **Development Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Code Standards

- **Style**: Follow PEP 8 with 100-character line limit
- **Type Hints**: Use type annotations for all function signatures
- **Documentation**: Comprehensive docstrings for all public methods
- **Error Handling**: Proper exception handling with logging
- **Testing**: Add tests for new functionality

### Project Structure Guidelines

- **Modularity**: Keep components loosely coupled
- **Configuration**: Use environment variables for all settings
- **Logging**: Add appropriate logging for debugging and monitoring
- **Documentation**: Update README and docstrings for changes

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

### Common Issues

**"OpenAI API Key not configured"**
- Solution: Add `OPENAI_API_KEY` to your `.env` file

**"No module named 'chromadb'"**
- Solution: Run `pip install -r requirements.txt`

**"Empty vector database"**
- Solution: Run `python initialize_knowledge_base.py --force`

### Getting Help

- **Documentation**: Check this README and inline code documentation
- **Issues**: Open a GitHub issue for bugs or feature requests
- **Discussions**: Use GitHub Discussions for questions and ideas

### Contact

For technical support or business inquiries, please open an issue on GitHub.

---

**Built with ❤️ for African entrepreneurs and startups**

*UbuntuAI - Empowering African business ecosystems through AI*