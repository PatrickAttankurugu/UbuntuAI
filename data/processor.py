"""
Modern Document Processing Pipeline for UbuntuAI
Uses LangChain document loaders and advanced chunking strategies
"""

import json
import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import os
from datetime import datetime

# LangChain document loaders
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    TextLoader,
    PDFPlumberLoader,
    CSVLoader,
    JSONLoader,
    WebBaseLoader,
    DirectoryLoader,
    UnstructuredFileLoader
)

# Data processing
import pandas as pd

from utils.chunking import chunking_manager
from utils.context_enhancer import context_enhancer
from knowledge_base.funding_database import funding_db
from knowledge_base.regulatory_info import regulatory_db
from config.settings import settings

logger = logging.getLogger(__name__)

class ModernDataProcessor:
    """
    Modern data processing pipeline with LangChain integration
    """
    
    def __init__(self):
        self.chunking_manager = chunking_manager
        self.context_enhancer = context_enhancer
        self.supported_extensions = {
            '.txt': TextLoader,
            '.pdf': PDFPlumberLoader,
            '.csv': CSVLoader,
            '.json': JSONLoader,
            '.md': TextLoader,
            '.html': UnstructuredFileLoader
        }
        
        logger.info("Modern data processor initialized")
    
    def process_funding_data(self) -> List[Document]:
        """Process funding database into LangChain documents"""
        
        logger.info("Processing funding opportunities data...")
        documents = []
        
        try:
            for opportunity in funding_db.funding_opportunities:
                # Format content
                content = self._format_funding_opportunity(opportunity)
                
                # Create metadata
                metadata = {
                    "source": "Internal Funding Database",
                    "type": "funding_opportunity",
                    "country": opportunity.get("country", ""),
                    "focus_sectors": opportunity.get("focus_sectors", []),
                    "funding_stages": opportunity.get("stage", []),
                    "funding_type": opportunity.get("type", ""),
                    "name": opportunity.get("name", ""),
                    "investment_range": opportunity.get("typical_investment", ""),
                    "processed_date": datetime.now().isoformat()
                }
                
                # Create document
                doc = Document(page_content=content, metadata=metadata)
                documents.append(doc)
            
            logger.info(f"Processed {len(documents)} funding opportunities")
            return documents
            
        except Exception as e:
            logger.error(f"Error processing funding data: {e}")
            return []
    
    def process_regulatory_data(self) -> List[Document]:
        """Process regulatory database into LangChain documents"""
        
        logger.info("Processing regulatory information data...")
        documents = []
        
        try:
            # Process business registration data
            for country, reg_info in regulatory_db.business_registration.items():
                content = self._format_regulatory_info(country, reg_info, "registration")
                
                metadata = {
                    "source": "Internal Regulatory Database",
                    "type": "regulatory_info",
                    "category": "business_registration",
                    "country": country,
                    "processed_date": datetime.now().isoformat()
                }
                
                doc = Document(page_content=content, metadata=metadata)
                documents.append(doc)
            
            # Process tax information
            for country, tax_info in regulatory_db.tax_information.items():
                content = self._format_tax_info(country, tax_info)
                
                metadata = {
                    "source": "Internal Regulatory Database",
                    "type": "tax_info",
                    "category": "taxation",
                    "country": country,
                    "processed_date": datetime.now().isoformat()
                }
                
                doc = Document(page_content=content, metadata=metadata)
                documents.append(doc)
            
            logger.info(f"Processed {len(documents)} regulatory documents")
            return documents
            
        except Exception as e:
            logger.error(f"Error processing regulatory data: {e}")
            return []
    
    def process_file(self, file_path: Union[str, Path]) -> List[Document]:
        """Process a single file into documents"""
        
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return []
        
        extension = file_path.suffix.lower()
        
        if extension not in self.supported_extensions:
            logger.warning(f"Unsupported file type: {extension}")
            return []
        
        try:
            # Get appropriate loader
            loader_class = self.supported_extensions[extension]
            
            # Special handling for different file types
            if extension == '.csv':
                loader = loader_class(
                    file_path=str(file_path),
                    csv_args={'delimiter': ','}
                )
            elif extension == '.json':
                loader = loader_class(
                    file_path=str(file_path),
                    jq_schema='.',
                    text_content=False
                )
            else:
                loader = loader_class(str(file_path))
            
            # Load documents
            documents = loader.load()
            
            # Enhance metadata
            for doc in documents:
                doc.metadata.update({
                    "source": f"File: {file_path.name}",
                    "file_path": str(file_path),
                    "file_type": extension,
                    "file_size": file_path.stat().st_size,
                    "processed_date": datetime.now().isoformat()
                })
            
            logger.info(f"Processed file {file_path.name}: {len(documents)} documents")
            return documents
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return []
    
    def process_directory(self, 
                         directory_path: Union[str, Path],
                         recursive: bool = True,
                         file_pattern: str = "**/*") -> List[Document]:
        """Process all supported files in a directory"""
        
        directory_path = Path(directory_path)
        
        if not directory_path.exists() or not directory_path.is_dir():
            logger.error(f"Directory not found: {directory_path}")
            return []
        
        try:
            # Use DirectoryLoader for efficient processing
            loader = DirectoryLoader(
                path=str(directory_path),
                glob=file_pattern,
                recursive=recursive,
                loader_kwargs={"autodetect_encoding": True}
            )
            
            documents = loader.load()
            
            # Enhance metadata
            for doc in documents:
                doc.metadata.update({
                    "source_directory": str(directory_path),
                    "processed_date": datetime.now().isoformat()
                })
            
            logger.info(f"Processed directory {directory_path}: {len(documents)} documents")
            return documents
            
        except Exception as e:
            logger.error(f"Error processing directory {directory_path}: {e}")
            return []
    
    def process_web_content(self, urls: List[str]) -> List[Document]:
        """Process web content from URLs"""
        
        if not urls:
            return []
        
        try:
            loader = WebBaseLoader(web_paths=urls)
            documents = loader.load()
            
            # Enhance metadata
            for doc in documents:
                doc.metadata.update({
                    "source_type": "web_content",
                    "processed_date": datetime.now().isoformat()
                })
            
            logger.info(f"Processed {len(urls)} URLs: {len(documents)} documents")
            return documents
            
        except Exception as e:
            logger.error(f"Error processing web content: {e}")
            return []
    
    def chunk_documents(self, 
                       documents: List[Document],
                       strategy: str = None,
                       **kwargs) -> List[Document]:
        """Chunk documents using specified strategy"""
        
        if not self.chunking_manager:
            logger.error("Chunking manager not available")
            return documents
        
        try:
            # Use chunking manager
            result = self.chunking_manager.chunk_documents(
                documents=documents,
                strategy=strategy,
                **kwargs
            )
            
            logger.info(f"Chunked {len(documents)} documents into {len(result.chunks)} chunks")
            logger.debug(f"Chunking statistics: {result.statistics}")
            
            return result.chunks
            
        except Exception as e:
            logger.error(f"Error chunking documents: {e}")
            return documents
    
    def enhance_documents_with_context(self, documents: List[Document]) -> List[Document]:
        """Enhance documents with business context"""
        
        enhanced_documents = []
        
        for doc in documents:
            try:
                # Extract business entities
                entities = self.context_enhancer.extract_business_entities(doc.page_content)
                
                # Classify content type
                classification = self.context_enhancer.classify_content_type(
                    doc.page_content, 
                    doc.metadata
                )
                
                # Create enhanced metadata
                enhanced_metadata = doc.metadata.copy()
                enhanced_metadata.update({
                    "extracted_entities": entities,
                    "content_classification": classification,
                    "context_enhanced": True
                })
                
                # Create enhanced document
                enhanced_doc = Document(
                    page_content=doc.page_content,
                    metadata=enhanced_metadata
                )
                
                enhanced_documents.append(enhanced_doc)
                
            except Exception as e:
                logger.warning(f"Failed to enhance document: {e}")
                enhanced_documents.append(doc)
        
        logger.info(f"Enhanced {len(enhanced_documents)} documents with business context")
        return enhanced_documents
    
    def generate_sample_documents(self) -> List[Document]:
        """Generate sample documents for testing and initial population"""
        
        logger.info("Generating sample documents...")
        documents = []
        
        # Sample funding documents
        funding_samples = [
            {
                "content": """TLcom Capital is a leading African VC firm that invests in early-stage technology companies across Africa. Based in Kenya with offices in Nigeria and London, TLcom focuses on fintech, agritech, and healthtech startups. The firm typically invests $500K to $15M in Series A and B rounds.

TLcom has a strong track record of supporting African entrepreneurs, with notable portfolio companies including Twiga Foods (Kenya's leading B2B food distribution platform), Andela (software engineering talent marketplace), and uLesson (education technology platform).

The firm looks for startups with strong local market understanding, scalable technology solutions, and experienced founding teams who can execute in challenging African markets. TLcom particularly values companies that can demonstrate product-market fit and have clear paths to profitability.

Application process involves submitting a comprehensive pitch deck and executive summary via email to their investment team. The firm conducts thorough due diligence including market analysis, technical evaluation, and founder background checks before making investment decisions.""",
                "metadata": {
                    "source": "Sample Data - Funding",
                    "type": "funding_opportunity",
                    "country": "Kenya",
                    "focus_sectors": ["Fintech", "Agritech", "Healthtech"],
                    "funding_stages": ["Series A", "Series B"],
                    "name": "TLcom Capital",
                    "investment_range": "$500K - $15M"
                }
            },
            {
                "content": """The African Development Bank (AfDB) Innovation Challenge is an annual competition that supports scalable solutions addressing development challenges across Africa. The program provides grants ranging from $50K to $500K for startups in agritech, clean energy, healthcare, and education sectors.

The Innovation Challenge targets early-stage companies with innovative solutions that can create significant social impact across African communities. Successful applicants receive not only funding but also mentorship from industry experts and access to AfDB's extensive network of development partners.

The application process involves multiple phases: initial application review, pitch presentations to expert panels, and comprehensive due diligence for finalists. Selection criteria include innovation potential, market viability, social impact, and team capability.

Previous winners have included SolarNow (renewable energy), Farmerline (agricultural technology), and mPharma (healthcare supply chain). The challenge particularly encourages solutions that can be scaled across multiple African countries and demonstrate sustainable business models.""",
                "metadata": {
                    "source": "Sample Data - Grant Program",
                    "type": "grant_program",
                    "country": "Continental",
                    "focus_sectors": ["Agritech", "Clean Energy", "Healthcare", "Education"],
                    "funding_stages": ["Pre-seed", "Seed"],
                    "name": "AfDB Innovation Challenge",
                    "investment_range": "$50K - $500K"
                }
            }
        ]
        
        # Sample regulatory documents
        regulatory_samples = [
            {
                "content": """Business registration in Nigeria is administered by the Corporate Affairs Commission (CAC) through their online portal at pre.cac.gov.ng. The registration process typically takes 5-10 business days and costs between ₦10,000 to ₦50,000 depending on the company's share capital structure.

Required documents include completed CAC forms, Memorandum and Articles of Association, evidence of payment of registration fees, and statement of share capital and returns of allotment. The process has been significantly streamlined with digital integration, allowing entrepreneurs to complete most steps online.

Available business structures include Private Limited Company (Ltd), Public Limited Company (PLC), Business Name registration, and Incorporated Trustees for non-profit organizations. Each structure has different compliance requirements and tax implications.

Additional requirements after incorporation include tax registration with the Federal Inland Revenue Service (FIRS), pension registration with the National Pension Commission for companies with employees, and Industrial Training Fund registration for companies with five or more employees. These registrations are essential for legal compliance and business operations.""",
                "metadata": {
                    "source": "Sample Data - Regulatory",
                    "type": "regulatory_info",
                    "country": "Nigeria",
                    "category": "business_registration"
                }
            },
            {
                "content": """Kenya's corporate tax system imposes a standard rate of 30% on resident companies, with Value Added Tax (VAT) at 16%. The Kenya Revenue Authority (KRA) manages tax administration through their iTax online platform, which has modernized tax compliance processes.

Companies must file annual returns within six months of their financial year-end, submit monthly VAT returns by the 20th of the following month, and remit Pay As You Earn (PAYE) taxes by the 9th of the following month. The digitalization of these processes has improved efficiency and transparency.

Kenya offers several tax incentives to encourage investment and economic growth. Export Processing Zones provide 10-year tax holidays for qualifying manufacturers, Special Economic Zones offer additional incentives for strategic investments, and investment deduction allowances are available for capital expenditures.

Small businesses may qualify for reduced compliance requirements and simplified tax procedures. The government has introduced various programs to support SME growth, including tax education initiatives and simplified filing procedures for businesses below certain revenue thresholds.""",
                "metadata": {
                    "source": "Sample Data - Tax Info",
                    "type": "tax_info",
                    "country": "Kenya",
                    "category": "taxation"
                }
            }
        ]
        
        # Sample success stories
        success_samples = [
            {
                "content": """Flutterwave, founded in 2016 by Iyinoluwa Aboyeji and Olugbenga Agboola, has become one of Africa's most valuable fintech unicorns with a valuation exceeding $3 billion. The Nigerian-based payment infrastructure company enables businesses to build customizable payment applications through robust APIs that support multiple currencies and payment methods.

Starting with the challenge of facilitating seamless payments across African borders, Flutterwave identified the fragmented nature of payment systems across the continent as a major opportunity. The founders leveraged their deep understanding of both local market needs and global technology standards to build a solution that bridges this gap.

The company's growth trajectory has been remarkable, processing over 200 million transactions annually for more than 900,000 businesses including global companies like Uber, Facebook, and Netflix. Flutterwave has raised over $400M in funding from leading investors and achieved unicorn status in 2021.

Key success factors include their focus on solving real payment challenges, building robust and reliable infrastructure, expanding strategically across multiple African markets, and maintaining strong relationships with both local and international partners. Their API-first approach has enabled rapid integration and scaling across diverse business models.""",
                "metadata": {
                    "source": "Sample Data - Success Story",
                    "type": "success_story",
                    "country": "Nigeria",
                    "focus_sectors": ["Fintech"],
                    "company": "Flutterwave",
                    "funding_raised": "$400M+"
                }
            }
        ]
        
        # Convert samples to Documents
        all_samples = funding_samples + regulatory_samples + success_samples
        
        for sample in all_samples:
            doc = Document(
                page_content=sample["content"],
                metadata={
                    **sample["metadata"],
                    "generated_sample": True,
                    "processed_date": datetime.now().isoformat()
                }
            )
            documents.append(doc)
        
        logger.info(f"Generated {len(documents)} sample documents")
        return documents
    
    def process_pipeline(self, 
                        sources: Dict[str, Any],
                        chunking_strategy: str = None,
                        enhance_context: bool = True) -> List[Document]:
        """Complete processing pipeline for multiple data sources"""
        
        logger.info("Starting complete document processing pipeline...")
        all_documents = []
        
        # Process different source types
        if sources.get("funding_data", True):
            funding_docs = self.process_funding_data()
            all_documents.extend(funding_docs)
        
        if sources.get("regulatory_data", True):
            regulatory_docs = self.process_regulatory_data()
            all_documents.extend(regulatory_docs)
        
        if sources.get("sample_data", True):
            sample_docs = self.generate_sample_documents()
            all_documents.extend(sample_docs)
        
        if sources.get("files"):
            for file_path in sources["files"]:
                file_docs = self.process_file(file_path)
                all_documents.extend(file_docs)
        
        if sources.get("directories"):
            for dir_path in sources["directories"]:
                dir_docs = self.process_directory(dir_path)
                all_documents.extend(dir_docs)
        
        if sources.get("urls"):
            web_docs = self.process_web_content(sources["urls"])
            all_documents.extend(web_docs)
        
        logger.info(f"Loaded {len(all_documents)} documents from all sources")
        
        # Enhance with business context
        if enhance_context:
            all_documents = self.enhance_documents_with_context(all_documents)
        
        # Chunk documents
        chunked_documents = self.chunk_documents(
            all_documents, 
            strategy=chunking_strategy
        )
        
        logger.info(f"Processing pipeline complete: {len(chunked_documents)} final chunks")
        
        return chunked_documents
    
    def _format_funding_opportunity(self, opportunity: Dict[str, Any]) -> str:
        """Format funding opportunity data into readable text"""
        
        content_parts = [
            f"Funding Opportunity: {opportunity['name']}",
            f"Type: {opportunity.get('type', 'N/A')}",
            f"Location: {opportunity.get('country', 'N/A')}",
            f"Focus Sectors: {', '.join(opportunity.get('focus_sectors', []))}",
            f"Funding Stages: {', '.join(opportunity.get('stage', []))}",
            f"Typical Investment: {opportunity.get('typical_investment', 'N/A')}",
            "",
            f"Description: {opportunity.get('description', '')}",
        ]
        
        if opportunity.get('application_process'):
            content_parts.extend([
                "",
                f"Application Process: {opportunity['application_process']}"
            ])
        
        if opportunity.get('portfolio'):
            content_parts.extend([
                "",
                f"Portfolio Companies: {', '.join(opportunity['portfolio'])}"
            ])
        
        if opportunity.get('contact'):
            content_parts.extend([
                "",
                f"Contact: {opportunity['contact']}"
            ])
        
        if opportunity.get('website'):
            content_parts.extend([
                "",
                f"Website: {opportunity['website']}"
            ])
        
        return "\n".join(content_parts)
    
    def _format_regulatory_info(self, country: str, reg_info: Dict[str, Any], category: str) -> str:
        """Format regulatory information into readable text"""
        
        content_parts = [
            f"Business Registration Requirements - {country}",
            "",
            f"Registration Authority: {reg_info.get('registration_authority', 'N/A')}",
            f"Online Portal: {reg_info.get('online_portal', 'N/A')}",
            f"Processing Time: {reg_info.get('processing_time', 'N/A')}",
            f"Registration Cost: {reg_info.get('cost', 'N/A')}",
        ]
        
        if reg_info.get('required_documents'):
            content_parts.extend([
                "",
                "Required Documents:"
            ])
            for doc in reg_info['required_documents']:
                content_parts.append(f"• {doc}")
        
        if reg_info.get('business_types'):
            content_parts.extend([
                "",
                "Available Business Types:"
            ])
            for btype in reg_info['business_types']:
                content_parts.append(f"• {btype}")
        
        if reg_info.get('additional_requirements'):
            content_parts.extend([
                "",
                "Additional Requirements:"
            ])
            for req, desc in reg_info['additional_requirements'].items():
                content_parts.append(f"• {req.replace('_', ' ').title()}: {desc}")
        
        return "\n".join(content_parts)
    
    def _format_tax_info(self, country: str, tax_info: Dict[str, Any]) -> str:
        """Format tax information into readable text"""
        
        content_parts = [
            f"Tax Information - {country}",
            "",
            f"Corporate Tax Rate: {tax_info.get('corporate_tax_rate', 'N/A')}",
            f"VAT Rate: {tax_info.get('vat_rate', 'N/A')}",
            f"Tax Authority: {tax_info.get('tax_authority', 'N/A')}",
        ]
        
        if tax_info.get('small_company_rate') or tax_info.get('small_business_rate'):
            rate = tax_info.get('small_company_rate') or tax_info.get('small_business_rate')
            content_parts.extend([
                "",
                f"Small Business Rate: {rate}"
            ])
        
        if tax_info.get('filing_requirements'):
            content_parts.extend([
                "",
                "Filing Requirements:"
            ])
            for req, desc in tax_info['filing_requirements'].items():
                content_parts.append(f"• {req.replace('_', ' ').title()}: {desc}")
        
        if tax_info.get('incentives'):
            content_parts.extend([
                "",
                "Tax Incentives:"
            ])
            for incentive in tax_info['incentives']:
                content_parts.append(f"• {incentive}")
        
        return "\n".join(content_parts)

# Global data processor instance
try:
    data_processor = ModernDataProcessor()
    logger.info("Modern data processor initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize data processor: {e}")
    data_processor = None