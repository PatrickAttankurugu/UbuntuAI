import json
from typing import List, Dict, Any, Optional
from utils.chunking import chunker
from utils.context_enhancer import context_enhancer
from knowledge_base.funding_database import funding_db
from knowledge_base.regulatory_info import regulatory_db

class DataProcessor:
    def __init__(self):
        self.chunker = chunker
        self.context_enhancer = context_enhancer
        
    def process_funding_data(self) -> List[Dict[str, Any]]:
        processed_chunks = []
        
        # Process funding opportunities
        for opportunity in funding_db.funding_opportunities:
            content = self._format_funding_opportunity(opportunity)
            metadata = {
                "source": "Internal Funding Database",
                "type": "funding_opportunity",
                "country": opportunity.get("country"),
                "sector": opportunity.get("focus_sectors", []),
                "funding_stage": opportunity.get("stage", []),
                "funding_type": opportunity.get("type")
            }
            
            chunks = self.chunker.chunk_document(content, metadata)
            processed_chunks.extend(chunks)
        
        return processed_chunks
    
    def process_regulatory_data(self) -> List[Dict[str, Any]]:
        processed_chunks = []
        
        # Process business registration data
        for country, reg_info in regulatory_db.business_registration.items():
            content = self._format_regulatory_info(country, reg_info)
            metadata = {
                "source": "Internal Regulatory Database",
                "type": "regulatory_info",
                "country": country,
                "category": "business_registration"
            }
            
            chunks = self.chunker.chunk_document(content, metadata)
            processed_chunks.extend(chunks)
        
        # Process tax information
        for country, tax_info in regulatory_db.tax_information.items():
            content = self._format_tax_info(country, tax_info)
            metadata = {
                "source": "Internal Regulatory Database",
                "type": "tax_info",
                "country": country,
                "category": "taxation"
            }
            
            chunks = self.chunker.chunk_document(content, metadata)
            processed_chunks.extend(chunks)
        
        return processed_chunks
    
    def _format_funding_opportunity(self, opportunity: Dict[str, Any]) -> str:
        content_parts = [
            f"Funding Opportunity: {opportunity['name']}",
            f"Type: {opportunity.get('type', 'N/A')}",
            f"Country: {opportunity.get('country', 'N/A')}",
            f"Focus Sectors: {', '.join(opportunity.get('focus_sectors', []))}",
            f"Funding Stages: {', '.join(opportunity.get('stage', []))}",
            f"Typical Investment: {opportunity.get('typical_investment', 'N/A')}",
            f"Description: {opportunity.get('description', '')}",
        ]
        
        if opportunity.get('application_process'):
            content_parts.append(f"Application Process: {opportunity['application_process']}")
        
        if opportunity.get('portfolio'):
            content_parts.append(f"Portfolio Companies: {', '.join(opportunity['portfolio'])}")
        
        if opportunity.get('contact'):
            content_parts.append(f"Contact: {opportunity['contact']}")
        
        return "\n".join(content_parts)
    
    def _format_regulatory_info(self, country: str, reg_info: Dict[str, Any]) -> str:
        content_parts = [
            f"Business Registration in {country}",
            f"Registration Authority: {reg_info.get('registration_authority', 'N/A')}",
            f"Online Portal: {reg_info.get('online_portal', 'N/A')}",
            f"Processing Time: {reg_info.get('processing_time', 'N/A')}",
            f"Cost: {reg_info.get('cost', 'N/A')}",
        ]
        
        if reg_info.get('required_documents'):
            content_parts.append("Required Documents:")
            for doc in reg_info['required_documents']:
                content_parts.append(f"- {doc}")
        
        if reg_info.get('business_types'):
            content_parts.append("Available Business Types:")
            for btype in reg_info['business_types']:
                content_parts.append(f"- {btype}")
        
        if reg_info.get('additional_requirements'):
            content_parts.append("Additional Requirements:")
            for req, desc in reg_info['additional_requirements'].items():
                content_parts.append(f"- {req.replace('_', ' ').title()}: {desc}")
        
        return "\n".join(content_parts)
    
    def _format_tax_info(self, country: str, tax_info: Dict[str, Any]) -> str:
        content_parts = [
            f"Tax Information for {country}",
            f"Corporate Tax Rate: {tax_info.get('corporate_tax_rate', 'N/A')}",
            f"VAT Rate: {tax_info.get('vat_rate', 'N/A')}",
            f"Tax Authority: {tax_info.get('tax_authority', 'N/A')}",
        ]
        
        if tax_info.get('small_company_rate') or tax_info.get('small_business_rate'):
            rate = tax_info.get('small_company_rate') or tax_info.get('small_business_rate')
            content_parts.append(f"Small Business Rate: {rate}")
        
        if tax_info.get('filing_requirements'):
            content_parts.append("Filing Requirements:")
            for req, desc in tax_info['filing_requirements'].items():
                content_parts.append(f"- {req.replace('_', ' ').title()}: {desc}")
        
        if tax_info.get('incentives'):
            content_parts.append("Tax Incentives:")
            for incentive in tax_info['incentives']:
                content_parts.append(f"- {incentive}")
        
        return "\n".join(content_parts)
    
    def generate_sample_documents(self) -> List[Dict[str, Any]]:
        """Generate sample documents for testing when no external data is available"""
        sample_docs = []
        
        # Sample funding documents
        funding_samples = [
            {
                "content": """TLcom Capital is a leading African VC firm that invests in early-stage technology companies across Africa. Based in Kenya with offices in Nigeria and London, TLcom focuses on fintech, agritech, and healthtech startups. The firm typically invests $500K to $15M in Series A and B rounds. Notable portfolio companies include Twiga Foods, Andela, and uLesson. TLcom looks for startups with strong local market understanding, scalable technology solutions, and experienced founding teams. Application process involves submitting a pitch deck and executive summary via email.""",
                "metadata": {
                    "source": "Sample Data",
                    "type": "funding_opportunity",
                    "country": "Kenya",
                    "sector": ["Fintech", "Agritech", "Healthtech"],
                    "funding_stage": ["Series A", "Series B"]
                }
            },
            {
                "content": """The African Development Bank (AfDB) Innovation Challenge supports scalable solutions addressing development challenges across Africa. The program provides grants ranging from $50K to $500K for startups in agritech, clean energy, healthcare, and education sectors. The annual competition involves multiple phases including application review, pitch presentations, and due diligence. Successful applicants receive funding, mentorship, and access to AfDB's network of development partners. The challenge is open to early-stage companies with innovative solutions that can create significant social impact.""",
                "metadata": {
                    "source": "Sample Data",
                    "type": "grant_program",
                    "country": "Continental",
                    "sector": ["Agritech", "Clean Energy", "Healthcare", "Education"],
                    "funding_stage": ["Pre-seed", "Seed"]
                }
            }
        ]
        
        # Sample regulatory documents
        regulatory_samples = [
            {
                "content": """Business registration in Nigeria is handled by the Corporate Affairs Commission (CAC) through their online portal at pre.cac.gov.ng. The process typically takes 5-10 business days and costs between ₦10,000 to ₦50,000 depending on share capital. Required documents include completed CAC forms, Memorandum and Articles of Association, evidence of payment, and statement of share capital. Available business types include Private Limited Company (Ltd), Public Limited Company (PLC), Business Name, and Incorporated Trustees for non-profits. Additional requirements include tax registration with FIRS, pension registration with PENCOM, and ITF registration for companies with 5+ employees.""",
                "metadata": {
                    "source": "Sample Data",
                    "type": "regulatory_info",
                    "country": "Nigeria",
                    "category": "business_registration"
                }
            },
            {
                "content": """Kenya's corporate tax rate is 30% for resident companies, with VAT at 16%. The Kenya Revenue Authority (KRA) manages tax administration. Companies must file annual returns within 6 months of year-end, monthly VAT returns by the 20th of the following month, and PAYE by the 9th of the following month. Tax incentives include Export Processing Zones offering 10-year tax holidays, Special Economic Zones incentives, and investment deduction allowances. Small businesses may qualify for reduced rates and simplified filing procedures.""",
                "metadata": {
                    "source": "Sample Data",
                    "type": "tax_info",
                    "country": "Kenya",
                    "category": "taxation"
                }
            }
        ]
        
        # Sample success stories
        success_samples = [
            {
                "content": """Flutterwave, founded in 2016 by Iyinoluwa Aboyeji and Olugbenga Agboola, became one of Africa's most valuable fintech unicorns. The Nigerian-based payment infrastructure company enables businesses to build customizable payment applications through robust APIs. Starting with the challenge of facilitating seamless payments across African borders, Flutterwave raised over $400M in funding and achieved a $3B valuation by 2022. The company processes over 200 million transactions annually for more than 900,000 businesses including Uber, Facebook, and Netflix. Key success factors include focus on solving real payment challenges, building robust infrastructure, and expanding across multiple African markets.""",
                "metadata": {
                    "source": "Sample Data",
                    "type": "success_story",
                    "country": "Nigeria",
                    "sector": ["Fintech"],
                    "company": "Flutterwave"
                }
            },
            {
                "content": """Andela, founded in 2014, pioneered the distributed engineering model by training African developers and connecting them with global technology companies. Starting in Nigeria with physical campuses, Andela evolved to a fully remote model serving developers across Africa. The company raised over $180M from investors including Google Ventures and has worked with companies like GitHub, ViacomCBS, and Cloudflare. Andela's success demonstrates the potential of African tech talent and the effectiveness of the remote work model. The pivot from bootcamp-style training to talent marketplace showed adaptability and market responsiveness.""",
                "metadata": {
                    "source": "Sample Data",
                    "type": "success_story",
                    "country": "Nigeria",
                    "sector": ["Technology", "Education"],
                    "company": "Andela"
                }
            }
        ]
        
        all_samples = funding_samples + regulatory_samples + success_samples
        
        # Process each sample through chunking
        for sample in all_samples:
            chunks = self.chunker.chunk_document(
                text=sample["content"],
                metadata=sample["metadata"]
            )
            sample_docs.extend(chunks)
        
        return sample_docs
    
    def prepare_documents_for_vectorstore(self, processed_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Prepare processed chunks for vector store insertion"""
        documents = []
        metadatas = []
        ids = []
        
        for i, chunk_data in enumerate(processed_chunks):
            documents.append(chunk_data["content"])
            metadatas.append(chunk_data["metadata"])
            ids.append(f"chunk_{i}")
        
        return {
            "documents": documents,
            "metadatas": metadatas,
            "ids": ids
        }

data_processor = DataProcessor()