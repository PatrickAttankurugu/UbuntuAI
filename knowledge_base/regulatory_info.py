from typing import Dict, List, Any

class RegulatoryInfoDatabase:
    def __init__(self):
        self.business_registration = self._load_business_registration_data()
        self.tax_information = self._load_tax_data()
        self.compliance_requirements = self._load_compliance_data()
        self.trade_regulations = self._load_trade_data()
    
    def _load_business_registration_data(self) -> Dict[str, Dict[str, Any]]:
        return {
            "Nigeria": {
                "registration_authority": "Corporate Affairs Commission (CAC)",
                "online_portal": "https://pre.cac.gov.ng",
                "required_documents": [
                    "Completed CAC forms",
                    "Memorandum and Articles of Association",
                    "Evidence of payment of registration fees",
                    "Statement of share capital and returns of allotment"
                ],
                "processing_time": "5-10 business days",
                "cost": "₦10,000 - ₦50,000 depending on share capital",
                "business_types": [
                    "Private Limited Company (Ltd)",
                    "Public Limited Company (PLC)",
                    "Business Name",
                    "Incorporated Trustees (Non-profit)"
                ],
                "additional_requirements": {
                    "tax_identification": "Federal Inland Revenue Service (FIRS)",
                    "pension_registration": "National Pension Commission",
                    "industrial_training_fund": "For companies with 5+ employees"
                }
            },
            "Kenya": {
                "registration_authority": "Registrar of Companies",
                "online_portal": "https://ecitizen.go.ke",
                "required_documents": [
                    "Memorandum and Articles of Association",
                    "Form CR1 (Application for Registration)",
                    "Certificate of compliance",
                    "Statement of nominal capital"
                ],
                "processing_time": "3-7 business days",
                "cost": "KSh 10,000 - KSh 20,000",
                "business_types": [
                    "Private Limited Company",
                    "Public Limited Company",
                    "Partnership",
                    "Sole Proprietorship"
                ],
                "additional_requirements": {
                    "pin_certificate": "Kenya Revenue Authority (KRA)",
                    "nhif_registration": "National Hospital Insurance Fund",
                    "nssf_registration": "National Social Security Fund"
                }
            },
            "South Africa": {
                "registration_authority": "Companies and Intellectual Property Commission (CIPC)",
                "online_portal": "https://www.cipc.co.za",
                "required_documents": [
                    "CoR 15.1A (Notice of Incorporation)",
                    "Memorandum of Incorporation",
                    "Form CoR 21.1 (Consent to Act as Director)"
                ],
                "processing_time": "5-10 business days",
                "cost": "R175 - R500",
                "business_types": [
                    "Private Company (Pty Ltd)",
                    "Public Company",
                    "Non-profit Company (NPC)",
                    "Close Corporation (CC)"
                ],
                "additional_requirements": {
                    "tax_registration": "South African Revenue Service (SARS)",
                    "uif_registration": "Unemployment Insurance Fund",
                    "compensation_fund": "Department of Employment and Labour"
                }
            },
            "Ghana": {
                "registration_authority": "Registrar General's Department",
                "online_portal": "https://www.rgd.gov.gh",
                "required_documents": [
                    "Statement and particulars of directors",
                    "Statement of share capital",
                    "Memorandum and Articles of Association",
                    "Statutory declaration of compliance"
                ],
                "processing_time": "3-5 business days",
                "cost": "GHS 500 - GHS 2,000",
                "business_types": [
                    "Private Limited Company",
                    "Public Limited Company",
                    "Partnership",
                    "Sole Proprietorship"
                ],
                "additional_requirements": {
                    "tin_registration": "Ghana Revenue Authority",
                    "ssnit_registration": "Social Security and National Insurance Trust",
                    "district_license": "Metropolitan/Municipal/District Assembly"
                }
            }
        }
    
    def _load_tax_data(self) -> Dict[str, Dict[str, Any]]:
        return {
            "Nigeria": {
                "corporate_tax_rate": "30%",
                "small_company_rate": "20% (for companies with turnover < ₦25M)",
                "vat_rate": "7.5%",
                "tax_authority": "Federal Inland Revenue Service (FIRS)",
                "filing_requirements": {
                    "annual_returns": "Within 6 months of year-end",
                    "monthly_vat": "21st of following month",
                    "paye": "Monthly remittance"
                },
                "incentives": [
                    "Pioneer Status Incentive (3-5 years tax holiday)",
                    "Investment Tax Credit",
                    "Research & Development allowance"
                ]
            },
            "Kenya": {
                "corporate_tax_rate": "30%",
                "resident_rate": "30%",
                "vat_rate": "16%",
                "tax_authority": "Kenya Revenue Authority (KRA)",
                "filing_requirements": {
                    "annual_returns": "Within 6 months of year-end",
                    "monthly_vat": "20th of following month",
                    "paye": "9th of following month"
                },
                "incentives": [
                    "Export Processing Zones (10-year tax holiday)",
                    "Special Economic Zones incentives",
                    "Investment deduction allowance"
                ]
            },
            "South Africa": {
                "corporate_tax_rate": "28%",
                "small_business_rate": "0-28% progressive rate",
                "vat_rate": "15%",
                "tax_authority": "South African Revenue Service (SARS)",
                "filing_requirements": {
                    "annual_returns": "Within 12 months of year-end",
                    "vat_returns": "Monthly or bi-monthly",
                    "paye": "7th or 15th of following month"
                },
                "incentives": [
                    "Section 12I Tax Allowance (manufacturing)",
                    "R&D Tax Incentive (150% deduction)",
                    "Skills Development Levy credits"
                ]
            }
        }
    
    def _load_compliance_data(self) -> Dict[str, List[str]]:
        return {
            "General Requirements": [
                "Maintain proper accounting records",
                "File annual returns with company registry",
                "Hold annual general meetings",
                "Maintain statutory registers",
                "Comply with tax obligations"
            ],
            "Employment Law": [
                "Register with relevant pension/social security authorities",
                "Maintain employee records",
                "Comply with minimum wage requirements",
                "Provide safe working conditions",
                "Follow proper dismissal procedures"
            ],
            "Data Protection": [
                "Comply with local data protection laws",
                "Obtain consent for data processing",
                "Implement data security measures",
                "Register with data protection authorities where required",
                "Provide privacy notices to users"
            ],
            "Intellectual Property": [
                "Register trademarks and patents",
                "Respect third-party IP rights",
                "Implement IP protection policies",
                "File appropriate applications",
                "Monitor for infringement"
            ]
        }
    
    def _load_trade_data(self) -> Dict[str, Any]:
        return {
            "Import/Export Requirements": {
                "common_documents": [
                    "Commercial invoice",
                    "Packing list",
                    "Bill of lading/Airway bill",
                    "Certificate of origin",
                    "Import/Export permit"
                ],
                "customs_procedures": [
                    "Register with customs authority",
                    "Obtain relevant licenses",
                    "Pay applicable duties and taxes",
                    "Comply with product standards",
                    "Submit required documentation"
                ]
            },
            "Regional Trade Agreements": {
                "AfCFTA": {
                    "name": "African Continental Free Trade Agreement",
                    "benefits": "Preferential tariffs between member countries",
                    "requirements": "Certificate of origin compliance"
                },
                "EAC": {
                    "name": "East African Community",
                    "members": ["Kenya", "Uganda", "Tanzania", "Rwanda", "Burundi", "South Sudan"],
                    "benefits": "Free movement of goods and services"
                },
                "ECOWAS": {
                    "name": "Economic Community of West African States",
                    "members": ["Nigeria", "Ghana", "Senegal", "Ivory Coast"],
                    "benefits": "Common external tariff and free trade"
                }
            }
        }
    
    def get_business_registration_info(self, country: str) -> Dict[str, Any]:
        return self.business_registration.get(country, {})
    
    def get_tax_information(self, country: str) -> Dict[str, Any]:
        return self.tax_information.get(country, {})
    
    def get_compliance_checklist(self, business_type: str = "general") -> List[str]:
        if business_type == "tech_startup":
            return (
                self.compliance_requirements["General Requirements"] +
                self.compliance_requirements["Employment Law"] +
                self.compliance_requirements["Data Protection"] +
                self.compliance_requirements["Intellectual Property"]
            )
        return self.compliance_requirements.get("General Requirements", [])
    
    def generate_country_guide(self, country: str) -> str:
        reg_info = self.get_business_registration_info(country)
        tax_info = self.get_tax_information(country)
        
        if not reg_info:
            return f"Regulatory information for {country} is not available in our database."
        
        guide = f"# Business Setup Guide for {country}\n\n"
        
        guide += "## Business Registration\n"
        guide += f"**Authority**: {reg_info.get('registration_authority')}\n"
        guide += f"**Online Portal**: {reg_info.get('online_portal')}\n"
        guide += f"**Processing Time**: {reg_info.get('processing_time')}\n"
        guide += f"**Cost**: {reg_info.get('cost')}\n\n"
        
        guide += "### Required Documents:\n"
        for doc in reg_info.get('required_documents', []):
            guide += f"- {doc}\n"
        
        guide += "\n### Business Types Available:\n"
        for btype in reg_info.get('business_types', []):
            guide += f"- {btype}\n"
        
        if tax_info:
            guide += "\n## Tax Information\n"
            guide += f"**Corporate Tax Rate**: {tax_info.get('corporate_tax_rate')}\n"
            guide += f"**VAT Rate**: {tax_info.get('vat_rate')}\n"
            guide += f"**Tax Authority**: {tax_info.get('tax_authority')}\n\n"
            
            if 'incentives' in tax_info:
                guide += "### Available Tax Incentives:\n"
                for incentive in tax_info['incentives']:
                    guide += f"- {incentive}\n"
        
        guide += "\n## Additional Requirements\n"
        additional = reg_info.get('additional_requirements', {})
        for req, desc in additional.items():
            guide += f"- **{req.replace('_', ' ').title()}**: {desc}\n"
        
        return guide

regulatory_db = RegulatoryInfoDatabase()