from typing import List, Dict, Any
from datetime import datetime

class FundingDatabase:
    def __init__(self):
        self.funding_opportunities = self._load_funding_data()
        self.vc_firms = self._load_vc_data()
        self.accelerators = self._load_accelerator_data()
        self.government_programs = self._load_government_programs()
        self.grants = self._load_grants_data()
    
    def _load_funding_data(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "Flutterwave Venture",
                "type": "VC Firm",
                "country": "Nigeria",
                "focus_sectors": ["Fintech", "E-commerce", "Logistics"],
                "stage": ["Seed", "Series A"],
                "typical_investment": "$100K - $5M",
                "description": "Pan-African fintech company that also invests in early-stage African startups.",
                "contact": "ventures@flutterwave.com",
                "website": "https://flutterwave.com/ventures",
                "application_process": "Online application through website",
                "portfolio": ["Paystack", "Mono", "Piggyvest"]
            },
            {
                "name": "TLcom Capital",
                "type": "VC Firm", 
                "country": "Kenya",
                "focus_sectors": ["Fintech", "Agritech", "Healthtech", "Edtech"],
                "stage": ["Seed", "Series A", "Series B"],
                "typical_investment": "$500K - $15M",
                "description": "Leading African VC firm focused on scalable tech companies across Africa.",
                "contact": "info@tlcom.co.uk",
                "website": "https://tlcom.co.uk",
                "application_process": "Email pitch deck and executive summary",
                "portfolio": ["Twiga Foods", "Andela", "uLesson"]
            },
            {
                "name": "Partech Africa",
                "type": "VC Firm",
                "country": "Senegal",
                "focus_sectors": ["Fintech", "E-commerce", "Logistics", "Healthtech"],
                "stage": ["Seed", "Series A"],
                "typical_investment": "$200K - $3M",
                "description": "Early-stage VC fund focused on startups in Africa and the Middle East.",
                "contact": "africa@partechpartners.com",
                "website": "https://partechpartners.com/africa",
                "application_process": "Online form submission",
                "portfolio": ["Wave", "TradeDepot", "Yoco"]
            },
            {
                "name": "Catalyst Fund",
                "type": "Accelerator",
                "country": "Global (Africa Focus)",
                "focus_sectors": ["Fintech", "Digital Commerce"],
                "stage": ["Pre-seed", "Seed"],
                "typical_investment": "$25K - $100K",
                "description": "Inclusive fintech accelerator supporting startups serving underserved markets.",
                "contact": "apply@catalystfund.org",
                "website": "https://www.catalystfund.org",
                "application_process": "Quarterly application cycles",
                "portfolio": ["Tala", "Branch", "Lendable"]
            },
            {
                "name": "AfDB Innovation Challenge",
                "type": "Grant",
                "country": "Continental",
                "focus_sectors": ["Agritech", "Clean Energy", "Health", "Education"],
                "stage": ["Pre-seed", "Seed"],
                "typical_investment": "$50K - $500K",
                "description": "African Development Bank's innovation fund supporting scalable solutions.",
                "contact": "innovation@afdb.org",
                "website": "https://www.afdb.org/innovation-challenge",
                "application_process": "Annual competition with multiple phases",
                "portfolio": ["SolarNow", "Farmerline", "mPharma"]
            },
            {
                "name": "Launch Africa Ventures",
                "type": "VC Firm",
                "country": "South Africa",
                "focus_sectors": ["Fintech", "Marketplace", "SaaS"],
                "stage": ["Pre-seed", "Seed"],
                "typical_investment": "$25K - $250K",
                "description": "Early-stage VC fund investing in tech-enabled startups across Africa.",
                "contact": "hello@launchafricaventures.com",
                "website": "https://www.launchafricaventures.com",
                "application_process": "Rolling applications via website",
                "portfolio": ["Luno", "Howler", "Strove"]
            },
            {
                "name": "Egypt Ventures",
                "type": "VC Firm",
                "country": "Egypt",
                "focus_sectors": ["Fintech", "E-commerce", "Transportation"],
                "stage": ["Seed", "Series A"],
                "typical_investment": "$100K - $2M",
                "description": "Leading Egyptian VC firm focused on MENA tech startups.",
                "contact": "info@egyptventures.com",
                "website": "https://egyptventures.com",
                "application_process": "Direct approach via email",
                "portfolio": ["Swvl", "Fawry", "MoneyFellows"]
            },
            {
                "name": "Flat6Labs",
                "type": "Accelerator",
                "country": "Egypt",
                "focus_sectors": ["Fintech", "E-commerce", "IoT", "Enterprise"],
                "stage": ["Pre-seed"],
                "typical_investment": "$50K - $100K",
                "description": "MENA's leading seed and early stage venture capital firm.",
                "contact": "cairo@flat6labs.com",
                "website": "https://www.flat6labs.com",
                "application_process": "4-month accelerator program",
                "portfolio": ["Instabug", "Aqarmap", "Eventtus"]
            },
            {
                "name": "GSMA Innovation Fund",
                "type": "Grant",
                "country": "Global",
                "focus_sectors": ["Mobile Technology", "Digital Inclusion"],
                "stage": ["Seed", "Growth"],
                "typical_investment": "$50K - $750K",
                "description": "Supporting mobile innovation for underserved populations.",
                "contact": "innovationfund@gsma.com",
                "website": "https://www.gsma.com/mobilefordevelopment/innovation-fund/",
                "application_process": "Bi-annual application rounds",
                "portfolio": ["M-Kopa", "Tala", "Off-Grid Electric"]
            },
            {
                "name": "Accion Venture Lab",
                "type": "Accelerator",
                "country": "Global (Africa Focus)",
                "focus_sectors": ["Fintech"],
                "stage": ["Seed", "Series A"],
                "typical_investment": "$100K - $2M",
                "description": "Early-stage investor in fintech startups serving underserved populations.",
                "contact": "venturelab@accion.org",
                "website": "https://www.accion.org/venturelab",
                "application_process": "Rolling applications",
                "portfolio": ["Tala", "Branch", "Oraan"]
            }
        ]
    
    def _load_vc_data(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "4DX Ventures",
                "location": "South Africa",
                "aum": "$50M",
                "focus": ["B2B SaaS", "Fintech", "Logistics"],
                "stage": ["Seed", "Series A"],
                "notable_investments": ["Aerobotics", "DataProphet", "FlexClub"]
            },
            {
                "name": "Knife Capital",
                "location": "South Africa",
                "aum": "$100M",
                "focus": ["Tech-enabled services", "SaaS", "Fintech"],
                "stage": ["Seed", "Series A"],
                "notable_investments": ["Aerobotics", "WhereIsMyTransport", "Custos"]
            },
            {
                "name": "Novastar Ventures",
                "location": "Kenya",
                "aum": "$200M",
                "focus": ["Financial Services", "Healthcare", "Agriculture", "Education"],
                "stage": ["Series A", "Series B"],
                "notable_investments": ["Tala", "Inclusive Health", "Apollo Agriculture"]
            },
            {
                "name": "CcHUB Growth Capital",
                "location": "Nigeria",
                "aum": "$60M",
                "focus": ["Fintech", "Health", "Education", "Agriculture"],
                "stage": ["Seed", "Series A"],
                "notable_investments": ["BudgIT", "Lifebank", "54gene"]
            }
        ]
    
    def _load_accelerator_data(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "Techstars",
                "location": "Multiple (including Cape Town)",
                "program_length": "3 months",
                "investment": "$20K + $100K convertible",
                "focus": ["Tech startups"],
                "application_deadline": "Rolling",
                "success_rate": "1-2%"
            },
            {
                "name": "500 Startups",
                "location": "Multiple (Africa program)",
                "program_length": "4 months",
                "investment": "$150K",
                "focus": ["Tech startups with global potential"],
                "application_deadline": "Bi-annual",
                "success_rate": "2-3%"
            },
            {
                "name": "Startupbootcamp AfriTech",
                "location": "Cape Town",
                "program_length": "3 months",
                "investment": "€15K + €100K follow-on",
                "focus": ["Fintech", "Insurtech", "Regtech"],
                "application_deadline": "Annual",
                "success_rate": "1%"
            },
            {
                "name": "Growth Africa",
                "location": "Kenya, Tanzania, Uganda",
                "program_length": "6 months",
                "investment": "$25K - $75K",
                "focus": ["SME growth businesses"],
                "application_deadline": "Quarterly",
                "success_rate": "5%"
            }
        ]
    
    def _load_government_programs(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "Nigeria Sovereign Investment Authority (NSIA)",
                "country": "Nigeria",
                "type": "Government Fund",
                "focus": ["Infrastructure", "Agriculture", "Healthcare", "Technology"],
                "funding_range": "$1M - $50M",
                "eligibility": "Nigerian companies with significant local impact"
            },
            {
                "name": "Kenya Climate Innovation Center (KCIC)",
                "country": "Kenya",
                "type": "Government Initiative",
                "focus": ["Clean Technology", "Climate Adaptation"],
                "funding_range": "$25K - $500K",
                "eligibility": "Early-stage climate tech ventures"
            },
            {
                "name": "South African SME Fund",
                "country": "South Africa",
                "type": "Government Program",
                "focus": ["Manufacturing", "Technology", "Agriculture"],
                "funding_range": "$50K - $2M",
                "eligibility": "South African SMEs with growth potential"
            },
            {
                "name": "Ghana Venture Capital Trust Fund",
                "country": "Ghana",
                "type": "Government Fund",
                "focus": ["SME Development", "Technology"],
                "funding_range": "$100K - $1M",
                "eligibility": "Ghanaian SMEs and startups"
            },
            {
                "name": "Rwanda Development Board Innovation Fund",
                "country": "Rwanda",
                "type": "Government Initiative",
                "focus": ["ICT", "Manufacturing", "Agriculture"],
                "funding_range": "$25K - $250K",
                "eligibility": "Registered Rwandan companies"
            }
        ]
    
    def _load_grants_data(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "Bill & Melinda Gates Foundation",
                "type": "International Grant",
                "focus": ["Health", "Agriculture", "Financial Services for the Poor"],
                "amount_range": "$100K - $10M",
                "application_cycle": "Continuous",
                "eligibility": "Organizations addressing global challenges"
            },
            {
                "name": "Mastercard Foundation Young Africa Works",
                "type": "Development Grant",
                "focus": ["Youth Employment", "Skills Development", "Entrepreneurship"],
                "amount_range": "$500K - $5M",
                "application_cycle": "Annual",
                "eligibility": "Organizations focusing on youth in Africa"
            },
            {
                "name": "USAID Development Innovation Ventures",
                "type": "Government Grant",
                "focus": ["Global Development Solutions"],
                "amount_range": "$25K - $1.5M",
                "application_cycle": "Continuous",
                "eligibility": "Evidence-based development solutions"
            },
            {
                "name": "World Bank Innovation Lab",
                "type": "International Grant",
                "focus": ["Digital Development", "Climate", "Pandemic Preparedness"],
                "amount_range": "$50K - $2M",
                "application_cycle": "Bi-annual",
                "eligibility": "Innovative solutions for development challenges"
            }
        ]
    
    def search_funding(self, 
                      country: str = None,
                      sector: str = None,
                      stage: str = None,
                      funding_type: str = None) -> List[Dict[str, Any]]:
        
        results = []
        
        for opportunity in self.funding_opportunities:
            match = True
            
            if country and opportunity.get('country') != country:
                if opportunity.get('country') != 'Global' and opportunity.get('country') != 'Continental':
                    match = False
            
            if sector and sector not in opportunity.get('focus_sectors', []):
                match = False
            
            if stage and stage not in opportunity.get('stage', []):
                match = False
            
            if funding_type and opportunity.get('type') != funding_type:
                match = False
            
            if match:
                results.append(opportunity)
        
        return results
    
    def get_funding_by_type(self, funding_type: str) -> List[Dict[str, Any]]:
        return [opp for opp in self.funding_opportunities if opp.get('type') == funding_type]
    
    def get_country_specific_funding(self, country: str) -> Dict[str, Any]:
        country_funding = {
            'vc_firms': [],
            'accelerators': [],
            'government_programs': [],
            'grants': []
        }
        
        # VC firms
        country_funding['vc_firms'] = [
            vc for vc in self.vc_firms 
            if vc.get('location') == country or 'Multiple' in vc.get('location', '')
        ]
        
        # Accelerators
        country_funding['accelerators'] = [
            acc for acc in self.accelerators
            if country in acc.get('location', '') or 'Multiple' in acc.get('location', '')
        ]
        
        # Government programs
        country_funding['government_programs'] = [
            prog for prog in self.government_programs
            if prog.get('country') == country
        ]
        
        # International grants (available to all countries)
        country_funding['grants'] = self.grants
        
        return country_funding
    
    def generate_funding_report(self, filters: Dict[str, Any] = None) -> str:
        filters = filters or {}
        
        matching_opportunities = self.search_funding(**filters)
        
        report = f"# African Funding Landscape Report\n\n"
        
        if filters:
            report += f"## Search Criteria\n"
            for key, value in filters.items():
                report += f"- **{key.title()}**: {value}\n"
            report += "\n"
        
        report += f"## Summary\n"
        report += f"Found **{len(matching_opportunities)}** matching funding opportunities.\n\n"
        
        # Group by type
        by_type = {}
        for opp in matching_opportunities:
            opp_type = opp.get('type', 'Other')
            if opp_type not in by_type:
                by_type[opp_type] = []
            by_type[opp_type].append(opp)
        
        for opp_type, opportunities in by_type.items():
            report += f"## {opp_type}s ({len(opportunities)})\n\n"
            
            for opp in opportunities:
                report += f"### {opp['name']}\n"
                report += f"- **Country**: {opp.get('country', 'N/A')}\n"
                report += f"- **Focus Sectors**: {', '.join(opp.get('focus_sectors', []))}\n"
                report += f"- **Stage**: {', '.join(opp.get('stage', []))}\n"
                report += f"- **Typical Investment**: {opp.get('typical_investment', 'N/A')}\n"
                report += f"- **Description**: {opp.get('description', '')}\n"
                
                if opp.get('website'):
                    report += f"- **Website**: {opp['website']}\n"
                
                if opp.get('application_process'):
                    report += f"- **Application Process**: {opp['application_process']}\n"
                
                report += "\n"
        
        return report

funding_db = FundingDatabase()