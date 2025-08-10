import openai
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import json
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from config.settings import settings

@dataclass
class ImpactMetric:
    metric_id: str
    name: str
    description: str
    category: str
    measurement_unit: str
    target_value: float
    current_value: float
    baseline_value: float
    progress_percentage: float
    trend: str
    data_collection_method: str
    frequency: str
    last_updated: str

@dataclass
class ImpactAssessment:
    assessment_id: str
    business_id: str
    assessment_date: str
    overall_impact_score: float
    sdg_alignment: Dict[str, float]
    impact_categories: Dict[str, float]
    key_achievements: List[str]
    areas_for_improvement: List[str]
    recommendations: List[str]
    social_return_on_investment: float
    beneficiary_feedback: Dict[str, Any]
    impact_metrics: List[ImpactMetric]

@dataclass
class LearningInsight:
    insight_id: str
    insight_type: str
    description: str
    evidence: List[str]
    implications: List[str]
    recommended_actions: List[str]
    confidence_level: float
    applicable_contexts: List[str]

class MERLAnalyticsEngine:
    """
    Monitoring, Evaluation, Reporting, and Learning (MERL) Analytics Engine
    Designed for impact measurement of social enterprises and development programs
    Optimized for emerging markets with limited data infrastructure
    """
    
    def __init__(self):
        # Impact measurement frameworks
        self.impact_frameworks = {
            "theory_of_change": self._load_theory_of_change_framework(),
            "sdg_mapping": self._load_sdg_framework(),
            "sroi": self._load_sroi_framework(),
            "lean_data": self._load_lean_data_framework()
        }
        
        # Beneficiary categorization
        self.beneficiary_categories = {
            "direct": {"weight": 1.0, "measurement_approach": "direct_survey"},
            "indirect": {"weight": 0.6, "measurement_approach": "proxy_indicators"},
            "broader_community": {"weight": 0.3, "measurement_approach": "community_level_data"}
        }
        
        # African development priorities alignment
        self.african_priorities = self._load_african_development_priorities()
        
        # Data collection methods for low-resource environments
        self.data_collection_methods = self._load_data_collection_methods()

    def create_impact_measurement_framework(self, 
                                          business_data: Dict[str, Any],
                                          theory_of_change: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Create comprehensive impact measurement framework for a business
        """
        
        # Step 1: Analyze business model for impact potential
        impact_analysis = self._analyze_impact_potential(business_data)
        
        # Step 2: Map to SDGs and development priorities
        sdg_mapping = self._map_to_sdgs(business_data, impact_analysis)
        
        # Step 3: Define impact metrics and KPIs
        impact_metrics = self._define_impact_metrics(
            business_data, impact_analysis, sdg_mapping
        )
        
        # Step 4: Design data collection strategy
        data_collection_strategy = self._design_data_collection_strategy(
            impact_metrics, business_data
        )
        
        # Step 5: Create monitoring and evaluation plan
        me_plan = self._create_monitoring_evaluation_plan(
            impact_metrics, data_collection_strategy
        )
        
        # Step 6: Design learning framework
        learning_framework = self._design_learning_framework(business_data)
        
        return {
            "framework_id": f"merl_{business_data.get('id', 'unknown')}_{datetime.now().strftime('%Y%m%d')}",
            "business_context": impact_analysis,
            "sdg_alignment": sdg_mapping,
            "impact_metrics": impact_metrics,
            "data_collection_strategy": data_collection_strategy,
            "monitoring_evaluation_plan": me_plan,
            "learning_framework": learning_framework,
            "reporting_schedule": self._create_reporting_schedule(impact_metrics),
            "implementation_guidance": self._create_implementation_guidance(business_data)
        }

    def conduct_impact_assessment(self, 
                                business_id: str,
                                assessment_data: Dict[str, Any],
                                historical_data: List[Dict[str, Any]] = None) -> ImpactAssessment:
        """
        Conduct comprehensive impact assessment for a business
        """
        
        # Step 1: Process and validate assessment data
        processed_data = self._process_assessment_data(assessment_data)
        
        # Step 2: Calculate impact metrics
        impact_metrics = self._calculate_impact_metrics(processed_data, historical_data)
        
        # Step 3: Assess SDG alignment and progress
        sdg_assessment = self._assess_sdg_progress(impact_metrics, processed_data)
        
        # Step 4: Calculate overall impact score
        overall_score = self._calculate_overall_impact_score(impact_metrics, sdg_assessment)
        
        # Step 5: Calculate SROI
        sroi = self._calculate_social_return_on_investment(processed_data, impact_metrics)
        
        # Step 6: Generate insights and recommendations
        insights = self._generate_impact_insights(impact_metrics, processed_data)
        
        # Step 7: Process beneficiary feedback
        beneficiary_analysis = self._analyze_beneficiary_feedback(
            processed_data.get('beneficiary_feedback', {})
        )
        
        return ImpactAssessment(
            assessment_id=f"impact_{business_id}_{datetime.now().strftime('%Y%m%d')}",
            business_id=business_id,
            assessment_date=datetime.now().isoformat(),
            overall_impact_score=overall_score,
            sdg_alignment=sdg_assessment,
            impact_categories=self._categorize_impact_scores(impact_metrics),
            key_achievements=insights.get('achievements', []),
            areas_for_improvement=insights.get('improvements', []),
            recommendations=insights.get('recommendations', []),
            social_return_on_investment=sroi,
            beneficiary_feedback=beneficiary_analysis,
            impact_metrics=impact_metrics
        )

    def generate_learning_insights(self, 
                                 assessments: List[ImpactAssessment],
                                 context_filters: Dict[str, Any] = None) -> List[LearningInsight]:
        """
        Generate learning insights from multiple impact assessments
        Cross-portfolio analysis for pattern identification
        """
        
        # Step 1: Aggregate and analyze assessment data
        aggregated_data = self._aggregate_assessment_data(assessments)
        
        # Step 2: Identify patterns and trends
        patterns = self._identify_impact_patterns(aggregated_data, context_filters)
        
        # Step 3: Comparative analysis
        comparative_insights = self._conduct_comparative_analysis(assessments)
        
        # Step 4: Success factor analysis
        success_factors = self._analyze_success_factors(assessments)
        
        # Step 5: Risk and challenge analysis
        risk_insights = self._analyze_risk_patterns(assessments)
        
        # Step 6: Generate actionable insights
        learning_insights = []
        
        # Add pattern-based insights
        for pattern in patterns:
            insight = self._create_learning_insight_from_pattern(pattern)
            learning_insights.append(insight)
        
        # Add comparative insights
        for comp_insight in comparative_insights:
            insight = self._create_learning_insight_from_comparison(comp_insight)
            learning_insights.append(insight)
        
        # Add success factor insights
        for success_factor in success_factors:
            insight = self._create_learning_insight_from_success_factor(success_factor)
            learning_insights.append(insight)
        
        return learning_insights

    def create_impact_dashboard_data(self, 
                                   business_data: Dict[str, Any],
                                   time_period: str = "12m") -> Dict[str, Any]:
        """
        Create dashboard data for impact visualization
        """
        
        # Step 1: Retrieve impact data for time period
        impact_data = self._retrieve_impact_data(business_data['id'], time_period)
        
        # Step 2: Create summary metrics
        summary_metrics = self._create_summary_metrics(impact_data)
        
        # Step 3: Generate trend analysis
        trend_analysis = self._generate_trend_analysis(impact_data)
        
        # Step 4: Create beneficiary analysis
        beneficiary_analysis = self._create_beneficiary_analysis(impact_data)
        
        # Step 5: Generate comparison data
        benchmark_data = self._generate_benchmark_comparison(business_data, impact_data)
        
        # Step 6: Create visualization configurations
        chart_configs = self._create_chart_configurations(
            summary_metrics, trend_analysis, beneficiary_analysis
        )
        
        return {
            "summary_metrics": summary_metrics,
            "trend_analysis": trend_analysis,
            "beneficiary_analysis": beneficiary_analysis,
            "benchmark_comparison": benchmark_data,
            "chart_configurations": chart_configs,
            "last_updated": datetime.now().isoformat(),
            "data_quality_indicators": self._assess_data_quality(impact_data)
        }

    def _analyze_impact_potential(self, business_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze business model for impact potential"""
        
        analysis_prompt = f"""
        Analyze this business for social impact potential:
        
        Business Data:
        {json.dumps(business_data, indent=2)}
        
        African Development Context:
        {json.dumps(self.african_priorities, indent=2)}
        
        Provide analysis in JSON format:
        {{
            "primary_impact_areas": [
                {{
                    "area": "economic_empowerment",
                    "description": "Job creation and income generation",
                    "potential_scale": "high",
                    "measurement_approach": "income_tracking"
                }}
            ],
            "beneficiary_groups": [
                {{
                    "group": "smallholder_farmers",
                    "size_estimate": 1000,
                    "impact_type": "direct",
                    "engagement_method": "platform_usage"
                }}
            ],
            "theory_of_change": {{
                "inputs": ["technology", "training", "capital"],
                "activities": ["platform_operation", "farmer_training"],
                "outputs": ["trained_farmers", "transactions_facilitated"],
                "outcomes": ["increased_income", "improved_productivity"],
                "impact": ["poverty_reduction", "food_security"]
            }},
            "measurement_challenges": [
                "attribution", "data_collection_costs", "baseline_establishment"
            ],
            "measurement_opportunities": [
                "digital_footprint", "transaction_data", "mobile_surveys"
            ]
        }}
        """
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": analysis_prompt}],
                temperature=0.3,
                max_tokens=1500
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            return self._fallback_impact_analysis(business_data)

    def _map_to_sdgs(self, 
                    business_data: Dict[str, Any],
                    impact_analysis: Dict[str, Any]) -> Dict[str, float]:
        """Map business activities to Sustainable Development Goals"""
        
        sdg_mapping = {}
        
        # SDG mapping based on business sector and activities
        sector = business_data.get('sector', '').lower()
        
        if 'agri' in sector or 'food' in sector:
            sdg_mapping.update({
                "SDG_1": 0.8,  # No Poverty
                "SDG_2": 0.9,  # Zero Hunger
                "SDG_8": 0.7,  # Decent Work
                "SDG_12": 0.6  # Responsible Consumption
            })
        
        if 'fintech' in sector or 'financial' in sector:
            sdg_mapping.update({
                "SDG_1": 0.9,  # No Poverty
                "SDG_8": 0.8,  # Decent Work
                "SDG_10": 0.7  # Reduced Inequalities
            })
        
        if 'health' in sector:
            sdg_mapping.update({
                "SDG_3": 0.9,  # Good Health
                "SDG_10": 0.6  # Reduced Inequalities
            })
        
        if 'education' in sector or 'edtech' in sector:
            sdg_mapping.update({
                "SDG_4": 0.9,  # Quality Education
                "SDG_8": 0.6,  # Decent Work
                "SDG_10": 0.7  # Reduced Inequalities
            })
        
        # Gender focus adjustments
        if business_data.get('women_focused', False):
            sdg_mapping["SDG_5"] = 0.8  # Gender Equality
        
        # Rural focus adjustments
        if business_data.get('serves_rural_areas', False):
            sdg_mapping["SDG_1"] = sdg_mapping.get("SDG_1", 0) + 0.1
            sdg_mapping["SDG_10"] = sdg_mapping.get("SDG_10", 0) + 0.1
        
        return sdg_mapping

    def _define_impact_metrics(self, 
                             business_data: Dict[str, Any],
                             impact_analysis: Dict[str, Any],
                             sdg_mapping: Dict[str, float]) -> List[ImpactMetric]:
        """Define specific impact metrics for measurement"""
        
        metrics = []
        
        # Universal metrics for all businesses
        universal_metrics = [
            {
                "name": "Direct Jobs Created",
                "category": "Economic Impact",
                "unit": "number",
                "collection_method": "employment_records",
                "frequency": "quarterly"
            },
            {
                "name": "Beneficiaries Reached",
                "category": "Social Impact", 
                "unit": "number",
                "collection_method": "user_registration_data",
                "frequency": "monthly"
            },
            {
                "name": "Revenue Generated",
                "category": "Economic Impact",
                "unit": "currency",
                "collection_method": "financial_records",
                "frequency": "monthly"
            }
        ]
        
        # Sector-specific metrics
        sector = business_data.get('sector', '').lower()
        
        if 'agri' in sector:
            sector_metrics = [
                {
                    "name": "Farmer Income Increase",
                    "category": "Economic Impact",
                    "unit": "percentage",
                    "collection_method": "farmer_surveys",
                    "frequency": "quarterly"
                },
                {
                    "name": "Crop Yield Improvement",
                    "category": "Productivity Impact",
                    "unit": "percentage",
                    "collection_method": "harvest_data",
                    "frequency": "seasonal"
                }
            ]
            universal_metrics.extend(sector_metrics)
        
        if 'fintech' in sector:
            sector_metrics = [
                {
                    "name": "Financial Inclusion Rate",
                    "category": "Social Impact",
                    "unit": "percentage", 
                    "collection_method": "user_behavior_analysis",
                    "frequency": "quarterly"
                },
                {
                    "name": "Transaction Volume",
                    "category": "Economic Impact",
                    "unit": "currency",
                    "collection_method": "platform_analytics",
                    "frequency": "monthly"
                }
            ]
            universal_metrics.extend(sector_metrics)
        
        # Convert to ImpactMetric objects
        for i, metric_data in enumerate(universal_metrics):
            metric = ImpactMetric(
                metric_id=f"metric_{i+1}",
                name=metric_data["name"],
                description=f"Measures {metric_data['name'].lower()} impact",
                category=metric_data["category"],
                measurement_unit=metric_data["unit"],
                target_value=0.0,  # To be set during framework implementation
                current_value=0.0,
                baseline_value=0.0,
                progress_percentage=0.0,
                trend="stable",
                data_collection_method=metric_data["collection_method"],
                frequency=metric_data["frequency"],
                last_updated=datetime.now().isoformat()
            )
            metrics.append(metric)
        
        return metrics

    def _design_data_collection_strategy(self, 
                                       impact_metrics: List[ImpactMetric],
                                       business_data: Dict[str, Any]) -> Dict[str, Any]:
        """Design data collection strategy optimized for African contexts"""
        
        strategy = {
            "collection_methods": {},
            "technology_stack": [],
            "human_resources": {},
            "budget_estimates": {},
            "implementation_timeline": {},
            "quality_assurance": {}
        }
        
        # Group metrics by collection method
        method_groups = {}
        for metric in impact_metrics:
            method = metric.data_collection_method
            if method not in method_groups:
                method_groups[method] = []
            method_groups[method].append(metric)
        
        # Design collection approach for each method
        for method, metrics in method_groups.items():
            if method == "mobile_surveys":
                strategy["collection_methods"][method] = {
                    "approach": "WhatsApp-based surveys with SMS fallback",
                    "tools": ["KoBo Toolbox", "SurveyCTO", "WhatsApp Business API"],
                    "frequency": "monthly",
                    "sample_size": "representative_sample",
                    "challenges": ["connectivity", "literacy", "language_barriers"],
                    "solutions": ["voice_surveys", "local_language_support", "incentives"]
                }
            
            elif method == "platform_analytics":
                strategy["collection_methods"][method] = {
                    "approach": "Automated data extraction from business systems",
                    "tools": ["Google Analytics", "Mixpanel", "Custom dashboards"],
                    "frequency": "real_time",
                    "data_quality": "high",
                    "privacy_considerations": ["anonymization", "consent", "gdpr_compliance"]
                }
            
            elif method == "farmer_surveys":
                strategy["collection_methods"][method] = {
                    "approach": "Mixed-method approach with digital and in-person collection",
                    "tools": ["Mobile forms", "Community data collectors", "Focus groups"],
                    "frequency": "seasonal",
                    "challenges": ["seasonal_availability", "trust_building", "data_accuracy"],
                    "solutions": ["local_partnerships", "incentive_structures", "verification_methods"]
                }
        
        return strategy

    def _calculate_social_return_on_investment(self, 
                                             assessment_data: Dict[str, Any],
                                             impact_metrics: List[ImpactMetric]) -> float:
        """Calculate Social Return on Investment (SROI)"""
        
        total_investment = assessment_data.get('total_investment', 100000)
        
        # Calculate social value created
        social_value = 0.0
        
        for metric in impact_metrics:
            if metric.category == "Economic Impact":
                # Direct economic value
                if "income" in metric.name.lower():
                    social_value += metric.current_value * 0.8  # 80% attribution
                elif "jobs" in metric.name.lower():
                    # Average annual salary * number of jobs
                    social_value += metric.current_value * 12000  # $1000/month average
            
            elif metric.category == "Social Impact":
                # Social value approximation
                if "beneficiaries" in metric.name.lower():
                    # Estimated value per beneficiary
                    social_value += metric.current_value * 200  # $200 per beneficiary
        
        # Apply African context multiplier (lower cost of living)
        african_adjustment = 1.5
        adjusted_social_value = social_value * african_adjustment
        
        # Calculate SROI
        sroi = adjusted_social_value / total_investment if total_investment > 0 else 0
        
        return round(sroi, 2)

    def _load_african_development_priorities(self) -> Dict[str, Any]:
        """Load African development priorities and frameworks"""
        return {
            "agenda_2063_goals": [
                "prosperous_africa",
                "integrated_continent", 
                "good_governance",
                "peaceful_secure_africa",
                "strong_cultural_identity"
            ],
            "priority_sectors": [
                "agriculture_food_security",
                "human_capital_development",
                "infrastructure_development",
                "financial_inclusion",
                "digital_transformation",
                "climate_resilience"
            ],
            "measurement_frameworks": [
                "agenda_2063_indicators",
                "sdg_indicators",
                "african_peer_review_mechanism",
                "regional_economic_community_frameworks"
            ]
        }

    def _load_data_collection_methods(self) -> Dict[str, Any]:
        """Load data collection methods optimized for African contexts"""
        return {
            "mobile_based": {
                "sms_surveys": {
                    "cost": "low",
                    "reach": "high", 
                    "data_quality": "medium",
                    "literacy_requirement": "low"
                },
                "whatsapp_surveys": {
                    "cost": "low",
                    "reach": "high",
                    "data_quality": "high",
                    "literacy_requirement": "medium"
                },
                "mobile_app_data": {
                    "cost": "very_low",
                    "reach": "medium",
                    "data_quality": "very_high",
                    "real_time": True
                }
            },
            "community_based": {
                "community_data_collectors": {
                    "cost": "medium",
                    "reach": "high",
                    "data_quality": "high",
                    "trust_factor": "very_high"
                },
                "focus_group_discussions": {
                    "cost": "medium",
                    "reach": "low",
                    "data_quality": "very_high",
                    "qualitative_insights": True
                }
            },
            "technology_enabled": {
                "satellite_imagery": {
                    "cost": "high",
                    "reach": "very_high",
                    "data_quality": "high",
                    "use_cases": ["agriculture", "infrastructure"]
                },
                "iot_sensors": {
                    "cost": "high",
                    "reach": "low",
                    "data_quality": "very_high",
                    "real_time": True
                }
            }
        }

    def _fallback_impact_analysis(self, business_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback impact analysis when AI fails"""
        return {
            "primary_impact_areas": [
                {
                    "area": "economic_empowerment",
                    "description": "Job creation and income generation",
                    "potential_scale": "medium",
                    "measurement_approach": "survey_based"
                }
            ],
            "beneficiary_groups": [
                {
                    "group": "local_community",
                    "size_estimate": 500,
                    "impact_type": "direct",
                    "engagement_method": "service_delivery"
                }
            ],
            "theory_of_change": {
                "inputs": ["capital", "technology", "human_resources"],
                "activities": ["service_delivery", "capacity_building"],
                "outputs": ["services_provided", "people_trained"],
                "outcomes": ["improved_livelihoods", "enhanced_capabilities"],
                "impact": ["poverty_reduction", "sustainable_development"]
            },
            "measurement_challenges": ["attribution", "data_availability", "cost"],
            "measurement_opportunities": ["digital_data", "partner_collaboration"]
        }

# Factory function
def create_merl_analytics_engine():
    return MERLAnalyticsEngine()