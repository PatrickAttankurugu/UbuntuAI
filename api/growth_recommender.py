import openai
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import json
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from config.settings import settings
from api.memory_chains import FeedbackMemoryChain
from api.hybrid_retrieval import HybridRetriever

@dataclass
class GrowthRecommendation:
    strategy_type: str
    priority_score: float
    implementation_timeline: str
    expected_impact: Dict[str, float]
    required_resources: Dict[str, Any]
    success_metrics: List[str]
    action_items: List[Dict[str, Any]]
    risk_assessment: Dict[str, float]
    similar_success_cases: List[Dict[str, Any]]
    confidence_level: float

@dataclass
class GrowthInsight:
    insight_type: str
    description: str
    impact_level: str
    urgency: str
    data_source: str
    recommendation: str

class GrowthRecommenderSystem:
    """
    AI-powered growth strategy recommender for SIGMA platform
    Analyzes business performance and recommends personalized growth strategies
    """
    
    def __init__(self):
        self.memory_chain = FeedbackMemoryChain()
        self.retriever = HybridRetriever()
        
        # Growth strategy frameworks
        self.growth_frameworks = {
            "customer_acquisition": self._load_acquisition_strategies(),
            "product_development": self._load_product_strategies(),
            "market_expansion": self._load_expansion_strategies(),
            "operational_efficiency": self._load_efficiency_strategies(),
            "revenue_optimization": self._load_revenue_strategies(),
            "partnership_development": self._load_partnership_strategies()
        }
        
        # African market growth patterns
        self.african_growth_patterns = self._load_african_growth_patterns()
        
        # ML models for growth prediction
        self.growth_predictor = None
        self.churn_predictor = None
        self._initialize_ml_models()

    def generate_growth_recommendations(self, 
                                      business_data: Dict[str, Any],
                                      performance_metrics: Dict[str, Any],
                                      market_context: Dict[str, Any] = None) -> List[GrowthRecommendation]:
        """
        Generate personalized growth recommendations based on business analysis
        """
        
        # Step 1: Analyze current business performance
        performance_analysis = self._analyze_business_performance(
            business_data, performance_metrics
        )
        
        # Step 2: Identify growth opportunities
        opportunities = self._identify_growth_opportunities(
            performance_analysis, market_context
        )
        
        # Step 3: Generate strategy recommendations
        recommendations = []
        for opportunity in opportunities:
            recommendation = self._generate_strategy_recommendation(
                opportunity, business_data, performance_analysis
            )
            recommendations.append(recommendation)
        
        # Step 4: Prioritize and validate recommendations
        prioritized_recommendations = self._prioritize_recommendations(
            recommendations, business_data
        )
        
        # Step 5: Add implementation guidance
        final_recommendations = []
        for rec in prioritized_recommendations:
            enhanced_rec = self._add_implementation_guidance(rec, business_data)
            final_recommendations.append(enhanced_rec)
        
        return final_recommendations

    def _analyze_business_performance(self, 
                                    business_data: Dict[str, Any],
                                    metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive business performance analysis"""
        
        analysis_prompt = f"""
        Analyze this business performance for growth opportunities:
        
        Business Data:
        {json.dumps(business_data, indent=2)}
        
        Performance Metrics:
        {json.dumps(metrics, indent=2)}
        
        African Growth Patterns:
        {json.dumps(self.african_growth_patterns, indent=2)}
        
        Provide analysis in JSON format:
        {{
            "performance_summary": {{
                "overall_health": "excellent/good/average/poor",
                "growth_stage": "early/growth/maturity/decline",
                "key_strengths": ["strength1", "strength2"],
                "key_weaknesses": ["weakness1", "weakness2"]
            }},
            "growth_potential": {{
                "customer_acquisition": 0.85,
                "revenue_growth": 0.70,
                "market_expansion": 0.60,
                "operational_scaling": 0.75
            }},
            "bottlenecks": [
                {{
                    "area": "customer_acquisition",
                    "severity": "high",
                    "description": "Limited marketing channels"
                }}
            ],
            "opportunities": [
                {{
                    "type": "market_expansion",
                    "potential_impact": 0.80,
                    "feasibility": 0.70,
                    "description": "Expand to rural markets"
                }}
            ],
            "risk_factors": [
                {{
                    "risk": "customer_churn",
                    "probability": 0.30,
                    "impact": 0.70,
                    "mitigation": "Improve customer retention"
                }}
            ]
        }}
        """
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": analysis_prompt}],
                temperature=0.2,
                max_tokens=1500
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            return self._fallback_performance_analysis(business_data, metrics)

    def _identify_growth_opportunities(self, 
                                     performance_analysis: Dict[str, Any],
                                     market_context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Identify specific growth opportunities using AI and data analysis"""
        
        opportunities = []
        
        # Extract opportunities from performance analysis
        if 'opportunities' in performance_analysis:
            opportunities.extend(performance_analysis['opportunities'])
        
        # Add ML-based opportunity detection
        ml_opportunities = self._detect_ml_opportunities(performance_analysis)
        opportunities.extend(ml_opportunities)
        
        # Add market-based opportunities
        if market_context:
            market_opportunities = self._detect_market_opportunities(
                performance_analysis, market_context
            )
            opportunities.extend(market_opportunities)
        
        # Add pattern-based opportunities from African data
        pattern_opportunities = self._detect_pattern_opportunities(performance_analysis)
        opportunities.extend(pattern_opportunities)
        
        return opportunities

    def _generate_strategy_recommendation(self, 
                                        opportunity: Dict[str, Any],
                                        business_data: Dict[str, Any],
                                        performance_analysis: Dict[str, Any]) -> GrowthRecommendation:
        """Generate specific strategy recommendation for an opportunity"""
        
        opportunity_type = opportunity.get('type', 'general')
        strategies = self.growth_frameworks.get(opportunity_type, {})
        
        strategy_prompt = f"""
        Generate a detailed growth strategy recommendation:
        
        Opportunity:
        {json.dumps(opportunity, indent=2)}
        
        Business Context:
        {json.dumps(business_data, indent=2)}
        
        Performance Analysis:
        {json.dumps(performance_analysis, indent=2)}
        
        Available Strategies:
        {json.dumps(strategies, indent=2)}
        
        Provide recommendation in JSON format:
        {{
            "strategy_type": "{opportunity_type}",
            "priority_score": 0.85,
            "implementation_timeline": "3-6 months",
            "expected_impact": {{
                "revenue_increase": 0.25,
                "customer_growth": 0.40,
                "market_share": 0.15
            }},
            "required_resources": {{
                "budget": 50000,
                "team_size": 3,
                "timeline_weeks": 12,
                "skills_needed": ["marketing", "product_development"]
            }},
            "success_metrics": [
                "customer_acquisition_rate",
                "revenue_per_customer",
                "market_penetration"
            ],
            "action_items": [
                {{
                    "task": "Market research in target segment",
                    "priority": "high",
                    "timeline": "2 weeks",
                    "owner": "marketing_team",
                    "resources_needed": ["budget", "research_tools"]
                }}
            ],
            "risk_assessment": {{
                "execution_risk": 0.30,
                "market_risk": 0.25,
                "financial_risk": 0.20,
                "competitive_risk": 0.35
            }},
            "similar_success_cases": [
                {{
                    "company": "Similar African startup",
                    "strategy": "Strategy description",
                    "outcome": "Results achieved",
                    "lessons": "Key learnings"
                }}
            ],
            "confidence_level": 0.75
        }}
        """
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": strategy_prompt}],
                temperature=0.3,
                max_tokens=2000
            )
            
            recommendation_data = json.loads(response.choices[0].message.content)
            return GrowthRecommendation(**recommendation_data)
            
        except Exception as e:
            return self._fallback_strategy_recommendation(opportunity, business_data)

    def _prioritize_recommendations(self, 
                                  recommendations: List[GrowthRecommendation],
                                  business_data: Dict[str, Any]) -> List[GrowthRecommendation]:
        """Prioritize recommendations based on impact, feasibility, and resources"""
        
        def calculate_priority_score(rec: GrowthRecommendation) -> float:
            # Extract business constraints
            budget_constraint = business_data.get('budget_available', 100000)
            team_constraint = business_data.get('team_size', 5)
            urgency = business_data.get('growth_urgency', 0.5)
            
            # Calculate impact score (weighted average of expected impacts)
            impact_score = np.mean(list(rec.expected_impact.values()))
            
            # Calculate feasibility score based on resources
            budget_feasibility = min(1.0, budget_constraint / rec.required_resources.get('budget', 1))
            team_feasibility = min(1.0, team_constraint / rec.required_resources.get('team_size', 1))
            feasibility_score = (budget_feasibility + team_feasibility) / 2
            
            # Calculate risk-adjusted score
            avg_risk = np.mean(list(rec.risk_assessment.values()))
            risk_adjusted_score = 1 - avg_risk
            
            # Combine scores with weights
            priority_score = (
                impact_score * 0.4 +
                feasibility_score * 0.3 +
                risk_adjusted_score * 0.2 +
                rec.confidence_level * 0.1
            )
            
            # Apply urgency multiplier
            priority_score *= (1 + urgency * 0.2)
            
            return priority_score
        
        # Calculate priority scores
        for rec in recommendations:
            rec.priority_score = calculate_priority_score(rec)
        
        # Sort by priority score
        return sorted(recommendations, key=lambda x: x.priority_score, reverse=True)

    def _add_implementation_guidance(self, 
                                   recommendation: GrowthRecommendation,
                                   business_data: Dict[str, Any]) -> GrowthRecommendation:
        """Add detailed implementation guidance to recommendation"""
        
        # Add location-specific guidance
        location = business_data.get('location', 'Ghana')
        location_guidance = self._get_location_specific_guidance(
            recommendation.strategy_type, location
        )
        
        # Add resource-specific adjustments
        available_resources = business_data.get('available_resources', {})
        resource_adjustments = self._adjust_for_available_resources(
            recommendation, available_resources
        )
        
        # Enhance action items with more detail
        enhanced_action_items = []
        for item in recommendation.action_items:
            enhanced_item = self._enhance_action_item(item, business_data)
            enhanced_action_items.append(enhanced_item)
        
        recommendation.action_items = enhanced_action_items
        
        return recommendation

    def generate_growth_insights(self, 
                               business_data: Dict[str, Any],
                               time_series_data: Dict[str, List[float]]) -> List[GrowthInsight]:
        """Generate actionable growth insights from business data"""
        
        insights = []
        
        # Trend analysis insights
        trend_insights = self._analyze_growth_trends(time_series_data)
        insights.extend(trend_insights)
        
        # Cohort analysis insights
        if 'customer_cohorts' in time_series_data:
            cohort_insights = self._analyze_customer_cohorts(time_series_data['customer_cohorts'])
            insights.extend(cohort_insights)
        
        # Benchmark insights
        benchmark_insights = self._generate_benchmark_insights(business_data)
        insights.extend(benchmark_insights)
        
        # AI-generated insights
        ai_insights = self._generate_ai_insights(business_data, time_series_data)
        insights.extend(ai_insights)
        
        return insights

    def _analyze_growth_trends(self, time_series_data: Dict[str, List[float]]) -> List[GrowthInsight]:
        """Analyze trends in business metrics"""
        insights = []
        
        for metric, values in time_series_data.items():
            if len(values) < 3:
                continue
                
            # Calculate trend
            x = np.arange(len(values))
            y = np.array(values)
            slope = np.polyfit(x, y, 1)[0]
            
            # Generate insight based on trend
            if slope > 0.1:
                insights.append(GrowthInsight(
                    insight_type="positive_trend",
                    description=f"{metric.replace('_', ' ').title()} showing strong upward trend",
                    impact_level="high",
                    urgency="medium",
                    data_source="trend_analysis",
                    recommendation=f"Capitalize on {metric} momentum with increased investment"
                ))
            elif slope < -0.1:
                insights.append(GrowthInsight(
                    insight_type="negative_trend",
                    description=f"{metric.replace('_', ' ').title()} showing declining trend",
                    impact_level="high",
                    urgency="high",
                    data_source="trend_analysis",
                    recommendation=f"Immediate action needed to reverse {metric} decline"
                ))
        
        return insights

    def _detect_ml_opportunities(self, performance_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Use ML models to detect growth opportunities"""
        opportunities = []
        
        # Extract features for ML analysis
        features = self._extract_ml_features(performance_analysis)
        
        if self.growth_predictor and len(features) > 0:
            try:
                # Predict growth potential in different areas
                growth_predictions = self.growth_predictor.predict([features])[0]
                
                # Convert predictions to opportunities
                opportunity_areas = [
                    "customer_acquisition", "product_development", 
                    "market_expansion", "operational_efficiency"
                ]
                
                for i, area in enumerate(opportunity_areas):
                    if growth_predictions[i] > 0.7:  # High growth potential
                        opportunities.append({
                            "type": area,
                            "potential_impact": growth_predictions[i],
                            "feasibility": 0.75,  # Default feasibility
                            "description": f"ML model indicates high potential in {area}",
                            "source": "ml_prediction"
                        })
            except Exception as e:
                pass  # ML prediction failed, continue without it
        
        return opportunities

    def _detect_pattern_opportunities(self, performance_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect opportunities based on African growth patterns"""
        opportunities = []
        
        # Pattern matching against successful African companies
        for pattern in self.african_growth_patterns.get('success_patterns', []):
            pattern_match_score = self._calculate_pattern_match(
                performance_analysis, pattern
            )
            
            if pattern_match_score > 0.6:
                opportunities.append({
                    "type": pattern.get('strategy_type', 'general'),
                    "potential_impact": pattern_match_score,
                    "feasibility": pattern.get('feasibility', 0.7),
                    "description": f"Pattern match with {pattern.get('description', 'successful strategy')}",
                    "source": "pattern_analysis"
                })
        
        return opportunities

    def _load_acquisition_strategies(self) -> Dict[str, Any]:
        """Load customer acquisition strategies for African markets"""
        return {
            "digital_marketing": {
                "channels": ["social_media", "search_engine", "content_marketing"],
                "african_focus": ["whatsapp_marketing", "facebook_groups", "radio_partnerships"],
                "cost_effectiveness": 0.75,
                "scalability": 0.85
            },
            "referral_programs": {
                "mechanisms": ["cash_rewards", "service_credits", "social_recognition"],
                "african_adaptations": ["community_rewards", "mobile_money_incentives"],
                "cost_effectiveness": 0.90,
                "scalability": 0.80
            },
            "partnership_channels": {
                "types": ["strategic_partnerships", "channel_partners", "influencer_collaborations"],
                "african_focus": ["local_business_partnerships", "community_leader_endorsements"],
                "cost_effectiveness": 0.80,
                "scalability": 0.70
            }
        }

    def _load_african_growth_patterns(self) -> Dict[str, Any]:
        """Load proven growth patterns from African startups"""
        return {
            "success_patterns": [
                {
                    "strategy_type": "mobile_first",
                    "description": "Mobile-first approach with offline capability",
                    "success_rate": 0.85,
                    "feasibility": 0.90,
                    "characteristics": ["mobile_app", "offline_sync", "ussd_backup"]
                },
                {
                    "strategy_type": "community_based",
                    "description": "Community-driven growth and distribution",
                    "success_rate": 0.75,
                    "feasibility": 0.80,
                    "characteristics": ["local_ambassadors", "word_of_mouth", "trust_networks"]
                },
                {
                    "strategy_type": "financial_inclusion",
                    "description": "Integration with mobile money and informal finance",
                    "success_rate": 0.80,
                    "feasibility": 0.85,
                    "characteristics": ["mobile_money", "micro_transactions", "informal_credit"]
                }
            ],
            "growth_stages": {
                "early": {
                    "focus": ["product_market_fit", "initial_traction"],
                    "key_metrics": ["user_acquisition", "engagement"],
                    "typical_challenges": ["funding", "team_building", "market_validation"]
                },
                "growth": {
                    "focus": ["scaling", "market_expansion"],
                    "key_metrics": ["revenue_growth", "market_share"],
                    "typical_challenges": ["operational_scaling", "competition", "talent_acquisition"]
                },
                "maturity": {
                    "focus": ["optimization", "diversification"],
                    "key_metrics": ["profitability", "efficiency"],
                    "typical_challenges": ["innovation", "market_saturation", "regulation"]
                }
            }
        }

    def _initialize_ml_models(self):
        """Initialize ML models for growth prediction"""
        # Note: In production, these would be pre-trained models
        # For this demo, we'll create placeholder models
        try:
            from sklearn.ensemble import RandomForestRegressor
            self.growth_predictor = RandomForestRegressor(n_estimators=100, random_state=42)
            self.churn_predictor = RandomForestRegressor(n_estimators=100, random_state=42)
            
            # In production, load pre-trained models here
            # self.growth_predictor = joblib.load('models/growth_predictor.pkl')
            # self.churn_predictor = joblib.load('models/churn_predictor.pkl')
            
        except ImportError:
            self.growth_predictor = None
            self.churn_predictor = None

    def _extract_ml_features(self, performance_analysis: Dict[str, Any]) -> List[float]:
        """Extract features for ML models"""
        features = []
        
        # Extract numerical features from performance analysis
        growth_potential = performance_analysis.get('growth_potential', {})
        features.extend([
            growth_potential.get('customer_acquisition', 0.5),
            growth_potential.get('revenue_growth', 0.5),
            growth_potential.get('market_expansion', 0.5),
            growth_potential.get('operational_scaling', 0.5)
        ])
        
        # Add more features as needed
        return features

    def _fallback_performance_analysis(self, business_data: Dict[str, Any], 
                                     metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback analysis when AI fails"""
        return {
            "performance_summary": {
                "overall_health": "good",
                "growth_stage": "growth",
                "key_strengths": ["local_market_knowledge", "mobile_optimization"],
                "key_weaknesses": ["limited_funding", "small_team"]
            },
            "growth_potential": {
                "customer_acquisition": 0.70,
                "revenue_growth": 0.65,
                "market_expansion": 0.60,
                "operational_scaling": 0.55
            },
            "opportunities": [
                {
                    "type": "customer_acquisition",
                    "potential_impact": 0.75,
                    "feasibility": 0.80,
                    "description": "Expand digital marketing efforts"
                }
            ]
        }

    def _fallback_strategy_recommendation(self, opportunity: Dict[str, Any],
                                        business_data: Dict[str, Any]) -> GrowthRecommendation:
        """Fallback recommendation when AI fails"""
        return GrowthRecommendation(
            strategy_type=opportunity.get('type', 'general'),
            priority_score=0.70,
            implementation_timeline="3-6 months",
            expected_impact={"revenue_increase": 0.20, "customer_growth": 0.30},
            required_resources={"budget": 25000, "team_size": 2, "timeline_weeks": 12},
            success_metrics=["customer_growth_rate", "revenue_per_customer"],
            action_items=[{
                "task": "Develop implementation plan",
                "priority": "high",
                "timeline": "1 week",
                "owner": "team_lead"
            }],
            risk_assessment={"execution_risk": 0.30, "market_risk": 0.25},
            similar_success_cases=[],
            confidence_level=0.60
        )

# Factory function
def create_growth_recommender():
    return GrowthRecommenderSystem()