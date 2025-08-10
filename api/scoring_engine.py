import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import re
from datetime import datetime
import openai
from config.settings import settings

@dataclass
class ScoringResult:
    overall_score: float
    component_scores: Dict[str, float]
    risk_factors: List[str]
    recommendations: List[str]
    confidence: float

class StartupReadinessScorer:
    def __init__(self):
        self.weight_config = {
            'business_model': 0.25,
            'market_readiness': 0.20,
            'team_strength': 0.20,
            'financial_health': 0.15,
            'traction': 0.10,
            'adaptability': 0.10
        }
        
        # Ghana-specific business context
        self.ghana_sectors = {
            'agritech': 0.9,  # High potential in Ghana
            'fintech': 0.85,
            'healthtech': 0.8,
            'edtech': 0.75,
            'logistics': 0.7,
            'ecommerce': 0.65
        }
        
        # Risk indicators specific to Ghanaian market
        self.risk_indicators = {
            'no_local_partnership': 0.8,
            'foreign_only_team': 0.7,
            'no_mobile_strategy': 0.9,  # Critical in Ghana
            'cash_only_model': 0.6,
            'no_local_compliance': 0.8
        }

    def score_startup(self, business_data: Dict[str, Any]) -> ScoringResult:
        """
        Score startup readiness based on business data
        Optimized for Ghanaian/West African context
        """
        
        # Extract and score each component
        component_scores = {
            'business_model': self._score_business_model(business_data),
            'market_readiness': self._score_market_readiness(business_data),
            'team_strength': self._score_team_strength(business_data),
            'financial_health': self._score_financial_health(business_data),
            'traction': self._score_traction(business_data),
            'adaptability': self._score_adaptability(business_data)
        }
        
        # Calculate weighted overall score
        overall_score = sum(
            score * self.weight_config[component] 
            for component, score in component_scores.items()
        )
        
        # Identify risk factors
        risk_factors = self._identify_risk_factors(business_data, component_scores)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(component_scores, business_data)
        
        # Calculate confidence based on data completeness
        confidence = self._calculate_confidence(business_data)
        
        return ScoringResult(
            overall_score=round(overall_score, 2),
            component_scores=component_scores,
            risk_factors=risk_factors,
            recommendations=recommendations,
            confidence=confidence
        )

    def _score_business_model(self, data: Dict[str, Any]) -> float:
        """Score business model viability"""
        score = 0.5  # Base score
        
        # Revenue model clarity
        if data.get('revenue_streams'):
            score += 0.2
            if len(data['revenue_streams']) > 1:
                score += 0.1  # Diversified revenue
        
        # Value proposition
        if data.get('value_proposition'):
            score += 0.15
        
        # Scalability indicators
        if data.get('scalable_model', False):
            score += 0.1
        
        # Ghana-specific adjustments
        sector = data.get('sector', '').lower()
        if sector in self.ghana_sectors:
            score *= self.ghana_sectors[sector]
        
        # Mobile-first approach (critical in Ghana)
        if data.get('mobile_first', False):
            score += 0.05
        
        return min(score, 1.0)

    def _score_market_readiness(self, data: Dict[str, Any]) -> float:
        """Score market readiness and opportunity"""
        score = 0.3  # Base score
        
        # Market research
        if data.get('market_research_done', False):
            score += 0.2
        
        # Target market definition
        if data.get('target_market'):
            score += 0.15
        
        # Competition analysis
        if data.get('competitor_analysis', False):
            score += 0.1
        
        # Local market understanding (crucial for Ghana)
        if data.get('local_market_knowledge', False):
            score += 0.15
        
        # Rural/informal market focus (high potential in Ghana)
        if data.get('serves_rural_market', False):
            score += 0.1
        
        return min(score, 1.0)

    def _score_team_strength(self, data: Dict[str, Any]) -> float:
        """Score team capabilities and composition"""
        score = 0.2  # Base score
        
        # Team size
        team_size = data.get('team_size', 0)
        if 2 <= team_size <= 5:
            score += 0.15
        elif team_size > 5:
            score += 0.1
        
        # Relevant experience
        if data.get('relevant_experience', False):
            score += 0.2
        
        # Technical capabilities
        if data.get('technical_team', False):
            score += 0.15
        
        # Local team presence (important for Ghana operations)
        if data.get('local_team_members', False):
            score += 0.15
        
        # Women-led (bonus for Seedstars focus)
        if data.get('women_led', False):
            score += 0.1
        
        # Previous startup experience
        if data.get('previous_startup_experience', False):
            score += 0.05
        
        return min(score, 1.0)

    def _score_financial_health(self, data: Dict[str, Any]) -> float:
        """Score financial status and planning"""
        score = 0.1  # Base score
        
        # Financial projections
        if data.get('financial_projections', False):
            score += 0.2
        
        # Current funding
        funding_status = data.get('funding_status', 'none')
        if funding_status == 'bootstrapped':
            score += 0.15
        elif funding_status == 'pre_seed':
            score += 0.25
        elif funding_status in ['seed', 'series_a']:
            score += 0.3
        
        # Burn rate understanding
        if data.get('burn_rate_tracked', False):
            score += 0.1
        
        # Revenue generation
        if data.get('generating_revenue', False):
            score += 0.2
        
        # Path to profitability
        if data.get('profitability_plan', False):
            score += 0.15
        
        return min(score, 1.0)

    def _score_traction(self, data: Dict[str, Any]) -> float:
        """Score market traction and validation"""
        score = 0.0  # Base score
        
        # Customer base
        customers = data.get('customer_count', 0)
        if customers > 1000:
            score += 0.3
        elif customers > 100:
            score += 0.2
        elif customers > 10:
            score += 0.1
        
        # Product development stage
        stage = data.get('product_stage', 'idea')
        if stage == 'launched':
            score += 0.25
        elif stage == 'beta':
            score += 0.15
        elif stage == 'prototype':
            score += 0.1
        
        # User engagement metrics
        if data.get('high_engagement', False):
            score += 0.15
        
        # Partnerships
        if data.get('strategic_partnerships', False):
            score += 0.1
        
        # Media coverage/recognition
        if data.get('media_coverage', False):
            score += 0.1
        
        return min(score, 1.0)

    def _score_adaptability(self, data: Dict[str, Any]) -> float:
        """Score adaptability and resilience"""
        score = 0.2  # Base score
        
        # Pivot capability
        if data.get('has_pivoted', False):
            score += 0.2
        
        # Remote operations capability
        if data.get('remote_capable', False):
            score += 0.15
        
        # Crisis response (important post-COVID)
        if data.get('crisis_resilient', False):
            score += 0.2
        
        # Technology adoption speed
        if data.get('tech_adaptable', False):
            score += 0.15
        
        # Regulatory compliance adaptability
        if data.get('compliance_adaptive', False):
            score += 0.1
        
        return min(score, 1.0)

    def _identify_risk_factors(self, data: Dict[str, Any], scores: Dict[str, float]) -> List[str]:
        """Identify key risk factors"""
        risks = []
        
        # Score-based risks
        for component, score in scores.items():
            if score < 0.4:
                risks.append(f"Low {component.replace('_', ' ')} score ({score:.2f})")
        
        # Ghana-specific risks
        if not data.get('local_team_members', False):
            risks.append("No local team presence in Ghana")
        
        if not data.get('mobile_first', False):
            risks.append("Not optimized for mobile-first market")
        
        if not data.get('local_market_knowledge', False):
            risks.append("Limited local market understanding")
        
        # General business risks
        if data.get('funding_status') == 'none' and not data.get('generating_revenue', False):
            risks.append("No funding and no revenue generation")
        
        if data.get('team_size', 0) == 1:
            risks.append("Single founder - high dependency risk")
        
        return risks

    def _generate_recommendations(self, scores: Dict[str, float], data: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Score-based recommendations
        if scores['business_model'] < 0.6:
            recommendations.append("Strengthen business model: clarify revenue streams and value proposition")
        
        if scores['market_readiness'] < 0.6:
            recommendations.append("Conduct deeper market research, especially in local Ghanaian context")
        
        if scores['team_strength'] < 0.6:
            recommendations.append("Consider team expansion with complementary skills")
        
        if scores['financial_health'] < 0.5:
            recommendations.append("Develop detailed financial projections and funding strategy")
        
        if scores['traction'] < 0.4:
            recommendations.append("Focus on customer acquisition and product-market fit validation")
        
        # Ghana-specific recommendations
        if not data.get('mobile_first', False):
            recommendations.append("Optimize for mobile-first users - critical for Ghanaian market")
        
        if not data.get('local_partnerships', False):
            recommendations.append("Establish local partnerships for market credibility")
        
        # Seedstars alignment recommendations
        if not data.get('social_impact', False):
            recommendations.append("Articulate clear social impact metrics for investor appeal")
        
        return recommendations

    def _calculate_confidence(self, data: Dict[str, Any]) -> float:
        """Calculate confidence based on data completeness"""
        required_fields = [
            'business_description', 'sector', 'team_size', 'product_stage',
            'funding_status', 'target_market', 'revenue_streams'
        ]
        
        available_fields = sum(1 for field in required_fields if data.get(field) is not None)
        base_confidence = available_fields / len(required_fields)
        
        # Boost confidence for verified data
        if data.get('verified_metrics', False):
            base_confidence += 0.1
        
        return min(base_confidence, 1.0)


class LoanRiskScorer:
    def __init__(self):
        self.ghana_risk_factors = {
            'informal_business': 0.3,  # Higher risk but common in Ghana
            'no_collateral': 0.4,
            'seasonal_income': 0.2,  # Common in agribusiness
            'first_time_borrower': 0.2,
            'rural_location': 0.1,  # Lower risk than urban in some cases
        }

    def score_loan_risk(self, applicant_data: Dict[str, Any]) -> ScoringResult:
        """Score loan default risk for Ghanaian context"""
        
        # Base scoring components
        component_scores = {
            'credit_history': self._score_credit_history(applicant_data),
            'income_stability': self._score_income_stability(applicant_data),
            'business_viability': self._score_business_viability(applicant_data),
            'collateral_security': self._score_collateral(applicant_data),
            'social_capital': self._score_social_capital(applicant_data)
        }
        
        # Ghana-specific adjustments
        ghana_adjustment = self._apply_ghana_context(applicant_data)
        
        # Calculate risk score (lower = better)
        risk_score = 1.0 - (
            component_scores['credit_history'] * 0.25 +
            component_scores['income_stability'] * 0.25 +
            component_scores['business_viability'] * 0.20 +
            component_scores['collateral_security'] * 0.15 +
            component_scores['social_capital'] * 0.15
        )
        
        # Apply Ghana context adjustment
        risk_score *= ghana_adjustment
        
        # Generate risk factors and recommendations
        risk_factors = self._identify_loan_risks(applicant_data, component_scores)
        recommendations = self._generate_loan_recommendations(component_scores, applicant_data)
        confidence = self._calculate_loan_confidence(applicant_data)
        
        return ScoringResult(
            overall_score=round(1.0 - risk_score, 2),  # Convert back to positive score
            component_scores=component_scores,
            risk_factors=risk_factors,
            recommendations=recommendations,
            confidence=confidence
        )

    def _score_credit_history(self, data: Dict[str, Any]) -> float:
        """Score credit history (adapted for limited formal credit in Ghana)"""
        score = 0.3  # Base score for limited formal credit systems
        
        # Formal credit history
        if data.get('formal_credit_history', False):
            score += 0.4
        
        # Mobile money history (important in Ghana)
        if data.get('mobile_money_history', False):
            score += 0.2
        
        # Microfinance history
        if data.get('microfinance_history', False):
            score += 0.1
        
        return min(score, 1.0)

    def _score_income_stability(self, data: Dict[str, Any]) -> float:
        """Score income stability"""
        score = 0.1
        
        income_type = data.get('income_type', 'irregular')
        if income_type == 'salary':
            score += 0.4
        elif income_type == 'business_regular':
            score += 0.3
        elif income_type == 'seasonal':
            score += 0.2
        
        # Income verification
        if data.get('income_verified', False):
            score += 0.2
        
        # Multiple income sources
        if data.get('multiple_income_sources', False):
            score += 0.1
        
        return min(score, 1.0)

    def _score_business_viability(self, data: Dict[str, Any]) -> float:
        """Score business viability for loan purposes"""
        score = 0.2
        
        # Business age
        business_age = data.get('business_age_months', 0)
        if business_age > 24:
            score += 0.3
        elif business_age > 12:
            score += 0.2
        elif business_age > 6:
            score += 0.1
        
        # Market position
        if data.get('established_customer_base', False):
            score += 0.2
        
        # Growth trajectory
        if data.get('growing_revenue', False):
            score += 0.15
        
        # Sector resilience
        sector = data.get('business_sector', '').lower()
        if sector in ['agriculture', 'food', 'healthcare']:
            score += 0.15  # Essential sectors
        
        return min(score, 1.0)

    def _score_collateral(self, data: Dict[str, Any]) -> float:
        """Score available collateral"""
        score = 0.0
        
        if data.get('property_ownership', False):
            score += 0.4
        
        if data.get('business_assets', False):
            score += 0.3
        
        if data.get('livestock_assets', False):
            score += 0.2  # Important in rural Ghana
        
        if data.get('group_guarantee', False):
            score += 0.1  # Common in microfinance
        
        return min(score, 1.0)

    def _score_social_capital(self, data: Dict[str, Any]) -> float:
        """Score social capital and community standing"""
        score = 0.2
        
        if data.get('community_leader', False):
            score += 0.3
        
        if data.get('religious_leader_reference', False):
            score += 0.2  # Important in Ghanaian context
        
        if data.get('trade_association_member', False):
            score += 0.2
        
        if data.get('long_term_resident', False):
            score += 0.1
        
        return min(score, 1.0)

    def _apply_ghana_context(self, data: Dict[str, Any]) -> float:
        """Apply Ghana-specific risk adjustments"""
        adjustment = 1.0
        
        # Location-based adjustments
        location = data.get('location_type', 'urban')
        if location == 'rural':
            adjustment *= 0.9  # Slightly lower risk in rural areas
        
        # Gender-based adjustments (women often better borrowers)
        if data.get('gender') == 'female':
            adjustment *= 0.85
        
        # Age adjustments
        age = data.get('age', 30)
        if 25 <= age <= 45:
            adjustment *= 0.9  # Prime age group
        
        return adjustment

    def _identify_loan_risks(self, data: Dict[str, Any], scores: Dict[str, float]) -> List[str]:
        """Identify specific loan risks"""
        risks = []
        
        if scores['credit_history'] < 0.4:
            risks.append("Limited credit history")
        
        if scores['income_stability'] < 0.5:
            risks.append("Irregular income pattern")
        
        if not data.get('income_verified', False):
            risks.append("Unverified income claims")
        
        if data.get('business_age_months', 0) < 6:
            risks.append("Very new business venture")
        
        return risks

    def _generate_loan_recommendations(self, scores: Dict[str, float], data: Dict[str, Any]) -> List[str]:
        """Generate loan-specific recommendations"""
        recommendations = []
        
        if scores['credit_history'] < 0.5:
            recommendations.append("Start with smaller loan amount to build credit history")
        
        if scores['collateral_security'] < 0.3:
            recommendations.append("Consider group lending or alternative guarantee mechanisms")
        
        if data.get('business_sector') == 'agriculture':
            recommendations.append("Structure repayment around harvest cycles")
        
        return recommendations

    def _calculate_loan_confidence(self, data: Dict[str, Any]) -> float:
        """Calculate confidence in loan risk assessment"""
        required_fields = ['income_type', 'business_age_months', 'location_type', 'business_sector']
        available_fields = sum(1 for field in required_fields if data.get(field) is not None)
        
        return available_fields / len(required_fields)


# Factory function for easy integration
def create_scoring_engine():
    return {
        'startup_scorer': StartupReadinessScorer(),
        'loan_scorer': LoanRiskScorer()
    }