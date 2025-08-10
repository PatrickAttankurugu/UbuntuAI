import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import re
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class ScoringResult:
    overall_score: float
    component_scores: Dict[str, float]
    risk_factors: List[str]
    recommendations: List[str]
    confidence: float

class StartupReadinessScorer:
    def __init__(self):
        self.weights = {
            'team': 0.25,
            'market': 0.25,
            'product': 0.20,
            'business_model': 0.15,
            'traction': 0.15
        }
    
    def score_startup(self, data: Dict[str, Any]) -> ScoringResult:
        """Score a startup's readiness"""
        
        # Calculate component scores
        component_scores = {
            'team': self._score_team(data),
            'market': self._score_market(data),
            'product': self._score_product(data),
            'business_model': self._score_business_model(data),
            'traction': self._score_traction(data)
        }
        
        # Calculate overall score
        overall_score = sum(
            score * self.weights[component] 
            for component, score in component_scores.items()
        )
        
        # Generate risk factors and recommendations
        risk_factors = self._identify_risk_factors(data, component_scores)
        recommendations = self._generate_recommendations(data, component_scores)
        
        return ScoringResult(
            overall_score=overall_score,
            component_scores=component_scores,
            risk_factors=risk_factors,
            recommendations=recommendations,
            confidence=0.8
        )
    
    def _score_team(self, data: Dict[str, Any]) -> float:
        score = 0.5  # base score
        
        team_size = data.get('team_size', 1)
        if team_size >= 2:
            score += 0.2
        if team_size >= 3:
            score += 0.1
        
        if data.get('local_team_members'):
            score += 0.1
        
        if data.get('local_market_knowledge'):
            score += 0.1
        
        return min(score, 1.0)
    
    def _score_market(self, data: Dict[str, Any]) -> float:
        score = 0.4  # base score
        
        if data.get('mobile_first'):
            score += 0.2
        
        if data.get('serves_rural_market'):
            score += 0.15
        
        sector = data.get('sector', '').lower()
        if sector in ['fintech', 'agritech', 'healthtech']:
            score += 0.15
        
        if data.get('target_market'):
            score += 0.1
        
        return min(score, 1.0)
    
    def _score_product(self, data: Dict[str, Any]) -> float:
        score = 0.3  # base score
        
        stage = data.get('product_stage', '').lower()
        if stage == 'idea':
            score += 0.1
        elif stage == 'prototype':
            score += 0.3
        elif stage == 'beta':
            score += 0.5
        elif stage == 'launched':
            score += 0.7
        
        return min(score, 1.0)
    
    def _score_business_model(self, data: Dict[str, Any]) -> float:
        score = 0.4  # base score
        
        if data.get('generating_revenue'):
            score += 0.3
        
        if data.get('business_description'):
            score += 0.2
        
        if data.get('clear_value_proposition'):
            score += 0.1
        
        return min(score, 1.0)
    
    def _score_traction(self, data: Dict[str, Any]) -> float:
        score = 0.2  # base score
        
        customer_count = data.get('customer_count', 0)
        if customer_count > 0:
            score += 0.2
        if customer_count > 50:
            score += 0.2
        if customer_count > 100:
            score += 0.2
        
        if data.get('partnership'):
            score += 0.2
        
        return min(score, 1.0)
    
    def _identify_risk_factors(self, data: Dict[str, Any], scores: Dict[str, float]) -> List[str]:
        risks = []
        
        if scores['team'] < 0.6:
            risks.append("Small team size may limit execution capacity")
        
        if scores['market'] < 0.6:
            risks.append("Market opportunity may be limited")
        
        if scores['product'] < 0.5:
            risks.append("Product development stage is early")
        
        if not data.get('generating_revenue'):
            risks.append("No revenue generation yet")
        
        if data.get('team_size', 1) == 1:
            risks.append("Single founder may face execution challenges")
        
        sector = data.get('sector', '').lower()
        if sector in ['cryptocurrency', 'gaming']:
            risks.append("Sector may face regulatory challenges in some African markets")
        
        return risks
    
    def _generate_recommendations(self, data: Dict[str, Any], scores: Dict[str, float]) -> List[str]:
        recommendations = []
        
        if scores['team'] < 0.7:
            recommendations.append("Consider expanding team with complementary skills")
        
        if not data.get('generating_revenue'):
            recommendations.append("Focus on customer validation and early revenue")
        
        if scores['market'] < 0.7:
            recommendations.append("Conduct more market research and validation")
        
        if scores['product'] < 0.6:
            recommendations.append("Develop MVP and gather user feedback")
        
        if data.get('team_size', 1) == 1:
            recommendations.append("Consider finding a co-founder or key team member")
        
        # Add Africa-specific recommendations
        if data.get('sector') in ['fintech', 'healthtech']:
            recommendations.append("Ensure compliance with local financial/health regulations")
        
        if not data.get('mobile_first'):
            recommendations.append("Consider mobile-first approach for African markets")
        
        recommendations.append("Consider applying for startup accelerator programs")
        
        return recommendations

class LoanRiskScorer:
    def __init__(self):
        self.weights = {
            'credit_history': 0.25,
            'income_stability': 0.25,
            'business_viability': 0.20,
            'collateral': 0.15,
            'social_capital': 0.15
        }
    
    def score_loan_risk(self, data: Dict[str, Any]) -> ScoringResult:
        """Score loan risk for African markets"""
        
        component_scores = {
            'credit_history': self._score_credit_history(data),
            'income_stability': self._score_income_stability(data),
            'business_viability': self._score_business_viability(data),
            'collateral': self._score_collateral(data),
            'social_capital': self._score_social_capital(data)
        }
        
        overall_score = sum(
            score * self.weights[component] 
            for component, score in component_scores.items()
        )
        
        risk_factors = self._identify_loan_risks(data, component_scores)
        recommendations = self._generate_loan_recommendations(data, component_scores)
        
        return ScoringResult(
            overall_score=overall_score,
            component_scores=component_scores,
            risk_factors=risk_factors,
            recommendations=recommendations,
            confidence=0.75
        )
    
    def _score_credit_history(self, data: Dict[str, Any]) -> float:
        score = 0.3  # base for no formal history
        
        if data.get('has_bank_account'):
            score += 0.2
        
        if data.get('mobile_money_history'):
            score += 0.3
        
        if data.get('previous_loans_paid'):
            score += 0.2
        
        return min(score, 1.0)
    
    def _score_income_stability(self, data: Dict[str, Any]) -> float:
        score = 0.4  # base score
        
        income_type = data.get('income_type', '').lower()
        if 'regular' in income_type:
            score += 0.3
        elif 'business' in income_type:
            score += 0.2
        
        business_age = data.get('business_age_months', 0)
        if business_age > 12:
            score += 0.2
        if business_age > 24:
            score += 0.1
        
        return min(score, 1.0)
    
    def _score_business_viability(self, data: Dict[str, Any]) -> float:
        score = 0.5  # base score
        
        if data.get('established_customer_base'):
            score += 0.2
        
        if data.get('growing_revenue'):
            score += 0.2
        
        if data.get('local_market_knowledge'):
            score += 0.1
        
        return min(score, 1.0)
    
    def _score_collateral(self, data: Dict[str, Any]) -> float:
        score = 0.2  # base for no formal collateral
        
        if data.get('property_ownership'):
            score += 0.4
        
        if data.get('business_assets'):
            score += 0.3
        
        if data.get('inventory_value'):
            score += 0.1
        
        return min(score, 1.0)
    
    def _score_social_capital(self, data: Dict[str, Any]) -> float:
        score = 0.3  # base score
        
        if data.get('community_leader'):
            score += 0.3
        
        if data.get('group_member'):
            score += 0.2
        
        if data.get('family_support'):
            score += 0.1
        
        if data.get('local_references'):
            score += 0.1
        
        return min(score, 1.0)
    
    def _identify_loan_risks(self, data: Dict[str, Any], scores: Dict[str, float]) -> List[str]:
        risks = []
        
        if scores['credit_history'] < 0.5:
            risks.append("Limited formal credit history")
        
        if scores['income_stability'] < 0.6:
            risks.append("Income may be irregular or seasonal")
        
        if scores['collateral'] < 0.5:
            risks.append("Limited collateral available")
        
        location_type = data.get('location_type', '').lower()
        if 'rural' in location_type:
            risks.append("Rural location may affect income stability")
        
        if not data.get('mobile_money_history'):
            risks.append("No mobile money transaction history for monitoring")
        
        return risks
    
    def _generate_loan_recommendations(self, data: Dict[str, Any], scores: Dict[str, float]) -> List[str]:
        recommendations = []
        
        if scores['social_capital'] > 0.7:
            recommendations.append("Consider group lending model")
        
        if scores['credit_history'] < 0.6:
            recommendations.append("Start with smaller loan amount")
        
        if data.get('mobile_money_history'):
            recommendations.append("Monitor repayments via mobile money")
        
        if not data.get('has_bank_account'):
            recommendations.append("Encourage opening a bank account")
        
        recommendations.append("Provide financial literacy training")
        
        if scores['business_viability'] > 0.7:
            recommendations.append("Consider business development support")
        
        return recommendations

class MarketOpportunityScorer:
    """Score market opportunities in African context"""
    
    def __init__(self):
        self.african_market_factors = {
            'mobile_penetration': 0.2,
            'economic_growth': 0.2,
            'regulatory_environment': 0.15,
            'infrastructure': 0.15,
            'competition_level': 0.15,
            'local_demand': 0.15
        }
    
    def score_market_opportunity(self, data: Dict[str, Any]) -> ScoringResult:
        """Score a market opportunity in African context"""
        
        component_scores = {
            'mobile_penetration': self._score_mobile_penetration(data),
            'economic_growth': self._score_economic_growth(data),
            'regulatory_environment': self._score_regulatory_environment(data),
            'infrastructure': self._score_infrastructure(data),
            'competition_level': self._score_competition_level(data),
            'local_demand': self._score_local_demand(data)
        }
        
        overall_score = sum(
            score * self.african_market_factors[component]
            for component, score in component_scores.items()
        )
        
        risk_factors = self._identify_market_risks(data, component_scores)
        recommendations = self._generate_market_recommendations(data, component_scores)
        
        return ScoringResult(
            overall_score=overall_score,
            component_scores=component_scores,
            risk_factors=risk_factors,
            recommendations=recommendations,
            confidence=0.7
        )
    
    def _score_mobile_penetration(self, data: Dict[str, Any]) -> float:
        # Higher scores for markets with high mobile penetration
        country = data.get('country', '').lower()
        
        high_mobile_countries = ['kenya', 'ghana', 'rwanda', 'uganda']
        medium_mobile_countries = ['nigeria', 'tanzania', 'senegal']
        
        if country in high_mobile_countries:
            return 0.9
        elif country in medium_mobile_countries:
            return 0.7
        else:
            return 0.5
    
    def _score_economic_growth(self, data: Dict[str, Any]) -> float:
        # Score based on economic indicators
        country = data.get('country', '').lower()
        
        fast_growing = ['ghana', 'rwanda', 'ivory coast', 'senegal']
        stable_growth = ['kenya', 'tanzania', 'uganda', 'botswana']
        
        if country in fast_growing:
            return 0.8
        elif country in stable_growth:
            return 0.7
        else:
            return 0.5
    
    def _score_regulatory_environment(self, data: Dict[str, Any]) -> float:
        country = data.get('country', '').lower()
        sector = data.get('sector', '').lower()
        
        # Countries with good business environments
        business_friendly = ['rwanda', 'mauritius', 'botswana', 'south africa']
        
        base_score = 0.8 if country in business_friendly else 0.6
        
        # Sector-specific adjustments
        if sector == 'fintech':
            fintech_friendly = ['kenya', 'ghana', 'nigeria', 'south africa']
            if country in fintech_friendly:
                base_score += 0.1
        
        return min(base_score, 1.0)
    
    def _score_infrastructure(self, data: Dict[str, Any]) -> float:
        country = data.get('country', '').lower()
        
        good_infrastructure = ['south africa', 'mauritius', 'botswana', 'namibia']
        developing_infrastructure = ['ghana', 'kenya', 'rwanda', 'senegal']
        
        if country in good_infrastructure:
            return 0.8
        elif country in developing_infrastructure:
            return 0.6
        else:
            return 0.4
    
    def _score_competition_level(self, data: Dict[str, Any]) -> float:
        # Lower competition = higher score
        sector = data.get('sector', '').lower()
        country = data.get('country', '').lower()
        
        saturated_markets = {
            'fintech': ['kenya', 'nigeria', 'south africa'],
            'e-commerce': ['nigeria', 'south africa', 'egypt'],
            'ride-sharing': ['nigeria', 'kenya', 'south africa']
        }
        
        if sector in saturated_markets and country in saturated_markets[sector]:
            return 0.4  # High competition
        else:
            return 0.7  # Lower competition
    
    def _score_local_demand(self, data: Dict[str, Any]) -> float:
        # Score based on local market demand indicators
        sector = data.get('sector', '').lower()
        
        high_demand_sectors = ['healthtech', 'edtech', 'agritech', 'fintech']
        moderate_demand_sectors = ['e-commerce', 'logistics', 'energy']
        
        if sector in high_demand_sectors:
            return 0.8
        elif sector in moderate_demand_sectors:
            return 0.6
        else:
            return 0.4
    
    def _identify_market_risks(self, data: Dict[str, Any], scores: Dict[str, float]) -> List[str]:
        risks = []
        
        if scores['regulatory_environment'] < 0.6:
            risks.append("Regulatory uncertainty may impact business operations")
        
        if scores['infrastructure'] < 0.5:
            risks.append("Poor infrastructure may limit business scalability")
        
        if scores['competition_level'] < 0.5:
            risks.append("High competition may require significant marketing investment")
        
        country = data.get('country', '').lower()
        if country in ['somalia', 'south sudan', 'central african republic']:
            risks.append("Political instability may affect business environment")
        
        return risks
    
    def _generate_market_recommendations(self, data: Dict[str, Any], scores: Dict[str, float]) -> List[str]:
        recommendations = []
        
        if scores['mobile_penetration'] > 0.7:
            recommendations.append("Leverage mobile-first strategy for market entry")
        
        if scores['infrastructure'] < 0.6:
            recommendations.append("Consider partnerships to overcome infrastructure limitations")
        
        if scores['competition_level'] < 0.6:
            recommendations.append("Focus on differentiation and niche market entry")
        
        if scores['regulatory_environment'] > 0.7:
            recommendations.append("Fast-track market entry while regulatory environment is favorable")
        
        recommendations.append("Conduct thorough local market research before launch")
        
        return recommendations

def create_scoring_engine() -> Dict[str, Any]:
    """Create and return scoring engine components"""
    try:
        return {
            'startup_scorer': StartupReadinessScorer(),
            'loan_scorer': LoanRiskScorer(),
            'market_scorer': MarketOpportunityScorer()
        }
    except Exception as e:
        logger.error(f"Failed to create scoring engines: {e}")
        return {}