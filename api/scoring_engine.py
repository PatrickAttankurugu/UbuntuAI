import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import re
from datetime import datetime
import openai
from config.settings import settings
import random

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
        
        return risks
    
    def _generate_recommendations(self, data: Dict[str, Any], scores: Dict[str, float]) -> List[str]:
        recommendations = []
        
        if scores['team'] < 0.7:
            recommendations.append("Consider expanding team with complementary skills")
        
        if not data.get('generating_revenue'):
            recommendations.append("Focus on customer validation and early revenue")
        
        if scores['market'] < 0.7:
            recommendations.append("Conduct more market research and validation")
        
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
        
        return risks
    
    def _generate_loan_recommendations(self, data: Dict[str, Any], scores: Dict[str, float]) -> List[str]:
        recommendations = []
        
        if scores['social_capital'] > 0.7:
            recommendations.append("Consider group lending model")
        
        if scores['credit_history'] < 0.6:
            recommendations.append("Start with smaller loan amount")
        
        if data.get('mobile_money_history'):
            recommendations.append("Monitor repayments via mobile money")
        
        recommendations.append("Provide financial literacy training")
        
        return recommendations

def create_scoring_engine() -> Dict[str, Any]:
    """Create and return scoring engine components"""
    return {
        'startup_scorer': StartupReadinessScorer(),
        'loan_scorer': LoanRiskScorer()
    }