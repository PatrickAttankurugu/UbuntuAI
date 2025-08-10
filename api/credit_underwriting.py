import openai
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import json
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib
from config.settings import settings
from api.memory_chains import FeedbackMemoryChain

@dataclass
class CreditAssessment:
    applicant_id: str
    credit_score: float
    risk_category: str
    loan_recommendation: Dict[str, Any]
    approval_probability: float
    recommended_amount: float
    recommended_term: int
    interest_rate_suggestion: float
    collateral_requirements: List[str]
    risk_factors: List[str]
    mitigating_factors: List[str]
    alternative_products: List[Dict[str, Any]]
    confidence_level: float
    assessment_date: str
    review_requirements: List[str]

@dataclass
class LoanPerformancePrediction:
    predicted_default_probability: float
    expected_loss: float
    performance_indicators: Dict[str, float]
    early_warning_signals: List[str]
    success_indicators: List[str]
    repayment_behavior_prediction: Dict[str, float]

class CreditUnderwritingEngine:
    """
    Advanced credit underwriting system for emerging markets
    Designed for informal economies and limited credit history scenarios
    Optimized for women-led businesses and small agribusinesses
    """
    
    def __init__(self):
        self.memory_chain = FeedbackMemoryChain()
        
        # ML Models
        self.default_predictor = None
        self.amount_recommender = None
        self.term_optimizer = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
        # African market credit parameters
        self.african_credit_context = self._load_african_credit_context()
        
        # Alternative data sources for credit scoring
        self.alternative_data_weights = {
            "mobile_money_history": 0.25,
            "social_network_analysis": 0.15,
            "business_transaction_patterns": 0.20,
            "community_standing": 0.10,
            "seasonal_income_patterns": 0.15,
            "informal_credit_history": 0.15
        }
        
        # Initialize models
        self._initialize_models()
        
        # Load historical performance data
        self.historical_performance = self._load_historical_performance()

    def assess_credit_application(self, 
                                applicant_data: Dict[str, Any],
                                loan_request: Dict[str, Any],
                                alternative_data: Dict[str, Any] = None) -> CreditAssessment:
        """
        Comprehensive credit assessment using traditional + alternative data
        """
        
        # Step 1: Extract and process features
        features = self._extract_credit_features(applicant_data, alternative_data)
        
        # Step 2: Calculate base credit score
        base_score = self._calculate_base_credit_score(features)
        
        # Step 3: Apply alternative data scoring
        alternative_score = self._score_alternative_data(alternative_data or {})
        
        # Step 4: Combine scores with African market adjustments
        final_score = self._combine_scores(base_score, alternative_score, applicant_data)
        
        # Step 5: Generate loan recommendations
        loan_recommendation = self._generate_loan_recommendation(
            final_score, loan_request, applicant_data, features
        )
        
        # Step 6: Risk assessment and categorization
        risk_assessment = self._assess_risks(features, applicant_data, loan_recommendation)
        
        # Step 7: Generate alternative products if needed
        alternatives = self._suggest_alternative_products(
            final_score, loan_request, risk_assessment
        )
        
        # Create comprehensive assessment
        assessment = CreditAssessment(
            applicant_id=applicant_data.get('id', f"app_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
            credit_score=final_score,
            risk_category=self._categorize_risk(final_score),
            loan_recommendation=loan_recommendation,
            approval_probability=loan_recommendation.get('approval_probability', 0.0),
            recommended_amount=loan_recommendation.get('recommended_amount', 0),
            recommended_term=loan_recommendation.get('recommended_term', 12),
            interest_rate_suggestion=loan_recommendation.get('interest_rate', 0.15),
            collateral_requirements=risk_assessment.get('collateral_requirements', []),
            risk_factors=risk_assessment.get('risk_factors', []),
            mitigating_factors=risk_assessment.get('mitigating_factors', []),
            alternative_products=alternatives,
            confidence_level=self._calculate_confidence(features, alternative_data),
            assessment_date=datetime.now().isoformat(),
            review_requirements=self._determine_review_requirements(final_score, loan_request)
        )
        
        # Store assessment for learning
        self.memory_chain.store_credit_assessment(applicant_data, assessment)
        
        return assessment

    def predict_loan_performance(self, 
                               applicant_data: Dict[str, Any],
                               loan_terms: Dict[str, Any]) -> LoanPerformancePrediction:
        """
        Predict loan performance and repayment behavior
        """
        
        # Extract performance prediction features
        features = self._extract_performance_features(applicant_data, loan_terms)
        
        # Predict default probability
        default_prob = self._predict_default_probability(features)
        
        # Calculate expected loss
        expected_loss = self._calculate_expected_loss(default_prob, loan_terms)
        
        # Generate performance indicators
        performance_indicators = self._generate_performance_indicators(features)
        
        # Identify early warning signals
        warning_signals = self._identify_warning_signals(features, applicant_data)
        
        # Identify success indicators
        success_indicators = self._identify_success_indicators(features, applicant_data)
        
        # Predict repayment behavior patterns
        repayment_behavior = self._predict_repayment_behavior(features)
        
        return LoanPerformancePrediction(
            predicted_default_probability=default_prob,
            expected_loss=expected_loss,
            performance_indicators=performance_indicators,
            early_warning_signals=warning_signals,
            success_indicators=success_indicators,
            repayment_behavior_prediction=repayment_behavior
        )

    def _extract_credit_features(self, 
                               applicant_data: Dict[str, Any],
                               alternative_data: Dict[str, Any] = None) -> np.ndarray:
        """Extract features for credit scoring"""
        
        features = []
        
        # Basic demographic features
        features.extend([
            applicant_data.get('age', 30) / 100,  # Normalized age
            1 if applicant_data.get('gender') == 'female' else 0,
            1 if applicant_data.get('marital_status') == 'married' else 0,
            applicant_data.get('dependents', 0) / 10,
            applicant_data.get('education_level', 2) / 5  # 1-5 scale
        ])
        
        # Employment and income features
        features.extend([
            np.log1p(applicant_data.get('monthly_income', 1000)),
            applicant_data.get('employment_duration_months', 12) / 120,
            1 if applicant_data.get('employment_type') == 'formal' else 0,
            1 if applicant_data.get('has_secondary_income', False) else 0,
            applicant_data.get('income_stability_score', 0.5)
        ])
        
        # Business-related features (if applicable)
        if applicant_data.get('business_owner', False):
            features.extend([
                applicant_data.get('business_age_months', 0) / 120,
                np.log1p(applicant_data.get('business_revenue', 1000)),
                applicant_data.get('business_profit_margin', 0.1),
                1 if applicant_data.get('business_registered', False) else 0,
                applicant_data.get('business_growth_rate', 0.0)
            ])
        else:
            features.extend([0, 0, 0, 0, 0])  # Zero padding
        
        # Financial behavior features
        features.extend([
            applicant_data.get('savings_amount', 0) / 10000,
            applicant_data.get('debt_to_income_ratio', 0.0),
            1 if applicant_data.get('has_bank_account', False) else 0,
            applicant_data.get('credit_history_length_months', 0) / 120,
            applicant_data.get('payment_history_score', 0.5)
        ])
        
        # Location and social features
        features.extend([
            1 if applicant_data.get('location_type') == 'urban' else 0,
            applicant_data.get('community_standing_score', 0.5),
            1 if applicant_data.get('has_guarantor', False) else 0,
            applicant_data.get('social_network_size', 50) / 500,
            applicant_data.get('local_reputation_score', 0.5)
        ])
        
        # Add alternative data features if available
        if alternative_data:
            alt_features = self._extract_alternative_features(alternative_data)
            features.extend(alt_features)
        else:
            features.extend([0] * 10)  # Zero padding for missing alternative data
        
        return np.array(features)

    def _extract_alternative_features(self, alternative_data: Dict[str, Any]) -> List[float]:
        """Extract features from alternative data sources"""
        
        features = []
        
        # Mobile money transaction features
        mobile_money = alternative_data.get('mobile_money_data', {})
        features.extend([
            mobile_money.get('transaction_frequency', 0) / 100,
            np.log1p(mobile_money.get('average_transaction_amount', 100)),
            mobile_money.get('account_age_months', 0) / 120,
            mobile_money.get('balance_consistency_score', 0.5),
            mobile_money.get('transaction_diversity_score', 0.5)
        ])
        
        # Social network analysis features
        social_data = alternative_data.get('social_network_data', {})
        features.extend([
            social_data.get('network_quality_score', 0.5),
            social_data.get('borrower_connections', 0) / 100,
            social_data.get('successful_borrower_ratio', 0.5),
            social_data.get('community_endorsements', 0) / 20,
            social_data.get('digital_footprint_score', 0.5)
        ])
        
        return features

    def _calculate_base_credit_score(self, features: np.ndarray) -> float:
        """Calculate base credit score using traditional methods"""
        
        if self.default_predictor is None:
            # Fallback scoring method
            return self._fallback_credit_scoring(features)
        
        try:
            # Use trained ML model for prediction
            scaled_features = self.scaler.transform([features])
            default_probability = self.default_predictor.predict_proba(scaled_features)[0][1]
            
            # Convert default probability to credit score (higher is better)
            credit_score = (1 - default_probability) * 100
            
            return max(min(credit_score, 100), 0)  # Bound between 0-100
            
        except Exception as e:
            return self._fallback_credit_scoring(features)

    def _score_alternative_data(self, alternative_data: Dict[str, Any]) -> float:
        """Score alternative data sources"""
        
        total_score = 0.0
        total_weight = 0.0
        
        # Mobile money scoring
        if 'mobile_money_data' in alternative_data:
            mm_data = alternative_data['mobile_money_data']
            mm_score = self._score_mobile_money_data(mm_data)
            weight = self.alternative_data_weights['mobile_money_history']
            total_score += mm_score * weight
            total_weight += weight
        
        # Social network scoring
        if 'social_network_data' in alternative_data:
            sn_data = alternative_data['social_network_data']
            sn_score = self._score_social_network_data(sn_data)
            weight = self.alternative_data_weights['social_network_analysis']
            total_score += sn_score * weight
            total_weight += weight
        
        # Business transaction patterns
        if 'business_transactions' in alternative_data:
            bt_data = alternative_data['business_transactions']
            bt_score = self._score_business_transactions(bt_data)
            weight = self.alternative_data_weights['business_transaction_patterns']
            total_score += bt_score * weight
            total_weight += weight
        
        # Community standing
        if 'community_data' in alternative_data:
            comm_data = alternative_data['community_data']
            comm_score = self._score_community_standing(comm_data)
            weight = self.alternative_data_weights['community_standing']
            total_score += comm_score * weight
            total_weight += weight
        
        # Return weighted average or 0 if no alternative data
        return total_score / total_weight if total_weight > 0 else 0.0

    def _score_mobile_money_data(self, mobile_money_data: Dict[str, Any]) -> float:
        """Score mobile money transaction history"""
        
        score = 0.0
        
        # Transaction frequency (daily transactions indicate active usage)
        freq = mobile_money_data.get('daily_transaction_frequency', 0)
        score += min(freq / 5, 1.0) * 25  # Max 25 points
        
        # Account age (longer history is better)
        age_months = mobile_money_data.get('account_age_months', 0)
        score += min(age_months / 24, 1.0) * 20  # Max 20 points
        
        # Balance consistency (steady balance indicates stability)
        consistency = mobile_money_data.get('balance_consistency_score', 0)
        score += consistency * 20  # Max 20 points
        
        # Transaction diversity (various transaction types indicate normal usage)
        diversity = mobile_money_data.get('transaction_diversity_score', 0)
        score += diversity * 15  # Max 15 points
        
        # No overdrafts or failed transactions
        reliability = mobile_money_data.get('transaction_reliability_score', 0.5)
        score += reliability * 20  # Max 20 points
        
        return min(score, 100)

    def _score_social_network_data(self, social_network_data: Dict[str, Any]) -> float:
        """Score social network and digital footprint"""
        
        score = 0.0
        
        # Network quality (connections with good credit history)
        network_quality = social_network_data.get('network_quality_score', 0)
        score += network_quality * 40  # Max 40 points
        
        # Community endorsements
        endorsements = social_network_data.get('community_endorsements', 0)
        score += min(endorsements / 10, 1.0) * 30  # Max 30 points
        
        # Digital footprint consistency
        digital_footprint = social_network_data.get('digital_footprint_score', 0)
        score += digital_footprint * 30  # Max 30 points
        
        return min(score, 100)

    def _combine_scores(self, 
                       base_score: float, 
                       alternative_score: float,
                       applicant_data: Dict[str, Any]) -> float:
        """Combine base and alternative scores with African market adjustments"""
        
        # Base combination (weighted average)
        if alternative_score > 0:
            combined_score = (base_score * 0.7) + (alternative_score * 0.3)
        else:
            combined_score = base_score
        
        # Apply African market adjustments
        combined_score = self._apply_african_adjustments(combined_score, applicant_data)
        
        return max(min(combined_score, 100), 0)

    def _apply_african_adjustments(self, score: float, applicant_data: Dict[str, Any]) -> float:
        """Apply African market-specific adjustments"""
        
        adjusted_score = score
        
        # Gender adjustment (women often better borrowers in African context)
        if applicant_data.get('gender') == 'female':
            adjusted_score += 5
        
        # Rural vs urban adjustment
        if applicant_data.get('location_type') == 'rural':
            # Rural borrowers often have lower default rates
            adjusted_score += 3
        
        # Agricultural business adjustment
        if applicant_data.get('business_sector') == 'agriculture':
            # Seasonal considerations but often stable
            if applicant_data.get('has_irrigation', False):
                adjusted_score += 4
            else:
                adjusted_score -= 2
        
        # Group guarantee adjustment
        if applicant_data.get('has_group_guarantee', False):
            adjusted_score += 8
        
        # Local language fluency (indicates local integration)
        if applicant_data.get('local_language_fluent', False):
            adjusted_score += 2
        
        # Microfinance history adjustment
        if applicant_data.get('microfinance_history', False):
            adjusted_score += 6
        
        return adjusted_score

    def _generate_loan_recommendation(self, 
                                    credit_score: float,
                                    loan_request: Dict[str, Any],
                                    applicant_data: Dict[str, Any],
                                    features: np.ndarray) -> Dict[str, Any]:
        """Generate comprehensive loan recommendation"""
        
        requested_amount = loan_request.get('amount', 10000)
        requested_term = loan_request.get('term_months', 12)
        
        # Determine approval probability
        approval_prob = self._calculate_approval_probability(credit_score)
        
        # Calculate recommended amount
        recommended_amount = self._calculate_recommended_amount(
            credit_score, requested_amount, applicant_data
        )
        
        # Optimize loan terms
        recommended_term = self._optimize_loan_term(
            credit_score, requested_term, applicant_data
        )
        
        # Calculate interest rate
        interest_rate = self._calculate_interest_rate(credit_score, applicant_data)
        
        # Determine loan structure
        loan_structure = self._determine_loan_structure(applicant_data)
        
        return {
            "approval_status": "approved" if approval_prob > 0.7 else "conditional" if approval_prob > 0.5 else "declined",
            "approval_probability": approval_prob,
            "recommended_amount": recommended_amount,
            "requested_amount": requested_amount,
            "amount_adjustment_reason": self._get_amount_adjustment_reason(
                recommended_amount, requested_amount, credit_score
            ),
            "recommended_term": recommended_term,
            "interest_rate": interest_rate,
            "loan_structure": loan_structure,
            "repayment_frequency": self._recommend_repayment_frequency(applicant_data),
            "grace_period_days": self._recommend_grace_period(applicant_data),
            "conditions": self._generate_loan_conditions(credit_score, applicant_data)
        }

    def _load_african_credit_context(self) -> Dict[str, Any]:
        """Load African credit market context"""
        return {
            "default_rates_by_sector": {
                "agriculture": 0.12,
                "retail": 0.18,
                "services": 0.15,
                "manufacturing": 0.14,
                "technology": 0.10
            },
            "seasonal_patterns": {
                "agriculture": {
                    "high_risk_months": [3, 4, 5, 6],  # Pre-harvest
                    "low_risk_months": [10, 11, 12, 1]  # Post-harvest
                }
            },
            "regional_risk_factors": {
                "northern_ghana": {"drought_risk": 0.3, "market_access": 0.6},
                "coastal_regions": {"flooding_risk": 0.2, "market_access": 0.8}
            },
            "successful_lending_patterns": {
                "group_lending": {"success_rate": 0.88, "optimal_group_size": "5-8"},
                "women_focused": {"success_rate": 0.85, "preferred_terms": "6-12 months"},
                "agribusiness": {"success_rate": 0.82, "seasonal_repayment": True}
            }
        }

    def _initialize_models(self):
        """Initialize ML models (in production, load pre-trained models)"""
        try:
            # In production, load pre-trained models
            # self.default_predictor = joblib.load('models/default_predictor.pkl')
            # self.amount_recommender = joblib.load('models/amount_recommender.pkl')
            
            # For demo, create placeholder models
            self.default_predictor = RandomForestClassifier(n_estimators=100, random_state=42)
            self.amount_recommender = GradientBoostingRegressor(n_estimators=100, random_state=42)
            
            # Initialize with dummy data for demonstration
            self._train_demo_models()
            
        except Exception as e:
            print(f"Model initialization failed: {e}")
            self.default_predictor = None
            self.amount_recommender = None

    def _train_demo_models(self):
        """Train models with synthetic data for demonstration"""
        # Generate synthetic training data
        n_samples = 1000
        n_features = 30
        
        X = np.random.rand(n_samples, n_features)
        y_default = np.random.choice([0, 1], n_samples, p=[0.85, 0.15])  # 15% default rate
        y_amount = np.random.rand(n_samples) * 100000  # Random loan amounts
        
        # Fit scaler
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        
        # Train models
        self.default_predictor.fit(X_scaled, y_default)
        self.amount_recommender.fit(X_scaled, y_amount)

    def _fallback_credit_scoring(self, features: np.ndarray) -> float:
        """Fallback credit scoring when ML models unavailable"""
        
        # Simple rule-based scoring
        score = 50  # Base score
        
        # Age factor (features[0])
        age_normalized = features[0]
        if 0.25 <= age_normalized <= 0.65:  # 25-65 years
            score += 10
        
        # Gender factor (features[1]) - women often better borrowers
        if features[1] == 1:  # Female
            score += 8
        
        # Income factor (features[5])
        income_log = features[5]
        if income_log > 8:  # Good income
            score += 15
        elif income_log > 7:
            score += 10
        elif income_log > 6:
            score += 5
        
        # Employment stability (features[6])
        emp_duration = features[6]
        if emp_duration > 0.5:  # > 5 years
            score += 12
        elif emp_duration > 0.25:  # > 2.5 years
            score += 8
        
        # Business factors if applicable
        if features[10] > 0:  # Has business
            score += 8
            if features[13] == 1:  # Business registered
                score += 5
        
        # Financial behavior
        if features[15] > 0.1:  # Has savings
            score += 10
        if features[17] == 1:  # Has bank account
            score += 5
        
        return max(min(score, 100), 0)

# Factory function
def create_credit_underwriting_engine():
    return CreditUnderwritingEngine()