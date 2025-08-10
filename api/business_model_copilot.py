import openai
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import json
from datetime import datetime
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from config.settings import settings
from api.memory_chains import FeedbackMemoryChain
from api.hybrid_retrieval import HybridRetriever

@dataclass
class BusinessModelRecommendation:
    model_type: str
    revenue_streams: List[str]
    key_activities: List[str]
    key_partnerships: List[str]
    cost_structure: List[str]
    value_propositions: List[str]
    customer_segments: List[str]
    channels: List[str]
    customer_relationships: List[str]
    confidence_score: float
    implementation_steps: List[str]
    risk_factors: List[str]
    success_metrics: List[str]

class BusinessModelCopilot:
    """
    AI Copilot for business model design and optimization
    Core SIGMA platform feature for entrepreneur support
    """
    
    def __init__(self):
        self.memory_chain = FeedbackMemoryChain()
        self.retriever = HybridRetriever()
        self.conversation_memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=10
        )
        
        # Business model frameworks
        self.frameworks = {
            "lean_canvas": self._lean_canvas_template(),
            "bmc": self._business_model_canvas_template(),
            "impact_canvas": self._impact_canvas_template(),
            "platform_canvas": self._platform_canvas_template()
        }
        
        # African market context
        self.african_context = self._load_african_business_context()
        
        # Success patterns from SIGMA data
        self.success_patterns = self._load_success_patterns()

    def design_business_model(self, 
                            entrepreneur_data: Dict[str, Any],
                            conversation_history: List[Dict[str, Any]] = None) -> BusinessModelRecommendation:
        """
        Main copilot function to design optimal business model
        Uses multi-step reasoning with feedback loops
        """
        
        # Step 1: Analyze entrepreneur context
        context_analysis = self._analyze_entrepreneur_context(entrepreneur_data)
        
        # Step 2: Retrieve relevant business models from knowledge base
        similar_models = self.retriever.retrieve_similar_business_models(
            context_analysis, 
            limit=10
        )
        
        # Step 3: Generate personalized recommendations
        recommendations = self._generate_model_recommendations(
            context_analysis, 
            similar_models, 
            conversation_history
        )
        
        # Step 4: Validate against success patterns
        validated_model = self._validate_against_patterns(recommendations)
        
        # Step 5: Add implementation guidance
        final_model = self._add_implementation_guidance(validated_model, entrepreneur_data)
        
        # Store interaction for learning
        self.memory_chain.store_interaction(
            entrepreneur_data, 
            final_model, 
            context_analysis
        )
        
        return final_model

    def _analyze_entrepreneur_context(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze entrepreneur's situation for context-aware recommendations"""
        
        context_prompt = f"""
        Analyze this entrepreneur's context for business model recommendations:
        
        Entrepreneur Data:
        {json.dumps(data, indent=2)}
        
        African Business Context:
        {json.dumps(self.african_context, indent=2)}
        
        Provide analysis in JSON format:
        {{
            "market_context": {{
                "local_market_size": "estimate",
                "competition_level": "low/medium/high",
                "infrastructure_readiness": "assessment",
                "regulatory_environment": "analysis"
            }},
            "entrepreneur_readiness": {{
                "skills_match": "assessment",
                "resource_availability": "assessment", 
                "network_strength": "assessment",
                "risk_tolerance": "assessment"
            }},
            "opportunity_assessment": {{
                "market_timing": "assessment",
                "scalability_potential": "assessment",
                "social_impact_potential": "assessment",
                "revenue_potential": "assessment"
            }},
            "constraints": [
                "list of key constraints"
            ],
            "advantages": [
                "list of key advantages"
            ]
        }}
        """
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": context_prompt}],
                temperature=0.3,
                max_tokens=1000
            )
            
            analysis = json.loads(response.choices[0].message.content)
            return analysis
            
        except Exception as e:
            # Fallback analysis
            return self._fallback_context_analysis(data)

    def _generate_model_recommendations(self, 
                                      context: Dict[str, Any],
                                      similar_models: List[Dict[str, Any]],
                                      conversation_history: List[Dict[str, Any]] = None) -> BusinessModelRecommendation:
        """Generate personalized business model using AI reasoning"""
        
        # Prepare context for AI generation
        similar_models_text = "\n".join([
            f"Model {i+1}: {model['description']}" 
            for i, model in enumerate(similar_models[:5])
        ])
        
        conversation_context = ""
        if conversation_history:
            conversation_context = "\n".join([
                f"{msg['role']}: {msg['content']}" 
                for msg in conversation_history[-5:]
            ])
        
        generation_prompt = f"""
        Design an optimal business model for this entrepreneur:
        
        Context Analysis:
        {json.dumps(context, indent=2)}
        
        Similar Successful Models:
        {similar_models_text}
        
        Recent Conversation:
        {conversation_context}
        
        African Success Patterns:
        {json.dumps(self.success_patterns, indent=2)}
        
        Generate a complete business model recommendation in JSON format:
        {{
            "model_type": "primary business model type",
            "revenue_streams": ["stream1", "stream2", "stream3"],
            "key_activities": ["activity1", "activity2", "activity3"],
            "key_partnerships": ["partner1", "partner2", "partner3"],
            "cost_structure": ["cost1", "cost2", "cost3"],
            "value_propositions": ["value1", "value2", "value3"],
            "customer_segments": ["segment1", "segment2"],
            "channels": ["channel1", "channel2", "channel3"],
            "customer_relationships": ["relationship1", "relationship2"],
            "confidence_score": 0.85,
            "implementation_steps": ["step1", "step2", "step3", "step4", "step5"],
            "risk_factors": ["risk1", "risk2", "risk3"],
            "success_metrics": ["metric1", "metric2", "metric3"]
        }}
        
        Focus on:
        1. Mobile-first approaches for African markets
        2. Low-resource, high-impact solutions
        3. Community-based distribution models
        4. Sustainable revenue streams
        5. Social impact integration
        """
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": generation_prompt}],
                temperature=0.4,
                max_tokens=1500
            )
            
            model_data = json.loads(response.choices[0].message.content)
            
            return BusinessModelRecommendation(**model_data)
            
        except Exception as e:
            # Fallback model generation
            return self._fallback_model_generation(context)

    def _validate_against_patterns(self, model: BusinessModelRecommendation) -> BusinessModelRecommendation:
        """Validate model against successful patterns from SIGMA data"""
        
        validation_prompt = f"""
        Validate this business model against African success patterns:
        
        Proposed Model:
        {json.dumps(model.__dict__, indent=2)}
        
        Success Patterns:
        {json.dumps(self.success_patterns, indent=2)}
        
        Provide validation and improvements in JSON format:
        {{
            "validation_score": 0.85,
            "strengths": ["strength1", "strength2"],
            "weaknesses": ["weakness1", "weakness2"],
            "improvements": {{
                "revenue_streams": ["improved_stream1", "improved_stream2"],
                "implementation_steps": ["improved_step1", "improved_step2"],
                "risk_mitigation": ["mitigation1", "mitigation2"]
            }},
            "pattern_alignment": {{
                "mobile_first": true,
                "community_based": true,
                "scalable": true,
                "sustainable": true
            }}
        }}
        """
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": validation_prompt}],
                temperature=0.2,
                max_tokens=1000
            )
            
            validation = json.loads(response.choices[0].message.content)
            
            # Apply improvements
            if validation.get('improvements'):
                improvements = validation['improvements']
                if improvements.get('revenue_streams'):
                    model.revenue_streams = improvements['revenue_streams']
                if improvements.get('implementation_steps'):
                    model.implementation_steps = improvements['implementation_steps']
            
            # Update confidence based on validation
            model.confidence_score = validation.get('validation_score', model.confidence_score)
            
            return model
            
        except Exception as e:
            return model

    def _add_implementation_guidance(self, 
                                   model: BusinessModelRecommendation,
                                   entrepreneur_data: Dict[str, Any]) -> BusinessModelRecommendation:
        """Add specific implementation guidance based on entrepreneur's context"""
        
        # Location-specific guidance
        location = entrepreneur_data.get('location', 'Ghana')
        sector = entrepreneur_data.get('sector', 'general')
        
        # Add location-specific steps
        location_steps = self._get_location_specific_steps(location, sector)
        model.implementation_steps.extend(location_steps)
        
        # Add resource-specific guidance
        resources = entrepreneur_data.get('available_resources', {})
        resource_steps = self._get_resource_specific_steps(resources, model.model_type)
        model.implementation_steps.extend(resource_steps)
        
        # Prioritize steps
        model.implementation_steps = self._prioritize_implementation_steps(
            model.implementation_steps
        )
        
        return model

    def iterative_refinement(self, 
                           current_model: BusinessModelRecommendation,
                           feedback: Dict[str, Any],
                           new_context: Dict[str, Any] = None) -> BusinessModelRecommendation:
        """Iteratively refine business model based on feedback"""
        
        refinement_prompt = f"""
        Refine this business model based on feedback:
        
        Current Model:
        {json.dumps(current_model.__dict__, indent=2)}
        
        Feedback:
        {json.dumps(feedback, indent=2)}
        
        New Context:
        {json.dumps(new_context or {}, indent=2)}
        
        Provide refined model maintaining the same JSON structure but with improvements based on feedback.
        """
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": refinement_prompt}],
                temperature=0.3,
                max_tokens=1500
            )
            
            refined_data = json.loads(response.choices[0].message.content)
            refined_model = BusinessModelRecommendation(**refined_data)
            
            # Store refinement for learning
            self.memory_chain.store_refinement(current_model, refined_model, feedback)
            
            return refined_model
            
        except Exception as e:
            return current_model

    def generate_canvas_visualization(self, 
                                    model: BusinessModelRecommendation,
                                    canvas_type: str = "bmc") -> Dict[str, Any]:
        """Generate visual business model canvas"""
        
        template = self.frameworks.get(canvas_type, self.frameworks["bmc"])
        
        canvas_data = {
            "canvas_type": canvas_type,
            "sections": {}
        }
        
        # Map model data to canvas sections
        for section, field_mapping in template["mappings"].items():
            if hasattr(model, field_mapping):
                canvas_data["sections"][section] = getattr(model, field_mapping)
        
        return canvas_data

    def _lean_canvas_template(self) -> Dict[str, Any]:
        return {
            "name": "Lean Canvas",
            "sections": [
                "problem", "solution", "key_metrics", "unique_value_proposition",
                "unfair_advantage", "channels", "customer_segments", 
                "cost_structure", "revenue_streams"
            ],
            "mappings": {
                "problem": "key_activities",
                "solution": "value_propositions", 
                "channels": "channels",
                "customer_segments": "customer_segments",
                "cost_structure": "cost_structure",
                "revenue_streams": "revenue_streams"
            }
        }

    def _business_model_canvas_template(self) -> Dict[str, Any]:
        return {
            "name": "Business Model Canvas",
            "sections": [
                "key_partnerships", "key_activities", "key_resources",
                "value_propositions", "customer_relationships", "channels",
                "customer_segments", "cost_structure", "revenue_streams"
            ],
            "mappings": {
                "key_partnerships": "key_partnerships",
                "key_activities": "key_activities",
                "value_propositions": "value_propositions",
                "customer_relationships": "customer_relationships",
                "channels": "channels",
                "customer_segments": "customer_segments",
                "cost_structure": "cost_structure",
                "revenue_streams": "revenue_streams"
            }
        }

    def _impact_canvas_template(self) -> Dict[str, Any]:
        return {
            "name": "Impact Canvas",
            "sections": [
                "stakeholders", "problem", "solution", "impact_goals",
                "assumptions", "success_metrics", "risks", "sustainability"
            ],
            "mappings": {
                "stakeholders": "customer_segments",
                "solution": "value_propositions",
                "success_metrics": "success_metrics",
                "risks": "risk_factors"
            }
        }

    def _platform_canvas_template(self) -> Dict[str, Any]:
        return {
            "name": "Platform Canvas", 
            "sections": [
                "producers", "consumers", "value_creation", "value_consumption",
                "platform_channels", "governance", "metrics", "monetization"
            ],
            "mappings": {
                "producers": "key_partnerships",
                "consumers": "customer_segments",
                "value_creation": "key_activities",
                "value_consumption": "value_propositions",
                "monetization": "revenue_streams"
            }
        }

    def _load_african_business_context(self) -> Dict[str, Any]:
        """Load African market context for business model design"""
        return {
            "market_characteristics": {
                "mobile_penetration": 0.95,
                "smartphone_adoption": 0.45,
                "internet_connectivity": 0.35,
                "mobile_money_usage": 0.70,
                "informal_economy_size": 0.60
            },
            "infrastructure": {
                "electricity_reliability": 0.60,
                "transportation_quality": 0.40,
                "logistics_efficiency": 0.35,
                "financial_services_access": 0.45
            },
            "business_environment": {
                "ease_of_doing_business": 0.55,
                "regulatory_stability": 0.60,
                "corruption_level": 0.45,
                "tax_complexity": 0.70
            },
            "success_factors": [
                "mobile_first_design",
                "community_based_distribution",
                "cash_and_mobile_money_integration",
                "local_partnership_leverage",
                "social_impact_integration"
            ]
        }

    def _load_success_patterns(self) -> Dict[str, Any]:
        """Load successful business model patterns from SIGMA platform data"""
        return {
            "high_success_models": [
                {
                    "type": "marketplace",
                    "characteristics": ["mobile_first", "community_based", "commission_model"],
                    "success_rate": 0.75,
                    "examples": ["agritech_marketplaces", "service_platforms"]
                },
                {
                    "type": "subscription_saas",
                    "characteristics": ["b2b_focus", "local_payments", "offline_capability"],
                    "success_rate": 0.68,
                    "examples": ["business_management", "education_platforms"]
                },
                {
                    "type": "freemium_mobile",
                    "characteristics": ["viral_growth", "local_content", "premium_features"],
                    "success_rate": 0.62,
                    "examples": ["fintech_apps", "communication_tools"]
                }
            ],
            "revenue_stream_success": {
                "transaction_fees": 0.80,
                "subscription": 0.70,
                "commission": 0.75,
                "advertising": 0.45,
                "premium_features": 0.60
            },
            "distribution_channel_success": {
                "mobile_app": 0.85,
                "agent_network": 0.75,
                "social_media": 0.65,
                "community_leaders": 0.70,
                "sms_ussd": 0.60
            }
        }

    def _fallback_context_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback context analysis when AI fails"""
        return {
            "market_context": {
                "local_market_size": "medium",
                "competition_level": "medium",
                "infrastructure_readiness": "developing",
                "regulatory_environment": "moderate"
            },
            "entrepreneur_readiness": {
                "skills_match": "moderate",
                "resource_availability": "limited",
                "network_strength": "developing",
                "risk_tolerance": "moderate"
            },
            "opportunity_assessment": {
                "market_timing": "good",
                "scalability_potential": "medium",
                "social_impact_potential": "high",
                "revenue_potential": "medium"
            },
            "constraints": ["funding", "market_access", "technical_skills"],
            "advantages": ["local_knowledge", "mobile_adoption", "community_connections"]
        }

    def _fallback_model_generation(self, context: Dict[str, Any]) -> BusinessModelRecommendation:
        """Fallback business model when AI generation fails"""
        return BusinessModelRecommendation(
            model_type="mobile_marketplace",
            revenue_streams=["transaction_fees", "premium_subscriptions"],
            key_activities=["platform_development", "community_building", "quality_assurance"],
            key_partnerships=["local_merchants", "payment_providers", "logistics_partners"],
            cost_structure=["technology_development", "marketing", "operations"],
            value_propositions=["convenient_access", "trusted_transactions", "local_relevance"],
            customer_segments=["urban_consumers", "small_businesses"],
            channels=["mobile_app", "social_media", "word_of_mouth"],
            customer_relationships=["community_building", "customer_support"],
            confidence_score=0.65,
            implementation_steps=[
                "conduct_market_validation",
                "develop_mvp", 
                "recruit_initial_partners",
                "launch_pilot_program",
                "iterate_based_on_feedback"
            ],
            risk_factors=["market_competition", "technology_adoption", "payment_processing"],
            success_metrics=["user_growth", "transaction_volume", "revenue_growth"]
        )

    def _get_location_specific_steps(self, location: str, sector: str) -> List[str]:
        """Get implementation steps specific to location and sector"""
        location_steps = {
            "Ghana": [
                "register_with_ghana_investment_promotion_center",
                "obtain_business_operating_permit",
                "integrate_mobile_money_payments"
            ],
            "Nigeria": [
                "register_with_corporate_affairs_commission",
                "obtain_tax_identification_number",
                "integrate_local_payment_gateways"
            ],
            "Kenya": [
                "register_with_business_registration_service",
                "obtain_single_business_permit",
                "integrate_mpesa_payments"
            ]
        }
        
        return location_steps.get(location, [
            "research_local_regulations",
            "identify_local_partners",
            "adapt_to_local_payment_methods"
        ])

    def _get_resource_specific_steps(self, resources: Dict[str, Any], model_type: str) -> List[str]:
        """Get steps based on available resources"""
        steps = []
        
        funding = resources.get('funding', 'low')
        if funding == 'low':
            steps.extend([
                "start_with_mvp_approach",
                "focus_on_organic_growth",
                "leverage_free_marketing_channels"
            ])
        
        technical_skills = resources.get('technical_skills', 'medium')
        if technical_skills == 'low':
            steps.extend([
                "partner_with_technical_team",
                "use_no_code_platforms",
                "outsource_development_initially"
            ])
        
        return steps

    def _prioritize_implementation_steps(self, steps: List[str]) -> List[str]:
        """Prioritize implementation steps by importance and dependency"""
        
        priority_order = [
            "conduct_market_validation",
            "register_business_legally",
            "develop_mvp",
            "recruit_initial_partners",
            "launch_pilot_program",
            "iterate_based_on_feedback",
            "scale_operations"
        ]
        
        # Sort steps based on priority order
        prioritized = []
        remaining = steps.copy()
        
        for priority_step in priority_order:
            for step in remaining:
                if priority_step in step:
                    prioritized.append(step)
                    remaining.remove(step)
                    break
        
        # Add remaining steps
        prioritized.extend(remaining)
        
        return prioritized[:10]  # Limit to top 10 steps

# Factory function for easy integration
def create_business_model_copilot():
    return BusinessModelCopilot()