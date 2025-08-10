import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
import sqlite3
import pickle
from pathlib import Path
from config.settings import settings

@dataclass
class InteractionRecord:
    interaction_id: str
    user_id: str
    timestamp: datetime
    query: str
    response: str
    context: Dict[str, Any]
    feedback: Optional[Dict[str, Any]] = None
    outcome: Optional[str] = None
    tools_used: List[str] = None
    performance_metrics: Dict[str, float] = None

@dataclass
class LearningPattern:
    pattern_id: str
    pattern_type: str
    description: str
    frequency: int
    confidence: float
    contexts: List[Dict[str, Any]]
    success_indicators: List[str]
    failure_indicators: List[str]
    recommended_actions: List[str]
    last_updated: datetime

@dataclass
class FeedbackInsight:
    insight_id: str
    insight_type: str
    description: str
    evidence: List[str]
    impact_level: str
    actionability: str
    applicable_scenarios: List[str]
    confidence_score: float

class FeedbackMemoryChain:
    """
    Advanced memory system with feedback loops for continuous learning
    Stores and analyzes interaction patterns to improve AI responses
    Implements memory consolidation and pattern recognition
    """
    
    def __init__(self):
        self.db_path = Path(settings.CHROMA_PERSIST_DIRECTORY) / "memory_chain.db"
        self.cache_size = 1000
        self.interaction_cache = deque(maxlen=self.cache_size)
        self.pattern_cache = {}
        
        # Initialize database
        self._init_database()
        
        # Memory consolidation settings
        self.consolidation_threshold = 100  # Minimum interactions before pattern analysis
        self.pattern_confidence_threshold = 0.7
        self.feedback_weight = 0.3
        self.outcome_weight = 0.7
        
        # Learning categories
        self.learning_categories = {
            "successful_strategies": [],
            "failed_approaches": [],
            "user_preferences": {},
            "context_patterns": {},
            "tool_effectiveness": {},
            "response_quality_patterns": {}
        }

    def store_interaction(self, 
                         user_id: str,
                         query: str,
                         response: str,
                         context: Dict[str, Any],
                         tools_used: List[str] = None) -> str:
        """Store interaction for learning and analysis"""
        
        interaction_id = f"int_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{user_id[:8]}"
        
        interaction = InteractionRecord(
            interaction_id=interaction_id,
            user_id=user_id,
            timestamp=datetime.now(),
            query=query,
            response=response,
            context=context,
            tools_used=tools_used or [],
            performance_metrics=self._calculate_interaction_metrics(query, response, context)
        )
        
        # Store in cache and database
        self.interaction_cache.append(interaction)
        self._store_interaction_db(interaction)
        
        # Trigger learning if threshold reached
        if len(self.interaction_cache) >= self.consolidation_threshold:
            self._trigger_memory_consolidation()
        
        return interaction_id

    def store_feedback(self, 
                      interaction_id: str,
                      feedback: Dict[str, Any]) -> bool:
        """Store user feedback for specific interaction"""
        
        try:
            # Update interaction with feedback
            updated = self._update_interaction_feedback(interaction_id, feedback)
            
            if updated:
                # Immediate learning from feedback
                self._process_immediate_feedback(interaction_id, feedback)
                return True
            
            return False
            
        except Exception as e:
            print(f"Error storing feedback: {e}")
            return False

    def store_outcome(self, 
                     interaction_id: str,
                     outcome: str,
                     success_metrics: Dict[str, float] = None) -> bool:
        """Store outcome/result of interaction for learning"""
        
        try:
            # Update interaction with outcome
            outcome_data = {
                "outcome": outcome,
                "success_metrics": success_metrics or {},
                "recorded_at": datetime.now().isoformat()
            }
            
            updated = self._update_interaction_outcome(interaction_id, outcome_data)
            
            if updated:
                # Learn from outcome
                self._process_outcome_learning(interaction_id, outcome_data)
                return True
            
            return False
            
        except Exception as e:
            print(f"Error storing outcome: {e}")
            return False

    def retrieve_relevant_memories(self, 
                                  query: str,
                                  context: Dict[str, Any],
                                  limit: int = 10) -> List[InteractionRecord]:
        """Retrieve relevant past interactions for context"""
        
        # Calculate similarity with cached interactions
        relevant_interactions = []
        
        for interaction in self.interaction_cache:
            similarity = self._calculate_similarity(query, context, interaction)
            
            if similarity > 0.6:  # Similarity threshold
                relevant_interactions.append((similarity, interaction))
        
        # Sort by similarity and return top matches
        relevant_interactions.sort(key=lambda x: x[0], reverse=True)
        return [interaction for _, interaction in relevant_interactions[:limit]]

    def get_learned_patterns(self, 
                           pattern_type: str = None,
                           context_filter: Dict[str, Any] = None) -> List[LearningPattern]:
        """Retrieve learned patterns for application"""
        
        patterns = self._load_patterns_from_db(pattern_type)
        
        if context_filter:
            patterns = self._filter_patterns_by_context(patterns, context_filter)
        
        # Sort by confidence and frequency
        patterns.sort(key=lambda p: (p.confidence, p.frequency), reverse=True)
        
        return patterns

    def generate_response_guidance(self, 
                                 query: str,
                                 context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate guidance for response based on learned patterns"""
        
        guidance = {
            "recommended_approaches": [],
            "approaches_to_avoid": [],
            "context_considerations": [],
            "tool_recommendations": [],
            "response_style_suggestions": [],
            "confidence_level": 0.0
        }
        
        # Get relevant memories
        relevant_memories = self.retrieve_relevant_memories(query, context)
        
        if not relevant_memories:
            return guidance
        
        # Analyze successful patterns
        successful_patterns = [
            m for m in relevant_memories 
            if m.feedback and m.feedback.get('satisfaction', 0) > 0.7
        ]
        
        # Analyze failed patterns
        failed_patterns = [
            m for m in relevant_memories
            if m.feedback and m.feedback.get('satisfaction', 0) < 0.4
        ]
        
        # Extract successful approaches
        if successful_patterns:
            guidance["recommended_approaches"] = self._extract_successful_approaches(successful_patterns)
            guidance["tool_recommendations"] = self._extract_tool_recommendations(successful_patterns)
        
        # Extract approaches to avoid
        if failed_patterns:
            guidance["approaches_to_avoid"] = self._extract_failed_approaches(failed_patterns)
        
        # Context-specific guidance
        guidance["context_considerations"] = self._extract_context_guidance(
            relevant_memories, context
        )
        
        # Calculate confidence based on evidence
        guidance["confidence_level"] = self._calculate_guidance_confidence(
            len(successful_patterns), len(failed_patterns), len(relevant_memories)
        )
        
        return guidance

    def analyze_conversation_patterns(self, 
                                    user_id: str,
                                    lookback_days: int = 30) -> Dict[str, Any]:
        """Analyze conversation patterns for specific user"""
        
        # Retrieve user's recent interactions
        user_interactions = self._get_user_interactions(user_id, lookback_days)
        
        if not user_interactions:
            return {"message": "Insufficient interaction history"}
        
        analysis = {
            "interaction_frequency": len(user_interactions) / lookback_days,
            "common_query_types": self._analyze_query_types(user_interactions),
            "satisfaction_trends": self._analyze_satisfaction_trends(user_interactions),
            "tool_usage_patterns": self._analyze_tool_usage(user_interactions),
            "context_preferences": self._analyze_context_preferences(user_interactions),
            "learning_progression": self._analyze_learning_progression(user_interactions),
            "recommendations": self._generate_user_recommendations(user_interactions)
        }
        
        return analysis

    def _init_database(self):
        """Initialize SQLite database for persistent memory storage"""
        
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            # Interactions table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS interactions (
                    interaction_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    timestamp TEXT,
                    query TEXT,
                    response TEXT,
                    context TEXT,
                    tools_used TEXT,
                    feedback TEXT,
                    outcome TEXT,
                    performance_metrics TEXT
                )
            """)
            
            # Patterns table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS patterns (
                    pattern_id TEXT PRIMARY KEY,
                    pattern_type TEXT,
                    description TEXT,
                    frequency INTEGER,
                    confidence REAL,
                    contexts TEXT,
                    success_indicators TEXT,
                    failure_indicators TEXT,
                    recommended_actions TEXT,
                    last_updated TEXT
                )
            """)
            
            # Insights table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS insights (
                    insight_id TEXT PRIMARY KEY,
                    insight_type TEXT,
                    description TEXT,
                    evidence TEXT,
                    impact_level TEXT,
                    actionability TEXT,
                    applicable_scenarios TEXT,
                    confidence_score REAL,
                    created_at TEXT
                )
            """)
            
            conn.commit()

    def _store_interaction_db(self, interaction: InteractionRecord):
        """Store interaction in database"""
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO interactions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                interaction.interaction_id,
                interaction.user_id,
                interaction.timestamp.isoformat(),
                interaction.query,
                interaction.response,
                json.dumps(interaction.context),
                json.dumps(interaction.tools_used or []),
                json.dumps(interaction.feedback) if interaction.feedback else None,
                interaction.outcome,
                json.dumps(interaction.performance_metrics or {})
            ))

    def _calculate_interaction_metrics(self, 
                                     query: str,
                                     response: str,
                                     context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate performance metrics for interaction"""
        
        metrics = {
            "response_length": len(response),
            "query_complexity": self._estimate_query_complexity(query),
            "context_richness": len(context),
            "response_time": context.get('response_time', 0.0),
            "tools_used_count": len(context.get('tools_used', [])),
            "confidence_level": context.get('confidence', 0.5)
        }
        
        return metrics

    def _estimate_query_complexity(self, query: str) -> float:
        """Estimate query complexity based on various factors"""
        
        complexity_score = 0.0
        
        # Length factor
        complexity_score += min(len(query.split()) / 20, 1.0) * 0.3
        
        # Question complexity indicators
        complexity_indicators = [
            'analyze', 'compare', 'evaluate', 'synthesize', 'recommend',
            'strategy', 'framework', 'assessment', 'optimization'
        ]
        
        indicator_count = sum(1 for indicator in complexity_indicators if indicator in query.lower())
        complexity_score += min(indicator_count / 3, 1.0) * 0.4
        
        # Multiple questions
        question_count = query.count('?') + query.count('how') + query.count('what') + query.count('why')
        complexity_score += min(question_count / 3, 1.0) * 0.3
        
        return min(complexity_score, 1.0)

    def _trigger_memory_consolidation(self):
        """Trigger memory consolidation process"""
        
        try:
            # Analyze recent interactions for patterns
            recent_interactions = list(self.interaction_cache)[-self.consolidation_threshold:]
            
            # Identify new patterns
            new_patterns = self._identify_patterns(recent_interactions)
            
            # Store patterns
            for pattern in new_patterns:
                self._store_pattern_db(pattern)
            
            # Generate insights
            new_insights = self._generate_insights(recent_interactions, new_patterns)
            
            # Store insights
            for insight in new_insights:
                self._store_insight_db(insight)
            
            # Update learning categories
            self._update_learning_categories(recent_interactions)
            
        except Exception as e:
            print(f"Memory consolidation error: {e}")

    def _identify_patterns(self, interactions: List[InteractionRecord]) -> List[LearningPattern]:
        """Identify patterns in interaction data"""
        
        patterns = []
        
        # Group interactions by similarity
        interaction_groups = self._group_similar_interactions(interactions)
        
        for group_id, group_interactions in interaction_groups.items():
            if len(group_interactions) < 3:  # Minimum for pattern
                continue
            
            # Analyze pattern characteristics
            pattern = self._analyze_interaction_group(group_interactions)
            
            if pattern and pattern.confidence > self.pattern_confidence_threshold:
                patterns.append(pattern)
        
        return patterns

    def _group_similar_interactions(self, 
                                  interactions: List[InteractionRecord]) -> Dict[str, List[InteractionRecord]]:
        """Group similar interactions for pattern analysis"""
        
        groups = defaultdict(list)
        
        for interaction in interactions:
            # Simple grouping by query type and context
            group_key = self._generate_group_key(interaction)
            groups[group_key].append(interaction)
        
        return groups

    def _generate_group_key(self, interaction: InteractionRecord) -> str:
        """Generate grouping key for interaction"""
        
        query_type = self._classify_query_type(interaction.query)
        context_type = interaction.context.get('type', 'general')
        tools_signature = '_'.join(sorted(interaction.tools_used or []))
        
        return f"{query_type}_{context_type}_{tools_signature}"

    def _classify_query_type(self, query: str) -> str:
        """Classify query into basic types"""
        
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['assess', 'evaluate', 'score']):
            return 'assessment'
        elif any(word in query_lower for word in ['fund', 'investment', 'capital']):
            return 'funding'
        elif any(word in query_lower for word in ['register', 'legal', 'compliance']):
            return 'regulatory'
        elif any(word in query_lower for word in ['market', 'research', 'analysis']):
            return 'research'
        elif any(word in query_lower for word in ['how', 'guide', 'help']):
            return 'guidance'
        else:
            return 'general'

    def _analyze_interaction_group(self, 
                                 group_interactions: List[InteractionRecord]) -> Optional[LearningPattern]:
        """Analyze a group of interactions to identify patterns"""
        
        # Calculate success rate
        successful_interactions = [
            i for i in group_interactions
            if i.feedback and i.feedback.get('satisfaction', 0) > 0.7
        ]
        
        success_rate = len(successful_interactions) / len(group_interactions)
        
        if success_rate < 0.3:  # Low success rate pattern
            return None
        
        # Extract common characteristics
        common_tools = self._find_common_tools(group_interactions)
        common_contexts = self._find_common_contexts(group_interactions)
        success_indicators = self._extract_success_indicators(successful_interactions)
        
        # Create pattern
        pattern_id = f"pattern_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        pattern = LearningPattern(
            pattern_id=pattern_id,
            pattern_type=self._classify_query_type(group_interactions[0].query),
            description=f"Pattern for {len(group_interactions)} similar interactions",
            frequency=len(group_interactions),
            confidence=success_rate,
            contexts=common_contexts,
            success_indicators=success_indicators,
            failure_indicators=[],
            recommended_actions=self._generate_pattern_recommendations(successful_interactions),
            last_updated=datetime.now()
        )
        
        return pattern

    def _find_common_tools(self, interactions: List[InteractionRecord]) -> List[str]:
        """Find commonly used tools across interactions"""
        
        tool_counts = defaultdict(int)
        
        for interaction in interactions:
            for tool in interaction.tools_used or []:
                tool_counts[tool] += 1
        
        # Return tools used in >50% of interactions
        threshold = len(interactions) * 0.5
        return [tool for tool, count in tool_counts.items() if count > threshold]

    def _find_common_contexts(self, interactions: List[InteractionRecord]) -> List[Dict[str, Any]]:
        """Find common context patterns"""
        
        contexts = []
        
        # Extract common context elements
        context_elements = defaultdict(list)
        
        for interaction in interactions:
            for key, value in interaction.context.items():
                context_elements[key].append(value)
        
        # Find frequently occurring context patterns
        for key, values in context_elements.items():
            if len(set(values)) < len(values) * 0.7:  # Some repetition
                most_common = max(set(values), key=values.count)
                contexts.append({key: most_common})
        
        return contexts

    def store_credit_assessment(self, 
                               applicant_data: Dict[str, Any],
                               assessment: Any) -> bool:
        """Store credit assessment for learning"""
        
        try:
            interaction_id = f"credit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Create pseudo-interaction for credit assessment
            credit_interaction = InteractionRecord(
                interaction_id=interaction_id,
                user_id=applicant_data.get('id', 'unknown'),
                timestamp=datetime.now(),
                query=f"Credit assessment for {applicant_data.get('business_type', 'business')}",
                response=f"Credit score: {assessment.credit_score}",
                context={
                    "assessment_type": "credit",
                    "applicant_sector": applicant_data.get('sector'),
                    "location": applicant_data.get('location'),
                    "amount_requested": applicant_data.get('loan_amount'),
                    "assessment_result": asdict(assessment)
                }
            )
            
            self._store_interaction_db(credit_interaction)
            return True
            
        except Exception as e:
            print(f"Error storing credit assessment: {e}")
            return False

    def store_refinement(self, 
                        original_model: Any,
                        refined_model: Any,
                        feedback: Dict[str, Any]) -> bool:
        """Store model refinement for learning"""
        
        try:
            interaction_id = f"refinement_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            refinement_interaction = InteractionRecord(
                interaction_id=interaction_id,
                user_id=feedback.get('user_id', 'unknown'),
                timestamp=datetime.now(),
                query="Business model refinement",
                response="Model updated based on feedback",
                context={
                    "refinement_type": "business_model",
                    "original_confidence": getattr(original_model, 'confidence_score', 0),
                    "refined_confidence": getattr(refined_model, 'confidence_score', 0),
                    "feedback_type": feedback.get('type'),
                    "changes_made": self._calculate_model_changes(original_model, refined_model)
                },
                feedback=feedback
            )
            
            self._store_interaction_db(refinement_interaction)
            return True
            
        except Exception as e:
            print(f"Error storing refinement: {e}")
            return False

    def _calculate_model_changes(self, original: Any, refined: Any) -> Dict[str, Any]:
        """Calculate changes between model versions"""
        
        changes = {}
        
        if hasattr(original, '__dict__') and hasattr(refined, '__dict__'):
            original_dict = original.__dict__
            refined_dict = refined.__dict__
            
            for key in original_dict:
                if key in refined_dict:
                    if original_dict[key] != refined_dict[key]:
                        changes[key] = {
                            "original": original_dict[key],
                            "refined": refined_dict[key]
                        }
        
        return changes

# Factory function
def create_feedback_memory_chain():
    return FeedbackMemoryChain()