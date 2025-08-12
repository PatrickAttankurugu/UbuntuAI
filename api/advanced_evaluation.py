"""
Advanced RAG Evaluation System for UbuntuAI - Industry Standard Implementation
Implements comprehensive evaluation, A/B testing, and production monitoring
"""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import asyncio
from abc import ABC, abstractmethod

# Advanced evaluation libraries
try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness, answer_relevancy, context_precision, context_recall,
        answer_similarity, answer_correctness, context_relevancy
    )
    from ragas.langchain import RagasCallbackHandler
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Monitoring and observability
try:
    from langfuse import Langfuse
    from langsmith import trace
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False

from datasets import Dataset
from config.settings import settings
from api.llm_providers import llm_manager

logger = logging.getLogger(__name__)

@dataclass
class AdvancedEvaluationResult:
    """Comprehensive evaluation result with industry-standard metrics"""
    query: str
    answer: str
    contexts: List[str]
    ground_truth: Optional[str]
    
    # RAGAS metrics
    faithfulness: Optional[float]
    answer_relevancy: Optional[float]
    context_precision: Optional[float]
    context_recall: Optional[float]
    answer_similarity: Optional[float]
    answer_correctness: Optional[float]
    context_relevancy: Optional[float]
    
    # Custom metrics
    ghana_specificity: float
    sector_relevance: float
    actionability_score: float
    cultural_appropriateness: float
    
    # Performance metrics
    response_time: float
    token_usage: Dict[str, int]
    cost_estimate: float
    
    # Metadata
    timestamp: str
    model_used: str
    evaluation_method: str
    confidence_interval: Tuple[float, float]

@dataclass
class ABTestResult:
    """A/B test result for RAG system comparison"""
    test_id: str
    variant_a: str
    variant_b: str
    metric: str
    variant_a_score: float
    variant_b_score: float
    statistical_significance: float
    winner: str
    confidence_level: float
    sample_size: int
    timestamp: str

class AdvancedRAGASEvaluator:
    """Advanced RAGAS-based evaluation system"""
    
    def __init__(self):
        self.is_available = RAGAS_AVAILABLE
        self.metrics_config = self._setup_metrics()
        self.evaluation_history = []
        
        if not self.is_available:
            logger.warning("RAGAS evaluation disabled - install ragas package to enable")
    
    def _setup_metrics(self) -> Dict[str, Any]:
        """Setup comprehensive RAGAS metrics"""
        if not self.is_available:
            return {}
        
        available_metrics = {
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_precision": context_precision,
            "context_recall": context_recall,
            "answer_similarity": answer_similarity,
            "answer_correctness": answer_correctness,
            "context_relevancy": context_relevancy
        }
        
        # Select metrics based on settings
        selected_metrics = []
        for metric_name in settings.RAGAS_METRICS:
            if metric_name in available_metrics:
                selected_metrics.append(available_metrics[metric_name])
            else:
                logger.warning(f"Unknown RAGAS metric: {metric_name}")
        
        return {
            "metrics": selected_metrics,
            "llm": llm_manager.get_langchain_llm() if llm_manager else None
        }
    
    def evaluate_comprehensive(self, 
                             query: str,
                             answer: str,
                             contexts: List[str],
                             ground_truth: Optional[str] = None,
                             metadata: Dict[str, Any] = None) -> AdvancedEvaluationResult:
        """Perform comprehensive RAG evaluation"""
        
        if not self.is_available:
            return self._create_advanced_fallback_result(query, answer, contexts, ground_truth, metadata)
        
        try:
            # Prepare data for RAGAS
            eval_data = {
                "question": [query],
                "answer": [answer],
                "contexts": [contexts]
            }
            
            if ground_truth:
                eval_data["ground_truth"] = [ground_truth]
            
            # Create dataset
            dataset = Dataset.from_dict(eval_data)
            
            # Run evaluation
            result = evaluate(
                dataset,
                metrics=self.metrics_config["metrics"],
                llm=self.metrics_config["llm"]
            )
            
            # Extract RAGAS metrics
            ragas_metrics = {}
            for metric_name, score in result.items():
                if isinstance(score, (int, float)):
                    ragas_metrics[metric_name] = float(score)
                elif hasattr(score, 'tolist'):
                    ragas_metrics[metric_name] = float(score.tolist()[0])
            
            # Calculate custom metrics
            custom_metrics = self._calculate_custom_metrics(query, answer, contexts, metadata)
            
            # Create comprehensive result
            evaluation_result = AdvancedEvaluationResult(
                query=query,
                answer=answer,
                contexts=contexts,
                ground_truth=ground_truth,
                
                # RAGAS metrics
                faithfulness=ragas_metrics.get('faithfulness'),
                answer_relevancy=ragas_metrics.get('answer_relevancy'),
                context_precision=ragas_metrics.get('context_precision'),
                context_recall=ragas_metrics.get('context_recall'),
                answer_similarity=ragas_metrics.get('answer_similarity'),
                answer_correctness=ragas_metrics.get('answer_correctness'),
                context_relevancy=ragas_metrics.get('context_relevancy'),
                
                # Custom metrics
                ghana_specificity=custom_metrics['ghana_specificity'],
                sector_relevance=custom_metrics['sector_relevance'],
                actionability_score=custom_metrics['actionability_score'],
                cultural_appropriateness=custom_metrics['cultural_appropriateness'],
                
                # Performance metrics
                response_time=metadata.get('response_time', 0.0) if metadata else 0.0,
                token_usage=metadata.get('token_usage', {}) if metadata else {},
                cost_estimate=metadata.get('cost_estimate', 0.0) if metadata else 0.0,
                
                # Metadata
                timestamp=datetime.now().isoformat(),
                model_used=metadata.get('model_used', 'unknown') if metadata else 'unknown',
                evaluation_method="ragas_comprehensive",
                confidence_interval=self._calculate_confidence_interval(ragas_metrics)
            )
            
            self.evaluation_history.append(evaluation_result)
            logger.info(f"Comprehensive evaluation completed - Faithfulness: {ragas_metrics.get('faithfulness', 'N/A')}")
            
            return evaluation_result
            
        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {e}")
            return self._create_advanced_fallback_result(query, answer, contexts, ground_truth, metadata)
    
    def _calculate_custom_metrics(self, 
                                query: str,
                                answer: str,
                                contexts: List[str],
                                metadata: Dict[str, Any]) -> Dict[str, float]:
        """Calculate custom Ghanaian startup ecosystem metrics"""
        
        metrics = {}
        
        # Ghana specificity
        ghana_keywords = [
            "ghana", "ghanian", "accra", "kumasi", "tamale", "tema",
            "bank of ghana", "gipc", "ghana enterprise agency", "mofa"
        ]
        ghana_mentions = sum(1 for keyword in ghana_keywords if keyword.lower() in answer.lower())
        metrics['ghana_specificity'] = min(ghana_mentions / 5.0, 1.0)  # Normalize to 0-1
        
        # Sector relevance
        sector_keywords = {
            "fintech": ["fintech", "financial", "banking", "mobile money", "digital payments"],
            "agritech": ["agritech", "agriculture", "farming", "crop", "livestock"],
            "healthtech": ["healthtech", "healthcare", "medical", "pharmaceutical"]
        }
        
        user_sector = metadata.get('user_sector', 'general') if metadata else 'general'
        if user_sector in sector_keywords:
            sector_mentions = sum(1 for keyword in sector_keywords[user_sector] if keyword.lower() in answer.lower())
            metrics['sector_relevance'] = min(sector_mentions / 3.0, 1.0)
        else:
            metrics['sector_relevance'] = 0.5
        
        # Actionability
        action_words = [
            "should", "must", "need to", "require", "apply", "register",
            "contact", "visit", "submit", "follow", "implement", "consider"
        ]
        action_mentions = sum(1 for word in action_words if word.lower() in answer.lower())
        metrics['actionability_score'] = min(action_mentions / 5.0, 1.0)
        
        # Cultural appropriateness
        cultural_indicators = [
            "ghanaian", "local", "regional", "cultural", "traditional",
            "community", "partnership", "collaboration", "local knowledge"
        ]
        cultural_mentions = sum(1 for indicator in cultural_indicators if indicator.lower() in answer.lower())
        metrics['cultural_appropriateness'] = min(cultural_mentions / 3.0, 1.0)
        
        return metrics
    
    def _calculate_confidence_interval(self, metrics: Dict[str, float]) -> Tuple[float, float]:
        """Calculate confidence interval for metrics"""
        
        if not metrics:
            return (0.0, 1.0)
        
        # Simple confidence interval calculation
        values = list(metrics.values())
        mean = np.mean(values)
        std = np.std(values)
        
        # 95% confidence interval
        confidence_interval = (mean - 1.96 * std, mean + 1.96 * std)
        return (max(0.0, confidence_interval[0]), min(1.0, confidence_interval[1]))
    
    def _create_advanced_fallback_result(self, 
                                       query: str,
                                       answer: str,
                                       contexts: List[str],
                                       ground_truth: Optional[str],
                                       metadata: Dict[str, Any]) -> AdvancedEvaluationResult:
        """Create advanced fallback evaluation result"""
        
        # Calculate basic metrics
        custom_metrics = self._calculate_custom_metrics(query, answer, contexts, metadata)
        
        return AdvancedEvaluationResult(
            query=query,
            answer=answer,
            contexts=contexts,
            ground_truth=ground_truth,
            
            # RAGAS metrics (not available)
            faithfulness=None,
            answer_relevancy=None,
            context_precision=None,
            context_recall=None,
            answer_similarity=None,
            answer_correctness=None,
            context_relevancy=None,
            
            # Custom metrics
            ghana_specificity=custom_metrics['ghana_specificity'],
            sector_relevance=custom_metrics['sector_relevance'],
            actionability_score=custom_metrics['actionability_score'],
            cultural_appropriateness=custom_metrics['cultural_appropriateness'],
            
            # Performance metrics
            response_time=metadata.get('response_time', 0.0) if metadata else 0.0,
            token_usage=metadata.get('token_usage', {}) if metadata else {},
            cost_estimate=metadata.get('cost_estimate', 0.0) if metadata else 0.0,
            
            # Metadata
            timestamp=datetime.now().isoformat(),
            model_used=metadata.get('model_used', 'unknown') if metadata else 'unknown',
            evaluation_method="fallback_heuristic",
            confidence_interval=(0.0, 1.0)
        )

class ABTestingFramework:
    """A/B testing framework for RAG system optimization"""
    
    def __init__(self):
        self.tests = {}
        self.results = []
        self.current_test_id = 0
    
    def create_test(self, 
                   name: str,
                   metric: str,
                   variants: List[str],
                   sample_size: int = 100) -> str:
        """Create a new A/B test"""
        
        test_id = f"test_{self.current_test_id}_{name}"
        self.current_test_id += 1
        
        self.tests[test_id] = {
            "name": name,
            "metric": metric,
            "variants": variants,
            "sample_size": sample_size,
            "results": {variant: [] for variant in variants},
            "start_time": datetime.now(),
            "status": "active"
        }
        
        logger.info(f"Created A/B test: {test_id} for {name}")
        return test_id
    
    def record_result(self, 
                     test_id: str,
                     variant: str,
                     score: float,
                     metadata: Dict[str, Any] = None):
        """Record a result for an A/B test"""
        
        if test_id not in self.tests:
            logger.warning(f"Unknown test ID: {test_id}")
            return
        
        if variant not in self.tests[test_id]["variants"]:
            logger.warning(f"Unknown variant: {variant} for test {test_id}")
            return
        
        self.tests[test_id]["results"][variant].append({
            "score": score,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        })
        
        logger.debug(f"Recorded result for test {test_id}, variant {variant}: {score}")
    
    def analyze_test(self, test_id: str) -> Optional[ABTestResult]:
        """Analyze A/B test results"""
        
        if test_id not in self.tests:
            logger.warning(f"Unknown test ID: {test_id}")
            return None
        
        test = self.tests[test_id]
        
        # Check if we have enough samples
        min_samples = min(len(results) for results in test["results"].values())
        if min_samples < 10:  # Minimum sample size for statistical significance
            logger.info(f"Test {test_id} needs more samples. Current: {min_samples}")
            return None
        
        # Calculate statistics for each variant
        variant_stats = {}
        for variant, results in test["results"].items():
            scores = [r["score"] for r in results]
            variant_stats[variant] = {
                "mean": np.mean(scores),
                "std": np.std(scores),
                "count": len(scores)
            }
        
        # Perform statistical test (t-test for two variants)
        if len(test["variants"]) == 2:
            variant_a, variant_b = test["variants"]
            stats_a = variant_stats[variant_a]
            stats_b = variant_stats[variant_b]
            
            # Calculate t-statistic and p-value
            t_stat, p_value = self._calculate_t_test(stats_a, stats_b)
            
            # Determine winner
            if p_value < 0.05:  # 95% confidence level
                winner = variant_a if stats_a["mean"] > stats_b["mean"] else variant_b
                statistical_significance = 1 - p_value
            else:
                winner = "inconclusive"
                statistical_significance = 0.0
            
            # Create result
            result = ABTestResult(
                test_id=test_id,
                variant_a=variant_a,
                variant_b=variant_b,
                metric=test["metric"],
                variant_a_score=stats_a["mean"],
                variant_b_score=stats_b["mean"],
                statistical_significance=statistical_significance,
                winner=winner,
                confidence_level=1 - p_value,
                sample_size=min_samples,
                timestamp=datetime.now().isoformat()
            )
            
            self.results.append(result)
            test["status"] = "completed"
            
            logger.info(f"A/B test {test_id} completed. Winner: {winner} (p={p_value:.4f})")
            return result
        
        return None
    
    def _calculate_t_test(self, stats_a: Dict[str, float], stats_b: Dict[str, float]) -> Tuple[float, float]:
        """Calculate t-test statistic and p-value"""
        
        try:
            from scipy import stats
            
            # This is a simplified t-test calculation
            # In production, you'd want to use proper statistical libraries
            
            # Calculate pooled standard error
            pooled_se = np.sqrt(
                (stats_a["std"]**2 / stats_a["count"]) + 
                (stats_b["std"]**2 / stats_b["count"])
            )
            
            # Calculate t-statistic
            t_stat = (stats_a["mean"] - stats_b["mean"]) / pooled_se
            
            # Calculate degrees of freedom
            df = stats_a["count"] + stats_b["count"] - 2
            
            # Calculate p-value (two-tailed test)
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
            
            return t_stat, p_value
            
        except ImportError:
            # Fallback calculation
            t_stat = (stats_a["mean"] - stats_b["mean"]) / np.sqrt(stats_a["std"]**2 + stats_b["std"]**2)
            p_value = 0.1  # Conservative estimate
            return t_stat, p_value
    
    def get_test_status(self, test_id: str) -> Dict[str, Any]:
        """Get current status of an A/B test"""
        
        if test_id not in self.tests:
            return {"error": "Unknown test ID"}
        
        test = self.tests[test_id]
        status = {
            "test_id": test_id,
            "name": test["name"],
            "metric": test["metric"],
            "variants": test["variants"],
            "status": test["status"],
            "start_time": test["start_time"].isoformat(),
            "sample_counts": {variant: len(results) for variant, results in test["results"].items()},
            "current_means": {}
        }
        
        # Calculate current means
        for variant, results in test["results"].items():
            if results:
                scores = [r["score"] for r in results]
                status["current_means"][variant] = np.mean(scores)
        
        return status

class ProductionMonitoring:
    """Production monitoring and alerting system"""
    
    def __init__(self):
        self.metrics_history = []
        self.alert_thresholds = {
            "faithfulness": 0.7,
            "answer_relevancy": 0.7,
            "context_precision": 0.6,
            "ghana_specificity": 0.6,
            "response_time": 10.0,  # seconds
            "error_rate": 0.05
        }
        self.alerts = []
        
        # Initialize monitoring if available
        self.langfuse = None
        if MONITORING_AVAILABLE and settings.USE_LANGFUSE:
            try:
                self.langfuse = Langfuse(
                    secret_key=settings.LANGFUSE_SECRET_KEY,
                    public_key=settings.LANGFUSE_PUBLIC_KEY,
                    host=settings.LANGFUSE_HOST
                )
                logger.info("LangFuse monitoring initialized for production")
            except Exception as e:
                logger.warning(f"Failed to initialize LangFuse: {e}")
    
    def record_metrics(self, metrics: Dict[str, Any]):
        """Record production metrics"""
        
        timestamp = datetime.now()
        metrics_record = {
            "timestamp": timestamp,
            "metrics": metrics
        }
        
        self.metrics_history.append(metrics_record)
        
        # Keep only last 1000 records
        if len(self.metrics_history) > 1000:
            self.metrics_history.pop(0)
        
        # Check for alerts
        self._check_alerts(metrics_record)
        
        # Send to monitoring service if available
        if self.langfuse:
            try:
                self._send_to_langfuse(metrics_record)
            except Exception as e:
                logger.error(f"Failed to send metrics to LangFuse: {e}")
    
    def _check_alerts(self, metrics_record: Dict[str, Any]):
        """Check if metrics trigger alerts"""
        
        metrics = metrics_record["metrics"]
        timestamp = metrics_record["timestamp"]
        
        for metric_name, threshold in self.alert_thresholds.items():
            if metric_name in metrics:
                value = metrics[metric_name]
                
                # Check if threshold is exceeded
                if metric_name == "response_time":
                    if value > threshold:
                        self._create_alert(metric_name, value, threshold, timestamp, "high")
                else:
                    if value < threshold:
                        self._create_alert(metric_name, value, threshold, timestamp, "low")
    
    def _create_alert(self, 
                     metric_name: str,
                     value: float,
                     threshold: float,
                     timestamp: datetime,
                     severity: str):
        """Create an alert"""
        
        alert = {
            "metric": metric_name,
            "value": value,
            "threshold": threshold,
            "timestamp": timestamp.isoformat(),
            "severity": severity,
            "message": f"{metric_name} threshold exceeded: {value} (threshold: {threshold})"
        }
        
        self.alerts.append(alert)
        logger.warning(f"ALERT: {alert['message']}")
        
        # In production, you'd send this to your alerting system
        # (Slack, PagerDuty, etc.)
    
    def _send_to_langfuse(self, metrics_record: Dict[str, Any]):
        """Send metrics to LangFuse"""
        
        try:
            # Create a trace for the metrics
            trace = self.langfuse.trace(
                name="rag_metrics",
                metadata=metrics_record["metrics"]
            )
            
            # Add metrics as spans
            for metric_name, value in metrics_record["metrics"].items():
                trace.span(
                    name=metric_name,
                    input={"value": value},
                    metadata={"timestamp": metrics_record["timestamp"].isoformat()}
                )
            
            trace.update(status="success")
            
        except Exception as e:
            logger.error(f"Failed to send to LangFuse: {e}")
    
    def get_metrics_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get metrics summary for the last N hours"""
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_metrics = [
            record for record in self.metrics_history
            if record["timestamp"] > cutoff_time
        ]
        
        if not recent_metrics:
            return {"error": "No metrics available for the specified time period"}
        
        # Calculate summary statistics
        summary = {
            "time_period_hours": hours,
            "total_requests": len(recent_metrics),
            "metrics": {}
        }
        
        # Aggregate metrics
        metric_names = set()
        for record in recent_metrics:
            metric_names.update(record["metrics"].keys())
        
        for metric_name in metric_names:
            values = [record["metrics"].get(metric_name) for record in recent_metrics if record["metrics"].get(metric_name) is not None]
            if values:
                summary["metrics"][metric_name] = {
                    "mean": np.mean(values),
                    "median": np.median(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "std": np.std(values)
                }
        
        return summary
    
    def get_alerts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get alerts from the last N hours"""
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_alerts = [
            alert for alert in self.alerts
            if datetime.fromisoformat(alert["timestamp"]) > cutoff_time
        ]
        
        return recent_alerts

# Global instances
advanced_ragas_evaluator = AdvancedRAGASEvaluator()
ab_testing_framework = ABTestingFramework()
production_monitoring = ProductionMonitoring()

# Convenience functions
def evaluate_rag_comprehensive(query: str, answer: str, contexts: List[str], **kwargs):
    """Comprehensive RAG evaluation"""
    return advanced_ragas_evaluator.evaluate_comprehensive(query, answer, contexts, **kwargs)

def create_ab_test(name: str, metric: str, variants: List[str], sample_size: int = 100):
    """Create A/B test"""
    return ab_testing_framework.create_test(name, metric, variants, sample_size)

def record_ab_result(test_id: str, variant: str, score: float, **kwargs):
    """Record A/B test result"""
    ab_testing_framework.record_result(test_id, variant, score, **kwargs)

def analyze_ab_test(test_id: str):
    """Analyze A/B test"""
    return ab_testing_framework.analyze_test(test_id)

def monitor_production_metrics(metrics: Dict[str, Any]):
    """Record production metrics"""
    production_monitoring.record_metrics(metrics) 