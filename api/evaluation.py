"""
RAGAS Evaluation Framework for UbuntuAI RAG System
Provides comprehensive evaluation metrics and monitoring
"""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import asyncio

try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        answer_similarity,
        answer_correctness
    )
    from ragas.langchain import RagasCallbackHandler
    RAGAS_AVAILABLE = True
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("RAGAS not available - evaluation features will be limited")
    RAGAS_AVAILABLE = False

from datasets import Dataset
from config.settings import settings
from api.llm_providers import llm_manager

logger = logging.getLogger(__name__)

@dataclass
class EvaluationResult:
    """Structured evaluation result"""
    query: str
    answer: str
    contexts: List[str]
    ground_truth: Optional[str]
    metrics: Dict[str, float]
    timestamp: str
    metadata: Dict[str, Any]

class RAGASEvaluator:
    """RAGAS-based evaluation system for RAG pipeline"""
    
    def __init__(self):
        self.is_available = RAGAS_AVAILABLE
        self.metrics_config = self._setup_metrics()
        self.evaluation_history = []
        
        if not self.is_available:
            logger.warning("RAGAS evaluation disabled - install ragas package to enable")
    
    def _setup_metrics(self) -> Dict[str, Any]:
        """Setup RAGAS metrics configuration"""
        if not self.is_available:
            return {}
        
        available_metrics = {
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_precision": context_precision,
            "context_recall": context_recall,
            "answer_similarity": answer_similarity,
            "answer_correctness": answer_correctness
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
    
    def evaluate_single(self, 
                       query: str,
                       answer: str,
                       contexts: List[str],
                       ground_truth: Optional[str] = None,
                       metadata: Dict[str, Any] = None) -> EvaluationResult:
        """Evaluate a single RAG response"""
        
        if not self.is_available:
            return self._create_fallback_result(query, answer, contexts, ground_truth, metadata)
        
        try:
            # Prepare data for RAGAS
            eval_data = {
                "question": [query],
                "answer": [answer],
                "contexts": [contexts]
            }
            
            # Add ground truth if available
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
            
            # Extract metrics
            metrics = {}
            for metric_name, score in result.items():
                if isinstance(score, (int, float)):
                    metrics[metric_name] = float(score)
                elif hasattr(score, 'tolist'):
                    metrics[metric_name] = float(score.tolist()[0])
            
            evaluation_result = EvaluationResult(
                query=query,
                answer=answer,
                contexts=contexts,
                ground_truth=ground_truth,
                metrics=metrics,
                timestamp=datetime.now().isoformat(),
                metadata=metadata or {}
            )
            
            self.evaluation_history.append(evaluation_result)
            logger.info(f"Evaluation completed - Metrics: {metrics}")
            
            return evaluation_result
            
        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {e}")
            return self._create_fallback_result(query, answer, contexts, ground_truth, metadata)
    
    def _create_fallback_result(self, 
                              query: str,
                              answer: str,
                              contexts: List[str],
                              ground_truth: Optional[str],
                              metadata: Dict[str, Any]) -> EvaluationResult:
        """Create fallback evaluation result when RAGAS is not available"""
        
        # Simple heuristic-based metrics
        metrics = {
            "answer_length": len(answer),
            "context_count": len(contexts),
            "avg_context_length": sum(len(ctx) for ctx in contexts) / len(contexts) if contexts else 0,
            "query_length": len(query)
        }
        
        # Basic relevancy check
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())
        word_overlap = len(query_words & answer_words) / len(query_words) if query_words else 0
        metrics["basic_relevancy"] = word_overlap
        
        return EvaluationResult(
            query=query,
            answer=answer,
            contexts=contexts,
            ground_truth=ground_truth,
            metrics=metrics,
            timestamp=datetime.now().isoformat(),
            metadata=metadata or {}
        )
    
    def evaluate_batch(self, 
                      evaluation_data: List[Dict[str, Any]]) -> List[EvaluationResult]:
        """Evaluate multiple RAG responses"""
        
        results = []
        for data in evaluation_data:
            result = self.evaluate_single(
                query=data["query"],
                answer=data["answer"],
                contexts=data["contexts"],
                ground_truth=data.get("ground_truth"),
                metadata=data.get("metadata")
            )
            results.append(result)
        
        return results
    
    def get_evaluation_summary(self, 
                             results: List[EvaluationResult] = None) -> Dict[str, Any]:
        """Get summary statistics of evaluations"""
        
        if results is None:
            results = self.evaluation_history
        
        if not results:
            return {"message": "No evaluation results available"}
        
        # Calculate aggregate metrics
        all_metrics = {}
        for result in results:
            for metric_name, value in result.metrics.items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].append(value)
        
        summary = {
            "total_evaluations": len(results),
            "avg_metrics": {},
            "metric_ranges": {},
            "latest_evaluation": results[-1].timestamp if results else None
        }
        
        for metric_name, values in all_metrics.items():
            summary["avg_metrics"][metric_name] = sum(values) / len(values)
            summary["metric_ranges"][metric_name] = {
                "min": min(values),
                "max": max(values)
            }
        
        return summary
    
    def export_results(self, 
                      filepath: str,
                      results: List[EvaluationResult] = None):
        """Export evaluation results to file"""
        
        if results is None:
            results = self.evaluation_history
        
        if not results:
            logger.warning("No evaluation results to export")
            return
        
        # Convert to serializable format
        export_data = []
        for result in results:
            export_data.append({
                "query": result.query,
                "answer": result.answer,
                "contexts": result.contexts,
                "ground_truth": result.ground_truth,
                "metrics": result.metrics,
                "timestamp": result.timestamp,
                "metadata": result.metadata
            })
        
        try:
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            logger.info(f"Exported {len(export_data)} evaluation results to {filepath}")
        except Exception as e:
            logger.error(f"Failed to export results: {e}")

class SelfReflectionEvaluator:
    """Self-reflection evaluation system for RAG responses"""
    
    def __init__(self):
        self.reflection_prompts = {
            "relevance": """
            Evaluate the relevance of this answer to the question on a scale of 1-10:
            Question: {question}
            Answer: {answer}
            
            Consider:
            - Does the answer directly address the question?
            - Is the information relevant to the query?
            
            Provide a score (1-10) and brief explanation.
            """,
            
            "factuality": """
            Evaluate the factual accuracy of this answer based on the provided context:
            Question: {question}
            Answer: {answer}
            Context: {context}
            
            Consider:
            - Are the facts stated in the answer supported by the context?
            - Are there any contradictions or inaccuracies?
            
            Provide a score (1-10) and identify any factual issues.
            """,
            
            "completeness": """
            Evaluate how complete this answer is for the given question:
            Question: {question}
            Answer: {answer}
            
            Consider:
            - Does the answer fully address all aspects of the question?
            - Is important information missing?
            
            Provide a score (1-10) and note any missing information.
            """
        }
    
    def reflect_on_response(self, 
                          question: str,
                          answer: str,
                          contexts: List[str]) -> Dict[str, Any]:
        """Perform self-reflection on a RAG response"""
        
        if not llm_manager or not llm_manager.get_available_providers():
            return {"error": "No LLM available for self-reflection"}
        
        reflection_results = {}
        
        for aspect, prompt_template in self.reflection_prompts.items():
            try:
                # Format prompt
                context_text = "\n".join(contexts[:3])  # Use top 3 contexts
                prompt = prompt_template.format(
                    question=question,
                    answer=answer,
                    context=context_text
                )
                
                # Get reflection response
                reflection_response = llm_manager.generate(
                    prompt=prompt,
                    provider=settings.REFLECTION_MODEL if settings.REFLECTION_MODEL != "auto" else None
                )
                
                # Extract score (simple pattern matching)
                score = self._extract_score(reflection_response)
                
                reflection_results[aspect] = {
                    "score": score,
                    "feedback": reflection_response
                }
                
            except Exception as e:
                logger.error(f"Self-reflection failed for {aspect}: {e}")
                reflection_results[aspect] = {
                    "score": 5.0,
                    "feedback": f"Reflection failed: {e}"
                }
        
        return reflection_results
    
    def _extract_score(self, reflection_text: str) -> float:
        """Extract numerical score from reflection text"""
        import re
        
        # Look for patterns like "Score: 8", "8/10", "8 out of 10"
        patterns = [
            r'score:?\s*(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)\s*/\s*10',
            r'(\d+(?:\.\d+)?)\s+out\s+of\s+10'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, reflection_text.lower())
            if match:
                try:
                    score = float(match.group(1))
                    return min(10.0, max(1.0, score))  # Clamp to 1-10 range
                except ValueError:
                    continue
        
        # Fallback to sentiment analysis
        if "excellent" in reflection_text.lower() or "perfect" in reflection_text.lower():
            return 9.0
        elif "good" in reflection_text.lower() or "accurate" in reflection_text.lower():
            return 7.0
        elif "poor" in reflection_text.lower() or "inaccurate" in reflection_text.lower():
            return 3.0
        
        return 5.0  # Default middle score

class ContinuousEvaluator:
    """Continuous evaluation system for production monitoring"""
    
    def __init__(self):
        self.ragas_evaluator = RAGASEvaluator()
        self.reflection_evaluator = SelfReflectionEvaluator()
        self.alert_thresholds = {
            "faithfulness": 0.7,
            "answer_relevancy": 0.7,
            "context_precision": 0.6
        }
        self.evaluation_queue = []
    
    def add_to_evaluation_queue(self, 
                              query: str,
                              answer: str,
                              contexts: List[str],
                              metadata: Dict[str, Any] = None):
        """Add response to evaluation queue"""
        self.evaluation_queue.append({
            "query": query,
            "answer": answer,
            "contexts": contexts,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat()
        })
    
    def process_evaluation_queue(self, batch_size: int = 10):
        """Process queued evaluations in batches"""
        if not self.evaluation_queue:
            return
        
        # Process in batches
        for i in range(0, len(self.evaluation_queue), batch_size):
            batch = self.evaluation_queue[i:i + batch_size]
            
            try:
                results = self.ragas_evaluator.evaluate_batch(batch)
                self._check_alerts(results)
                
            except Exception as e:
                logger.error(f"Batch evaluation failed: {e}")
        
        # Clear processed items
        self.evaluation_queue = []
    
    def _check_alerts(self, results: List[EvaluationResult]):
        """Check evaluation results against alert thresholds"""
        for result in results:
            for metric, threshold in self.alert_thresholds.items():
                if metric in result.metrics and result.metrics[metric] < threshold:
                    logger.warning(
                        f"Quality alert: {metric} = {result.metrics[metric]:.3f} "
                        f"below threshold {threshold} for query: {result.query[:100]}..."
                    )

# Global evaluator instances
try:
    ragas_evaluator = RAGASEvaluator()
    reflection_evaluator = SelfReflectionEvaluator()
    continuous_evaluator = ContinuousEvaluator()
    logger.info("Evaluation framework initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize evaluation framework: {e}")
    ragas_evaluator = None
    reflection_evaluator = None
    continuous_evaluator = None