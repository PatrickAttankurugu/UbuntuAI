import openai
import asyncio
import json
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod
import uuid
from config.settings import settings
from api.memory_chains import FeedbackMemoryChain

class TaskType(Enum):
    ONBOARDING = "onboarding"
    BUSINESS_ASSESSMENT = "business_assessment"
    CREDIT_EVALUATION = "credit_evaluation"
    GROWTH_PLANNING = "growth_planning"
    IMPACT_MEASUREMENT = "impact_measurement"
    REGULATORY_GUIDANCE = "regulatory_guidance"

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    DELEGATED = "delegated"

@dataclass
class Task:
    task_id: str
    task_type: TaskType
    description: str
    input_data: Dict[str, Any]
    context: Dict[str, Any]
    priority: int
    status: TaskStatus
    assigned_agent: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime = None
    completed_at: Optional[datetime] = None
    dependencies: List[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.dependencies is None:
            self.dependencies = []
        if self.metadata is None:
            self.metadata = {}

@dataclass
class AgentCapability:
    capability_id: str
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    supported_task_types: List[TaskType]
    performance_metrics: Dict[str, float]
    resource_requirements: Dict[str, Any]

class BaseAgent(ABC):
    """Base class for modular AI agents following MCP patterns"""
    
    def __init__(self, agent_id: str, capabilities: List[AgentCapability]):
        self.agent_id = agent_id
        self.capabilities = {cap.capability_id: cap for cap in capabilities}
        self.memory = FeedbackMemoryChain()
        self.shared_memory = None  # Will be set by orchestrator
        self.status = "idle"
        self.current_task = None
        
    @abstractmethod
    async def execute_task(self, task: Task) -> Dict[str, Any]:
        """Execute a specific task"""
        pass
    
    @abstractmethod
    def can_handle_task(self, task: Task) -> bool:
        """Check if agent can handle the given task"""
        pass
    
    def get_capability(self, capability_id: str) -> Optional[AgentCapability]:
        """Get specific capability"""
        return self.capabilities.get(capability_id)
    
    def update_performance_metrics(self, capability_id: str, metrics: Dict[str, float]):
        """Update performance metrics for a capability"""
        if capability_id in self.capabilities:
            self.capabilities[capability_id].performance_metrics.update(metrics)

class SharedMemoryManager:
    """Manages shared memory and prompt logic across agents"""
    
    def __init__(self):
        self.session_memory = {}
        self.user_profiles = {}
        self.conversation_contexts = {}
        self.business_contexts = {}
        self.cross_agent_learnings = {}
        
    def get_user_context(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive user context"""
        return {
            "profile": self.user_profiles.get(user_id, {}),
            "conversation_history": self.conversation_contexts.get(user_id, []),
            "business_context": self.business_contexts.get(user_id, {}),
            "session_data": self.session_memory.get(user_id, {})
        }
    
    def update_user_context(self, user_id: str, context_update: Dict[str, Any]):
        """Update user context across all dimensions"""
        
        if "profile" in context_update:
            if user_id not in self.user_profiles:
                self.user_profiles[user_id] = {}
            self.user_profiles[user_id].update(context_update["profile"])
        
        if "conversation" in context_update:
            if user_id not in self.conversation_contexts:
                self.conversation_contexts[user_id] = []
            self.conversation_contexts[user_id].append(context_update["conversation"])
        
        if "business" in context_update:
            if user_id not in self.business_contexts:
                self.business_contexts[user_id] = {}
            self.business_contexts[user_id].update(context_update["business"])
        
        if "session" in context_update:
            if user_id not in self.session_memory:
                self.session_memory[user_id] = {}
            self.session_memory[user_id].update(context_update["session"])
    
    def store_cross_agent_learning(self, learning_data: Dict[str, Any]):
        """Store learnings that can be shared across agents"""
        
        learning_id = f"learning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.cross_agent_learnings[learning_id] = {
            **learning_data,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_relevant_learnings(self, task: Task) -> List[Dict[str, Any]]:
        """Get relevant cross-agent learnings for a task"""
        
        relevant_learnings = []
        
        for learning in self.cross_agent_learnings.values():
            if self._is_learning_relevant(learning, task):
                relevant_learnings.append(learning)
        
        return relevant_learnings
    
    def _is_learning_relevant(self, learning: Dict[str, Any], task: Task) -> bool:
        """Check if a learning is relevant to the current task"""
        
        # Check task type similarity
        if learning.get("task_type") == task.task_type.value:
            return True
        
        # Check context similarity
        learning_context = learning.get("context", {})
        task_context = task.context
        
        # Simple context matching (can be enhanced with ML)
        common_keys = set(learning_context.keys()) & set(task_context.keys())
        if len(common_keys) > 0:
            matching_values = sum(
                1 for key in common_keys 
                if learning_context[key] == task_context[key]
            )
            similarity = matching_values / len(common_keys)
            return similarity > 0.5
        
        return False

class TaskOrchestrator:
    """Orchestrates tasks across multiple agents using MCP patterns"""
    
    def __init__(self):
        self.agents = {}
        self.task_queue = []
        self.active_tasks = {}
        self.completed_tasks = {}
        self.shared_memory = SharedMemoryManager()
        
        # Workflow definitions
        self.workflows = {
            "entrepreneur_onboarding": self._define_onboarding_workflow(),
            "credit_assessment": self._define_credit_workflow(),
            "growth_strategy": self._define_growth_workflow(),
            "impact_evaluation": self._define_impact_workflow()
        }
    
    def register_agent(self, agent: BaseAgent):
        """Register an agent with the orchestrator"""
        
        self.agents[agent.agent_id] = agent
        agent.shared_memory = self.shared_memory
    
    async def execute_workflow(self, 
                             workflow_name: str,
                             user_id: str,
                             input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a complete workflow across multiple agents"""
        
        if workflow_name not in self.workflows:
            raise ValueError(f"Unknown workflow: {workflow_name}")
        
        workflow = self.workflows[workflow_name]
        workflow_id = f"workflow_{uuid.uuid4().hex[:8]}"
        
        # Initialize workflow context
        workflow_context = {
            "workflow_id": workflow_id,
            "user_id": user_id,
            "input_data": input_data,
            "user_context": self.shared_memory.get_user_context(user_id),
            "results": {},
            "status": "in_progress"
        }
        
        try:
            # Execute workflow steps
            for step in workflow["steps"]:
                step_result = await self._execute_workflow_step(step, workflow_context)
                workflow_context["results"][step["step_id"]] = step_result
                
                # Update shared memory with step results
                self._update_shared_memory_from_step(user_id, step, step_result)
            
            workflow_context["status"] = "completed"
            return workflow_context
            
        except Exception as e:
            workflow_context["status"] = "failed"
            workflow_context["error"] = str(e)
            return workflow_context
    
    async def _execute_workflow_step(self, 
                                   step: Dict[str, Any],
                                   workflow_context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single workflow step"""
        
        # Create task for the step
        task = Task(
            task_id=f"task_{uuid.uuid4().hex[:8]}",
            task_type=TaskType(step["task_type"]),
            description=step["description"],
            input_data=step.get("input_data", {}),
            context=workflow_context,
            priority=step.get("priority", 5),
            status=TaskStatus.PENDING,
            dependencies=step.get("dependencies", [])
        )
        
        # Find suitable agent
        suitable_agent = self._find_suitable_agent(task)
        
        if not suitable_agent:
            raise ValueError(f"No suitable agent found for task: {task.task_type}")
        
        # Execute task
        task.assigned_agent = suitable_agent.agent_id
        task.status = TaskStatus.IN_PROGRESS
        
        result = await suitable_agent.execute_task(task)
        
        task.result = result
        task.status = TaskStatus.COMPLETED
        task.completed_at = datetime.now()
        
        return result
    
    def _find_suitable_agent(self, task: Task) -> Optional[BaseAgent]:
        """Find the most suitable agent for a task"""
        
        suitable_agents = []
        
        for agent in self.agents.values():
            if agent.can_handle_task(task):
                # Calculate suitability score
                score = self._calculate_agent_suitability(agent, task)
                suitable_agents.append((score, agent))
        
        if not suitable_agents:
            return None
        
        # Return agent with highest suitability score
        suitable_agents.sort(key=lambda x: x[0], reverse=True)
        return suitable_agents[0][1]
    
    def _calculate_agent_suitability(self, agent: BaseAgent, task: Task) -> float:
        """Calculate how suitable an agent is for a task"""
        
        score = 0.0
        
        # Check capability match
        for capability in agent.capabilities.values():
            if task.task_type in capability.supported_task_types:
                score += 0.5
                
                # Performance bonus
                performance = capability.performance_metrics.get("success_rate", 0.5)
                score += performance * 0.3
        
        # Availability bonus (not currently handling a task)
        if agent.status == "idle":
            score += 0.2
        
        return score
    
    def _define_onboarding_workflow(self) -> Dict[str, Any]:
        """Define entrepreneur onboarding workflow"""
        
        return {
            "name": "entrepreneur_onboarding",
            "description": "Complete entrepreneur onboarding and assessment",
            "steps": [
                {
                    "step_id": "profile_collection",
                    "task_type": "onboarding",
                    "description": "Collect entrepreneur profile and business information",
                    "priority": 1,
                    "dependencies": []
                },
                {
                    "step_id": "business_assessment",
                    "task_type": "business_assessment", 
                    "description": "Assess business readiness and potential",
                    "priority": 2,
                    "dependencies": ["profile_collection"]
                },
                {
                    "step_id": "growth_recommendations",
                    "task_type": "growth_planning",
                    "description": "Generate growth strategy recommendations",
                    "priority": 3,
                    "dependencies": ["business_assessment"]
                }
            ]
        }
    
    def _define_credit_workflow(self) -> Dict[str, Any]:
        """Define credit assessment workflow"""
        
        return {
            "name": "credit_assessment",
            "description": "Complete credit evaluation and loan recommendation",
            "steps": [
                {
                    "step_id": "data_collection",
                    "task_type": "onboarding",
                    "description": "Collect financial and business data",
                    "priority": 1,
                    "dependencies": []
                },
                {
                    "step_id": "credit_scoring",
                    "task_type": "credit_evaluation",
                    "description": "Perform credit scoring and risk assessment",
                    "priority": 2,
                    "dependencies": ["data_collection"]
                },
                {
                    "step_id": "loan_structuring",
                    "task_type": "credit_evaluation",
                    "description": "Structure loan terms and recommendations",
                    "priority": 3,
                    "dependencies": ["credit_scoring"]
                }
            ]
        }
    
    def _define_growth_workflow(self) -> Dict[str, Any]:
        """Define growth strategy workflow"""
        
        return {
            "name": "growth_strategy",
            "description": "Develop comprehensive growth strategy",
            "steps": [
                {
                    "step_id": "performance_analysis",
                    "task_type": "business_assessment",
                    "description": "Analyze current business performance",
                    "priority": 1,
                    "dependencies": []
                },
                {
                    "step_id": "growth_planning",
                    "task_type": "growth_planning",
                    "description": "Generate growth recommendations",
                    "priority": 2,
                    "dependencies": ["performance_analysis"]
                },
                {
                    "step_id": "implementation_roadmap",
                    "task_type": "growth_planning",
                    "description": "Create implementation roadmap",
                    "priority": 3,
                    "dependencies": ["growth_planning"]
                }
            ]
        }
    
    def _define_impact_workflow(self) -> Dict[str, Any]:
        """Define impact measurement workflow"""
        
        return {
            "name": "impact_evaluation",
            "description": "Comprehensive impact measurement and reporting",
            "steps": [
                {
                    "step_id": "framework_design",
                    "task_type": "impact_measurement",
                    "description": "Design impact measurement framework",
                    "priority": 1,
                    "dependencies": []
                },
                {
                    "step_id": "data_collection",
                    "task_type": "impact_measurement",
                    "description": "Collect impact data",
                    "priority": 2,
                    "dependencies": ["framework_design"]
                },
                {
                    "step_id": "impact_analysis",
                    "task_type": "impact_measurement",
                    "description": "Analyze impact and generate insights",
                    "priority": 3,
                    "dependencies": ["data_collection"]
                }
            ]
        }
    
    def _update_shared_memory_from_step(self, 
                                      user_id: str,
                                      step: Dict[str, Any],
                                      step_result: Dict[str, Any]):
        """Update shared memory with step results"""
        
        context_update = {}
        
        # Extract relevant information based on step type
        if step["task_type"] == "onboarding":
            context_update["profile"] = step_result.get("profile_data", {})
            context_update["business"] = step_result.get("business_data", {})
        
        elif step["task_type"] == "business_assessment":
            context_update["business"] = {
                "assessment_score": step_result.get("overall_score"),
                "readiness_level": step_result.get("readiness_level"),
                "key_strengths": step_result.get("strengths", []),
                "improvement_areas": step_result.get("weaknesses", [])
            }
        
        elif step["task_type"] == "credit_evaluation":
            context_update["business"] = {
                "credit_score": step_result.get("credit_score"),
                "risk_category": step_result.get("risk_category"),
                "loan_eligibility": step_result.get("loan_recommendation", {})
            }
        
        # Update shared memory
        if context_update:
            self.shared_memory.update_user_context(user_id, context_update)

class OnboardingAgent(BaseAgent):
    """Agent specialized in entrepreneur onboarding"""
    
    def __init__(self):
        capabilities = [
            AgentCapability(
                capability_id="profile_collection",
                name="Profile Collection",
                description="Collect and validate entrepreneur profiles",
                input_schema={"type": "object"},
                output_schema={"type": "object"},
                supported_task_types=[TaskType.ONBOARDING],
                performance_metrics={"success_rate": 0.95, "avg_completion_time": 300},
                resource_requirements={"memory": "low", "compute": "low"}
            )
        ]
        super().__init__("onboarding_agent", capabilities)
    
    async def execute_task(self, task: Task) -> Dict[str, Any]:
        """Execute onboarding task"""
        
        if task.task_type == TaskType.ONBOARDING:
            return await self._handle_onboarding(task)
        else:
            raise ValueError(f"Cannot handle task type: {task.task_type}")
    
    def can_handle_task(self, task: Task) -> bool:
        """Check if agent can handle the task"""
        return task.task_type == TaskType.ONBOARDING
    
    async def _handle_onboarding(self, task: Task) -> Dict[str, Any]:
        """Handle onboarding process"""
        
        input_data = task.input_data
        context = task.context
        
        # Simulate onboarding process
        onboarding_result = {
            "profile_data": {
                "user_id": context.get("user_id"),
                "business_type": input_data.get("business_type"),
                "sector": input_data.get("sector"),
                "location": input_data.get("location"),
                "team_size": input_data.get("team_size"),
                "experience_level": input_data.get("experience_level")
            },
            "business_data": {
                "business_description": input_data.get("business_description"),
                "target_market": input_data.get("target_market"),
                "revenue_model": input_data.get("revenue_model"),
                "current_stage": input_data.get("current_stage"),
                "funding_needs": input_data.get("funding_needs")
            },
            "completion_status": "completed",
            "next_recommended_action": "business_assessment"
        }
        
        return onboarding_result

class CreditAgent(BaseAgent):
    """Agent specialized in credit evaluation"""
    
    def __init__(self):
        capabilities = [
            AgentCapability(
                capability_id="credit_scoring",
                name="Credit Scoring",
                description="Perform credit scoring and risk assessment",
                input_schema={"type": "object"},
                output_schema={"type": "object"},
                supported_task_types=[TaskType.CREDIT_EVALUATION],
                performance_metrics={"success_rate": 0.88, "accuracy": 0.92},
                resource_requirements={"memory": "medium", "compute": "medium"}
            )
        ]
        super().__init__("credit_agent", capabilities)
    
    async def execute_task(self, task: Task) -> Dict[str, Any]:
        """Execute credit evaluation task"""
        
        if task.task_type == TaskType.CREDIT_EVALUATION:
            return await self._handle_credit_evaluation(task)
        else:
            raise ValueError(f"Cannot handle task type: {task.task_type}")
    
    def can_handle_task(self, task: Task) -> bool:
        """Check if agent can handle the task"""
        return task.task_type == TaskType.CREDIT_EVALUATION
    
    async def _handle_credit_evaluation(self, task: Task) -> Dict[str, Any]:
        """Handle credit evaluation process"""
        
        # Simulate credit evaluation using the credit underwriting engine
        from api.credit_underwriting import create_credit_underwriting_engine
        
        credit_engine = create_credit_underwriting_engine()
        
        # Extract applicant data from context
        user_context = task.context.get("user_context", {})
        business_context = user_context.get("business_context", {})
        
        applicant_data = {
            "id": task.context.get("user_id"),
            "age": business_context.get("founder_age", 35),
            "gender": business_context.get("gender", "unknown"),
            "monthly_income": business_context.get("monthly_revenue", 5000),
            "business_owner": True,
            "business_age_months": business_context.get("business_age_months", 12),
            "business_sector": business_context.get("sector", "general"),
            "location": business_context.get("location", "Ghana")
        }
        
        loan_request = task.input_data.get("loan_request", {
            "amount": 50000,
            "term_months": 12,
            "purpose": "business_expansion"
        })
        
        # Perform credit assessment
        assessment = credit_engine.assess_credit_application(
            applicant_data, loan_request
        )
        
        return {
            "credit_score": assessment.credit_score,
            "risk_category": assessment.risk_category,
            "loan_recommendation": asdict(assessment.loan_recommendation),
            "approval_probability": assessment.approval_probability,
            "recommended_amount": assessment.recommended_amount,
            "interest_rate": assessment.interest_rate_suggestion,
            "risk_factors": assessment.risk_factors,
            "alternative_products": assessment.alternative_products
        }

# Factory functions
def create_task_orchestrator():
    """Create and configure task orchestrator with agents"""
    
    orchestrator = TaskOrchestrator()
    
    # Register agents
    orchestrator.register_agent(OnboardingAgent())
    orchestrator.register_agent(CreditAgent())
    
    return orchestrator

def create_shared_memory_manager():
    return SharedMemoryManager()