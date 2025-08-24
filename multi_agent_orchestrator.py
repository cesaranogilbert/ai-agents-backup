import logging
import asyncio
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
from models import AIAgent, ReplitApp
from app import db
from sqlalchemy import func, desc
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress" 
    COMPLETED = "completed"
    FAILED = "failed"
    ESCALATED = "escalated"
    SPECIALIST_REQUIRED = "specialist_required"

class AgentRole(Enum):
    GENERALIST = "generalist"
    SPECIALIST = "specialist"
    QUALITY_ASSURANCE = "qa"
    COORDINATOR = "coordinator"
    RESEARCHER = "researcher"
    OPTIMIZER = "optimizer"

@dataclass
class TaskRequest:
    id: str
    description: str
    requirements: List[str]
    priority: int  # 1-10, 10 being highest
    max_attempts: int = 3
    current_attempts: int = 0
    status: TaskStatus = TaskStatus.PENDING
    assigned_agents: List[str] = None
    results: List[Dict] = None
    feedback_history: List[Dict] = None
    created_at: datetime = None
    deadline: Optional[datetime] = None
    complexity_score: float = 0.5
    
    def __post_init__(self):
        if self.assigned_agents is None:
            self.assigned_agents = []
        if self.results is None:
            self.results = []
        if self.feedback_history is None:
            self.feedback_history = []
        if self.created_at is None:
            self.created_at = datetime.utcnow()

@dataclass
class AgentCapability:
    agent_id: int
    role: AgentRole
    specialization: List[str]
    effectiveness_score: float
    current_load: int
    max_concurrent_tasks: int
    response_time_avg: float
    success_rate: float
    cost_per_task: float

class MultiAgentOrchestrator:
    """
    Advanced Multi-Agent Orchestration System
    - Parallelized execution across multiple agents
    - Multi-dimensional quality assurance
    - Automatic specialist agent generation
    - Self-improving feedback loops
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.active_tasks: Dict[str, TaskRequest] = {}
        self.agent_pool: Dict[int, AgentCapability] = {}
        self.task_queue = queue.PriorityQueue()
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.running = False
        self.generation_threshold = 3  # Auto-generate specialist after 3 failures
        self.quality_gates = [
            self._gate_completeness_check,
            self._gate_accuracy_validation, 
            self._gate_performance_verification,
            self._gate_consistency_analysis,
            self._gate_security_compliance
        ]
        
        # Initialize agent pool
        self._initialize_agent_pool()
        
        # Start orchestrator
        self._start_orchestrator()
    
    def submit_task(self, description: str, requirements: List[str], 
                   priority: int = 5, deadline: Optional[datetime] = None) -> str:
        """Submit a new task for multi-agent processing"""
        
        task_id = self._generate_task_id(description)
        complexity = self._analyze_task_complexity(description, requirements)
        
        task = TaskRequest(
            id=task_id,
            description=description,
            requirements=requirements,
            priority=priority,
            deadline=deadline,
            complexity_score=complexity
        )
        
        self.active_tasks[task_id] = task
        
        # Add to priority queue (negative priority for max-heap behavior)
        self.task_queue.put((-priority, task_id))
        
        self.logger.info(f"Task {task_id} submitted with priority {priority}, complexity {complexity:.2f}")
        return task_id
    
    async def process_task_parallel(self, task_id: str) -> Dict[str, Any]:
        """Process task using parallel multi-agent collaboration"""
        
        task = self.active_tasks.get(task_id)
        if not task:
            return {'error': 'Task not found'}
        
        try:
            task.status = TaskStatus.IN_PROGRESS
            task.current_attempts += 1
            
            # Phase 1: Agent Selection and Assignment
            selected_agents = await self._select_optimal_agents(task)
            task.assigned_agents = [str(agent.agent_id) for agent in selected_agents]
            
            # Phase 2: Parallel Execution
            execution_results = await self._execute_parallel_processing(task, selected_agents)
            
            # Phase 3: Multi-Level Quality Assurance
            qa_results = await self._run_quality_assurance_pipeline(task, execution_results)
            
            # Phase 4: Result Synthesis and Validation
            final_result = await self._synthesize_results(task, execution_results, qa_results)
            
            # Phase 5: Feedback Loop Analysis
            if not self._meets_quality_threshold(final_result):
                return await self._handle_quality_failure(task, final_result)
            
            task.status = TaskStatus.COMPLETED
            task.results.append(final_result)
            
            # Phase 6: Performance Learning
            await self._update_agent_performance_metrics(task, selected_agents, final_result)
            
            return {
                'task_id': task_id,
                'status': 'completed',
                'result': final_result,
                'agents_used': len(selected_agents),
                'execution_time': (datetime.utcnow() - task.created_at).total_seconds(),
                'quality_score': final_result.get('quality_score', 0),
                'cost_estimate': sum([agent.cost_per_task for agent in selected_agents])
            }
            
        except Exception as e:
            self.logger.error(f"Error processing task {task_id}: {str(e)}")
            return await self._handle_processing_error(task, str(e))
    
    async def _select_optimal_agents(self, task: TaskRequest) -> List[AgentCapability]:
        """Select optimal agents for task using multi-dimensional scoring"""
        
        available_agents = [
            agent for agent in self.agent_pool.values() 
            if agent.current_load < agent.max_concurrent_tasks
        ]
        
        # Score agents based on multiple factors
        scored_agents = []
        for agent in available_agents:
            score = self._calculate_agent_task_score(agent, task)
            scored_agents.append((score, agent))
        
        # Sort by score (highest first)
        scored_agents.sort(key=lambda x: x[0], reverse=True)
        
        # Select top agents based on task complexity
        num_agents = min(
            max(1, int(task.complexity_score * 5)),  # 1-5 agents based on complexity
            len(scored_agents)
        )
        
        selected = [agent for _, agent in scored_agents[:num_agents]]
        
        # Ensure we have at least one QA agent for complex tasks
        if task.complexity_score > 0.7 and not any(agent.role == AgentRole.QUALITY_ASSURANCE for agent in selected):
            qa_agents = [agent for agent in available_agents if agent.role == AgentRole.QUALITY_ASSURANCE]
            if qa_agents:
                selected.append(qa_agents[0])
        
        # Update agent load
        for agent in selected:
            agent.current_load += 1
        
        return selected
    
    async def _execute_parallel_processing(self, task: TaskRequest, agents: List[AgentCapability]) -> List[Dict[str, Any]]:
        """Execute task across multiple agents in parallel"""
        
        # Create subtasks for parallel execution
        subtasks = self._decompose_task(task, len(agents))
        
        # Execute in parallel
        futures = []
        for i, agent in enumerate(agents):
            future = self.executor.submit(
                self._execute_agent_task,
                agent,
                subtasks[i] if i < len(subtasks) else task.description,
                task.requirements
            )
            futures.append(future)
        
        # Collect results
        results = []
        for future in as_completed(futures, timeout=300):  # 5-minute timeout
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                self.logger.error(f"Agent execution failed: {str(e)}")
                results.append({'error': str(e), 'success': False})
        
        return results
    
    async def _run_quality_assurance_pipeline(self, task: TaskRequest, results: List[Dict]) -> Dict[str, Any]:
        """Run multi-level quality assurance pipeline"""
        
        qa_results = {
            'gates_passed': 0,
            'total_gates': len(self.quality_gates),
            'gate_results': [],
            'overall_score': 0.0,
            'recommendations': []
        }
        
        # Run each quality gate
        for i, gate_func in enumerate(self.quality_gates):
            try:
                gate_result = await gate_func(task, results)
                qa_results['gate_results'].append(gate_result)
                
                if gate_result.get('passed', False):
                    qa_results['gates_passed'] += 1
                else:
                    qa_results['recommendations'].extend(gate_result.get('recommendations', []))
                    
            except Exception as e:
                self.logger.error(f"Quality gate {i} failed: {str(e)}")
                qa_results['gate_results'].append({
                    'gate': f"gate_{i}",
                    'passed': False,
                    'error': str(e)
                })
        
        qa_results['overall_score'] = qa_results['gates_passed'] / qa_results['total_gates']
        return qa_results
    
    async def _synthesize_results(self, task: TaskRequest, execution_results: List[Dict], qa_results: Dict) -> Dict[str, Any]:
        """Synthesize multi-agent results into final output"""
        
        successful_results = [r for r in execution_results if r.get('success', False)]
        
        if not successful_results:
            return {
                'success': False,
                'error': 'No successful agent results',
                'quality_score': 0.0,
                'recommendations': ['All agents failed to complete task']
            }
        
        # Synthesis strategies based on task type
        if len(successful_results) == 1:
            synthesized = successful_results[0]
        elif task.complexity_score > 0.8:
            # High complexity: Use ensemble approach
            synthesized = self._ensemble_synthesis(successful_results)
        else:
            # Lower complexity: Use best result
            synthesized = max(successful_results, key=lambda x: x.get('confidence', 0))
        
        # Combine with QA insights
        final_result = {
            'success': True,
            'content': synthesized.get('content', ''),
            'confidence': synthesized.get('confidence', 0.8),
            'quality_score': qa_results['overall_score'],
            'qa_gates_passed': qa_results['gates_passed'],
            'total_qa_gates': qa_results['total_gates'],
            'agent_count': len(execution_results),
            'synthesis_method': 'ensemble' if task.complexity_score > 0.8 else 'best_result',
            'recommendations': qa_results.get('recommendations', []),
            'metadata': {
                'task_complexity': task.complexity_score,
                'execution_results_count': len(execution_results),
                'successful_results_count': len(successful_results)
            }
        }
        
        return final_result
    
    async def _handle_quality_failure(self, task: TaskRequest, result: Dict) -> Dict[str, Any]:
        """Handle quality failure and decide next steps"""
        
        if task.current_attempts >= task.max_attempts:
            # Escalate to specialist agent generation
            return await self._escalate_to_specialist_generation(task, result)
        
        # Add feedback and retry
        task.feedback_history.append({
            'attempt': task.current_attempts,
            'timestamp': datetime.utcnow().isoformat(),
            'quality_score': result.get('quality_score', 0),
            'issues': result.get('recommendations', []),
            'action': 'retry_with_feedback'
        })
        
        task.status = TaskStatus.PENDING
        self.task_queue.put((-task.priority, task.id))
        
        return {
            'task_id': task.id,
            'status': 'retrying',
            'attempt': task.current_attempts,
            'quality_issues': result.get('recommendations', []),
            'next_action': 'retry_with_enhanced_agents'
        }
    
    async def _escalate_to_specialist_generation(self, task: TaskRequest, result: Dict) -> Dict[str, Any]:
        """Escalate to automatic specialist agent generation"""
        
        task.status = TaskStatus.SPECIALIST_REQUIRED
        
        # Analyze failure patterns
        failure_analysis = self._analyze_failure_patterns(task)
        
        # Generate specialist agent specification
        specialist_spec = await self._generate_specialist_specification(task, failure_analysis)
        
        # Create new specialist agent
        new_agent = await self._create_specialist_agent(specialist_spec)
        
        if new_agent:
            # Retry task with new specialist
            task.current_attempts = 0  # Reset attempts
            task.status = TaskStatus.PENDING
            task.feedback_history.append({
                'attempt': 'specialist_generation',
                'timestamp': datetime.utcnow().isoformat(),
                'action': 'created_specialist_agent',
                'specialist_id': new_agent.id,
                'specialist_role': specialist_spec['role']
            })
            
            self.task_queue.put((-10, task.id))  # High priority for specialist retry
            
            return {
                'task_id': task.id,
                'status': 'specialist_created',
                'specialist_agent_id': new_agent.id,
                'specialist_role': specialist_spec['role'],
                'next_action': 'retry_with_specialist'
            }
        
        return {
            'task_id': task.id,
            'status': 'failed',
            'error': 'Unable to create specialist agent',
            'final_attempt': True
        }
    
    async def _generate_specialist_specification(self, task: TaskRequest, failure_analysis: Dict) -> Dict[str, Any]:
        """Generate specification for new specialist agent"""
        
        # Identify specialization needed
        failed_areas = failure_analysis.get('failed_areas', [])
        task_domain = self._extract_task_domain(task.description, task.requirements)
        
        specialist_role = self._determine_specialist_role(failed_areas, task_domain)
        
        return {
            'role': specialist_role,
            'agent_type': 'specialist',
            'specialization': failed_areas,
            'domain': task_domain,
            'model_requirements': self._determine_model_requirements(task, failure_analysis),
            'capabilities': self._determine_required_capabilities(task, failure_analysis),
            'training_focus': failed_areas,
            'effectiveness_target': 0.95,  # High target for specialists
            'name': f"{specialist_role.title()} Specialist - {task_domain.title()}",
            'description': f"Specialized agent for {task_domain} tasks, focusing on {', '.join(failed_areas)}"
        }
    
    async def _create_specialist_agent(self, spec: Dict[str, Any]) -> Optional[AIAgent]:
        """Create new specialist AI agent"""
        
        try:
            # Create new agent record
            new_agent = AIAgent()
            new_agent.app_id = 9  # Associate with main app
            new_agent.agent_type = 'Specialist'
            new_agent.agent_name = spec['name']
            new_agent.model_name = spec['model_requirements'].get('preferred_model', 'gpt-4')
            new_agent.role_description = spec['description']
            new_agent.effectiveness_score = 0.8  # Start with good baseline
            new_agent.cost_estimate = 2.0  # Moderate cost for specialist
            new_agent.features_used = spec['capabilities']
            
            db.session.add(new_agent)
            db.session.commit()
            
            # Add to agent pool
            capability = AgentCapability(
                agent_id=new_agent.id,
                role=AgentRole.SPECIALIST,
                specialization=spec['specialization'],
                effectiveness_score=0.8,
                current_load=0,
                max_concurrent_tasks=3,
                response_time_avg=2.0,
                success_rate=0.95,
                cost_per_task=2.0
            )
            
            self.agent_pool[new_agent.id] = capability
            
            self.logger.info(f"Created specialist agent {new_agent.id}: {spec['name']}")
            return new_agent
            
        except Exception as e:
            self.logger.error(f"Failed to create specialist agent: {str(e)}")
            return None
    
    # Quality Gate Functions
    async def _gate_completeness_check(self, task: TaskRequest, results: List[Dict]) -> Dict[str, Any]:
        """Check if results address all requirements"""
        
        successful_results = [r for r in results if r.get('success', False)]
        if not successful_results:
            return {'gate': 'completeness', 'passed': False, 'reason': 'No successful results'}
        
        # Check if all requirements are addressed
        addressed_requirements = set()
        for result in successful_results:
            content = str(result.get('content', '')).lower()
            for req in task.requirements:
                if req.lower() in content:
                    addressed_requirements.add(req)
        
        coverage = len(addressed_requirements) / max(len(task.requirements), 1)
        passed = coverage >= 0.8  # 80% requirement coverage
        
        return {
            'gate': 'completeness',
            'passed': passed,
            'coverage': coverage,
            'addressed_requirements': list(addressed_requirements),
            'missing_requirements': [r for r in task.requirements if r not in addressed_requirements],
            'recommendations': ['Address missing requirements'] if not passed else []
        }
    
    async def _gate_accuracy_validation(self, task: TaskRequest, results: List[Dict]) -> Dict[str, Any]:
        """Validate accuracy of results"""
        
        successful_results = [r for r in results if r.get('success', False)]
        if not successful_results:
            return {'gate': 'accuracy', 'passed': False, 'reason': 'No results to validate'}
        
        # Cross-validate results
        confidence_scores = [r.get('confidence', 0.5) for r in successful_results]
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        
        # Check for consistency between agents
        consistency_score = self._calculate_result_consistency(successful_results)
        
        overall_accuracy = (avg_confidence + consistency_score) / 2
        passed = overall_accuracy >= 0.7
        
        return {
            'gate': 'accuracy',
            'passed': passed,
            'accuracy_score': overall_accuracy,
            'avg_confidence': avg_confidence,
            'consistency_score': consistency_score,
            'recommendations': ['Improve result consistency'] if consistency_score < 0.6 else []
        }
    
    async def _gate_performance_verification(self, task: TaskRequest, results: List[Dict]) -> Dict[str, Any]:
        """Verify performance requirements are met"""
        
        execution_time = (datetime.utcnow() - task.created_at).total_seconds()
        target_time = 60 * task.complexity_score  # Scale with complexity
        
        performance_score = min(1.0, target_time / max(execution_time, 1))
        passed = performance_score >= 0.7
        
        return {
            'gate': 'performance',
            'passed': passed,
            'execution_time': execution_time,
            'target_time': target_time,
            'performance_score': performance_score,
            'recommendations': ['Optimize execution time'] if not passed else []
        }
    
    async def _gate_consistency_analysis(self, task: TaskRequest, results: List[Dict]) -> Dict[str, Any]:
        """Analyze consistency across agent results"""
        
        successful_results = [r for r in results if r.get('success', False)]
        if len(successful_results) < 2:
            return {'gate': 'consistency', 'passed': True, 'reason': 'Single result, consistency not applicable'}
        
        consistency_score = self._calculate_result_consistency(successful_results)
        passed = consistency_score >= 0.6
        
        return {
            'gate': 'consistency',
            'passed': passed,
            'consistency_score': consistency_score,
            'result_count': len(successful_results),
            'recommendations': ['Improve agent alignment'] if not passed else []
        }
    
    async def _gate_security_compliance(self, task: TaskRequest, results: List[Dict]) -> Dict[str, Any]:
        """Check security compliance of results"""
        
        security_issues = []
        for result in results:
            content = str(result.get('content', ''))
            
            # Check for potential security issues
            if 'api_key' in content.lower() and '=' in content:
                security_issues.append('Potential API key exposure')
            if 'password' in content.lower() and '=' in content:
                security_issues.append('Potential password exposure')
            if 'eval(' in content or 'exec(' in content:
                security_issues.append('Unsafe code execution')
        
        passed = len(security_issues) == 0
        
        return {
            'gate': 'security',
            'passed': passed,
            'security_issues': security_issues,
            'recommendations': security_issues if security_issues else []
        }
    
    # Helper Methods
    def _initialize_agent_pool(self):
        """Initialize agent pool from existing agents"""
        
        agents = AIAgent.query.filter(AIAgent.effectiveness_score > 0.7).all()
        
        for agent in agents:
            capability = AgentCapability(
                agent_id=agent.id,
                role=self._determine_agent_role(agent),
                specialization=self._extract_specialization(agent),
                effectiveness_score=agent.effectiveness_score,
                current_load=0,
                max_concurrent_tasks=5,
                response_time_avg=1.5,
                success_rate=agent.effectiveness_score,
                cost_per_task=agent.cost_estimate / max(agent.usage_frequency, 1)
            )
            self.agent_pool[agent.id] = capability
    
    def _start_orchestrator(self):
        """Start the orchestrator background process"""
        
        def orchestrator_loop():
            self.running = True
            while self.running:
                try:
                    if not self.task_queue.empty():
                        priority, task_id = self.task_queue.get(timeout=1)
                        
                        # Process task asynchronously
                        asyncio.create_task(self.process_task_parallel(task_id))
                        
                except queue.Empty:
                    continue
                except Exception as e:
                    self.logger.error(f"Orchestrator error: {str(e)}")
        
        # Start orchestrator in background thread
        orchestrator_thread = threading.Thread(target=orchestrator_loop, daemon=True)
        orchestrator_thread.start()
    
    def _generate_task_id(self, description: str) -> str:
        """Generate unique task ID"""
        return hashlib.md5(f"{description}_{datetime.utcnow().isoformat()}".encode()).hexdigest()[:12]
    
    def _analyze_task_complexity(self, description: str, requirements: List[str]) -> float:
        """Analyze task complexity (0.0 to 1.0)"""
        
        factors = {
            'description_length': min(1.0, len(description) / 1000),
            'requirement_count': min(1.0, len(requirements) / 10),
            'technical_keywords': self._count_technical_keywords(description) / 20,
            'integration_complexity': self._assess_integration_complexity(description, requirements)
        }
        
        return min(1.0, sum(factors.values()) / len(factors))
    
    def _count_technical_keywords(self, text: str) -> int:
        """Count technical keywords in text"""
        keywords = [
            'api', 'database', 'authentication', 'integration', 'algorithm',
            'optimization', 'machine learning', 'ai', 'neural network',
            'microservice', 'deployment', 'security', 'encryption'
        ]
        return sum(1 for keyword in keywords if keyword in text.lower())
    
    def _assess_integration_complexity(self, description: str, requirements: List[str]) -> float:
        """Assess integration complexity"""
        integration_indicators = ['connect', 'integrate', 'sync', 'api', 'webhook', 'database']
        count = sum(1 for indicator in integration_indicators 
                   if any(indicator in text.lower() for text in [description] + requirements))
        return min(1.0, count / 5)
    
    def _calculate_agent_task_score(self, agent: AgentCapability, task: TaskRequest) -> float:
        """Calculate agent suitability score for task"""
        
        # Base score from effectiveness
        score = agent.effectiveness_score * 0.4
        
        # Specialization match
        specialization_match = 0
        for spec in agent.specialization:
            if spec.lower() in task.description.lower():
                specialization_match += 0.2
        score += min(0.3, specialization_match)
        
        # Load factor (prefer less loaded agents)
        load_factor = 1 - (agent.current_load / agent.max_concurrent_tasks)
        score += load_factor * 0.2
        
        # Cost efficiency
        cost_efficiency = 1 / max(agent.cost_per_task, 0.1)
        score += min(0.1, cost_efficiency / 10)
        
        return score
    
    def _decompose_task(self, task: TaskRequest, num_agents: int) -> List[str]:
        """Decompose task into subtasks for parallel execution"""
        
        if num_agents == 1:
            return [task.description]
        
        # Simple decomposition - can be enhanced with AI
        base_description = task.description
        subtasks = []
        
        for i in range(num_agents):
            if i == 0:
                subtasks.append(f"Primary analysis: {base_description}")
            elif i == 1 and len(task.requirements) > 0:
                subtasks.append(f"Requirements validation: {'; '.join(task.requirements)}")
            else:
                subtasks.append(f"Perspective {i}: {base_description}")
        
        return subtasks
    
    def _execute_agent_task(self, agent: AgentCapability, task_description: str, requirements: List[str]) -> Dict[str, Any]:
        """Execute task on specific agent"""
        
        try:
            # Simulate agent execution - replace with actual agent calls
            import time
            time.sleep(agent.response_time_avg)  # Simulate processing time
            
            # Mock result based on agent effectiveness
            import random
            success_probability = agent.effectiveness_score
            
            if random.random() < success_probability:
                return {
                    'agent_id': agent.agent_id,
                    'success': True,
                    'content': f"Agent {agent.agent_id} completed: {task_description}",
                    'confidence': agent.effectiveness_score,
                    'processing_time': agent.response_time_avg
                }
            else:
                return {
                    'agent_id': agent.agent_id,
                    'success': False,
                    'error': 'Agent processing failed',
                    'confidence': 0.0
                }
                
        except Exception as e:
            return {
                'agent_id': agent.agent_id,
                'success': False,
                'error': str(e),
                'confidence': 0.0
            }
        finally:
            # Reduce agent load
            agent.current_load = max(0, agent.current_load - 1)
    
    def _meets_quality_threshold(self, result: Dict) -> bool:
        """Check if result meets quality threshold"""
        return result.get('quality_score', 0) >= 0.7
    
    def _ensemble_synthesis(self, results: List[Dict]) -> Dict[str, Any]:
        """Synthesize results using ensemble approach"""
        
        # Weight by confidence
        weighted_content = []
        total_weight = 0
        
        for result in results:
            confidence = result.get('confidence', 0.5)
            content = result.get('content', '')
            weighted_content.append((content, confidence))
            total_weight += confidence
        
        # For now, return highest confidence result
        # Can be enhanced with more sophisticated ensemble methods
        best_result = max(results, key=lambda x: x.get('confidence', 0))
        
        return {
            'content': best_result.get('content', ''),
            'confidence': total_weight / len(results),
            'synthesis_method': 'confidence_weighted'
        }
    
    def _calculate_result_consistency(self, results: List[Dict]) -> float:
        """Calculate consistency score between results"""
        
        if len(results) < 2:
            return 1.0
        
        # Simple consistency check based on content similarity
        # Can be enhanced with semantic similarity
        contents = [str(r.get('content', '')) for r in results]
        
        # Calculate pairwise similarity
        similarities = []
        for i in range(len(contents)):
            for j in range(i + 1, len(contents)):
                # Simple Jaccard similarity on words
                words1 = set(contents[i].lower().split())
                words2 = set(contents[j].lower().split())
                
                if not words1 and not words2:
                    similarity = 1.0
                elif not words1 or not words2:
                    similarity = 0.0
                else:
                    similarity = len(words1.intersection(words2)) / len(words1.union(words2))
                
                similarities.append(similarity)
        
        return sum(similarities) / len(similarities) if similarities else 0.0
    
    def _analyze_failure_patterns(self, task: TaskRequest) -> Dict[str, Any]:
        """Analyze patterns in task failures"""
        
        failure_patterns = {
            'failed_areas': [],
            'common_issues': [],
            'complexity_factors': [],
            'resource_requirements': []
        }
        
        # Analyze feedback history
        for feedback in task.feedback_history:
            issues = feedback.get('issues', [])
            failure_patterns['common_issues'].extend(issues)
        
        # Identify failed areas from task description and requirements
        if 'api' in task.description.lower():
            failure_patterns['failed_areas'].append('api_integration')
        if 'database' in task.description.lower():
            failure_patterns['failed_areas'].append('database_operations')
        if any('security' in req.lower() for req in task.requirements):
            failure_patterns['failed_areas'].append('security')
        
        # Default failed area if none identified
        if not failure_patterns['failed_areas']:
            failure_patterns['failed_areas'] = ['general_processing']
        
        return failure_patterns
    
    def _extract_task_domain(self, description: str, requirements: List[str]) -> str:
        """Extract primary domain from task"""
        
        text = (description + ' ' + ' '.join(requirements)).lower()
        
        domains = {
            'web_development': ['html', 'css', 'javascript', 'web', 'frontend', 'backend'],
            'data_analysis': ['data', 'analysis', 'statistics', 'visualization', 'pandas'],
            'machine_learning': ['ml', 'ai', 'model', 'prediction', 'training'],
            'api_development': ['api', 'rest', 'endpoint', 'service', 'integration'],
            'database': ['database', 'sql', 'query', 'table', 'schema'],
            'security': ['security', 'authentication', 'encryption', 'authorization']
        }
        
        for domain, keywords in domains.items():
            if any(keyword in text for keyword in keywords):
                return domain
        
        return 'general'
    
    def _determine_specialist_role(self, failed_areas: List[str], domain: str) -> str:
        """Determine specialist role needed"""
        
        role_mapping = {
            'api_integration': 'API_Integration_Specialist',
            'database_operations': 'Database_Specialist',
            'security': 'Security_Specialist',
            'performance': 'Performance_Specialist',
            'general_processing': f'{domain.title()}_Specialist'
        }
        
        # Use the first failed area or default to domain specialist
        primary_failure = failed_areas[0] if failed_areas else 'general_processing'
        return role_mapping.get(primary_failure, f'{domain.title()}_Specialist')
    
    def _determine_model_requirements(self, task: TaskRequest, failure_analysis: Dict) -> Dict[str, Any]:
        """Determine model requirements for specialist"""
        
        requirements = {
            'preferred_model': 'gpt-4',
            'min_context_length': 4000,
            'capabilities': ['text_generation']
        }
        
        # Adjust based on task complexity
        if task.complexity_score > 0.8:
            requirements['preferred_model'] = 'gpt-4'
            requirements['min_context_length'] = 8000
        
        # Adjust based on failed areas
        failed_areas = failure_analysis.get('failed_areas', [])
        if 'api_integration' in failed_areas:
            requirements['capabilities'].append('code_generation')
        if 'security' in failed_areas:
            requirements['capabilities'].append('security_analysis')
        
        return requirements
    
    def _determine_required_capabilities(self, task: TaskRequest, failure_analysis: Dict) -> List[str]:
        """Determine required capabilities for specialist"""
        
        capabilities = ['task_completion']
        
        failed_areas = failure_analysis.get('failed_areas', [])
        
        capability_mapping = {
            'api_integration': ['api_calls', 'json_processing', 'http_handling'],
            'database_operations': ['sql_generation', 'data_validation', 'schema_analysis'],
            'security': ['security_scanning', 'vulnerability_assessment', 'compliance_checking'],
            'performance': ['optimization', 'profiling', 'bottleneck_analysis']
        }
        
        for area in failed_areas:
            capabilities.extend(capability_mapping.get(area, []))
        
        return list(set(capabilities))  # Remove duplicates
    
    def _determine_agent_role(self, agent: AIAgent) -> AgentRole:
        """Determine agent role from agent data"""
        
        if 'specialist' in agent.agent_name.lower():
            return AgentRole.SPECIALIST
        elif 'qa' in agent.agent_name.lower() or 'quality' in agent.agent_name.lower():
            return AgentRole.QUALITY_ASSURANCE
        elif 'coordinator' in agent.agent_name.lower():
            return AgentRole.COORDINATOR
        elif 'research' in agent.agent_name.lower():
            return AgentRole.RESEARCHER
        elif 'optim' in agent.agent_name.lower():
            return AgentRole.OPTIMIZER
        else:
            return AgentRole.GENERALIST
    
    def _extract_specialization(self, agent: AIAgent) -> List[str]:
        """Extract specialization from agent data"""
        
        specializations = []
        
        name_lower = agent.agent_name.lower()
        role_lower = (agent.role_description or '').lower()
        
        specialization_keywords = {
            'content': ['content', 'writing', 'documentation'],
            'code': ['code', 'programming', 'development'],
            'analysis': ['analysis', 'analytics', 'data'],
            'api': ['api', 'integration', 'service'],
            'security': ['security', 'auth', 'encryption'],
            'ui': ['ui', 'frontend', 'interface'],
            'database': ['database', 'sql', 'data']
        }
        
        for spec, keywords in specialization_keywords.items():
            if any(keyword in name_lower or keyword in role_lower for keyword in keywords):
                specializations.append(spec)
        
        return specializations if specializations else ['general']
    
    async def _handle_processing_error(self, task: TaskRequest, error: str) -> Dict[str, Any]:
        """Handle processing errors"""
        
        task.status = TaskStatus.FAILED
        task.feedback_history.append({
            'attempt': task.current_attempts,
            'timestamp': datetime.utcnow().isoformat(),
            'error': error,
            'action': 'processing_failed'
        })
        
        return {
            'task_id': task.id,
            'status': 'failed',
            'error': error,
            'attempts': task.current_attempts
        }
    
    async def _update_agent_performance_metrics(self, task: TaskRequest, agents: List[AgentCapability], result: Dict):
        """Update agent performance metrics based on task result"""
        
        success = result.get('success', False)
        quality_score = result.get('quality_score', 0)
        
        for agent in agents:
            # Update success rate
            current_rate = agent.success_rate
            agent.success_rate = (current_rate * 0.9) + (1.0 if success else 0.0) * 0.1
            
            # Update effectiveness score
            if success:
                agent.effectiveness_score = min(1.0, agent.effectiveness_score * 0.95 + quality_score * 0.05)
            else:
                agent.effectiveness_score = agent.effectiveness_score * 0.98
    
    def get_orchestrator_status(self) -> Dict[str, Any]:
        """Get current orchestrator status"""
        
        return {
            'active_tasks': len(self.active_tasks),
            'queued_tasks': self.task_queue.qsize(),
            'agent_pool_size': len(self.agent_pool),
            'running': self.running,
            'total_tasks_processed': len([t for t in self.active_tasks.values() if t.status == TaskStatus.COMPLETED]),
            'specialist_agents_created': len([a for a in self.agent_pool.values() if a.role == AgentRole.SPECIALIST])
        }

# Singleton instance
multi_agent_orchestrator = MultiAgentOrchestrator()