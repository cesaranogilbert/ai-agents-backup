# Multi-Agent Orchestration API Routes
# Autonomous Multi-Agent System with Parallel Processing and Quality Assurance

from flask import request, jsonify
from services.multi_agent_orchestrator import multi_agent_orchestrator, TaskStatus
from datetime import datetime, timedelta
import asyncio

def register_multi_agent_routes(app):
    """Register all multi-agent orchestration routes"""
    
    @app.route('/api/multi-agent/submit', methods=['POST'])
    def submit_multi_agent_task():
        """Submit task for multi-agent parallel processing"""
        try:
            data = request.get_json()
            
            description = data.get('description', '')
            requirements = data.get('requirements', [])
            priority = data.get('priority', 5)
            deadline_hours = data.get('deadline_hours')
            
            if not description:
                return jsonify({
                    'success': False,
                    'error': 'Task description is required'
                }), 400
            
            # Calculate deadline
            deadline = None
            if deadline_hours:
                deadline = datetime.utcnow() + timedelta(hours=deadline_hours)
            
            # Submit task
            task_id = multi_agent_orchestrator.submit_task(
                description=description,
                requirements=requirements,
                priority=priority,
                deadline=deadline
            )
            
            return jsonify({
                'success': True,
                'task_id': task_id,
                'status': 'submitted',
                'message': 'Task submitted for multi-agent parallel processing',
                'estimated_agents': min(5, len(requirements) + 1),
                'features': [
                    'Parallel execution across multiple AI agents',
                    'Multi-level quality assurance pipeline',
                    'Automatic specialist agent generation if needed',
                    'Self-improving feedback loops'
                ]
            })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/multi-agent/status/<task_id>', methods=['GET'])
    def get_task_status(task_id):
        """Get status of multi-agent task"""
        try:
            task = multi_agent_orchestrator.active_tasks.get(task_id)
            
            if not task:
                return jsonify({
                    'success': False,
                    'error': 'Task not found'
                }), 404
            
            return jsonify({
                'success': True,
                'task_id': task_id,
                'status': task.status.value,
                'attempts': task.current_attempts,
                'max_attempts': task.max_attempts,
                'assigned_agents': task.assigned_agents,
                'complexity_score': task.complexity_score,
                'created_at': task.created_at.isoformat(),
                'results_count': len(task.results),
                'feedback_history': task.feedback_history,
                'next_action': 'processing' if task.status == TaskStatus.IN_PROGRESS else 'waiting'
            })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/multi-agent/results/<task_id>', methods=['GET'])
    def get_task_results(task_id):
        """Get results of completed multi-agent task"""
        try:
            task = multi_agent_orchestrator.active_tasks.get(task_id)
            
            if not task:
                return jsonify({
                    'success': False,
                    'error': 'Task not found'
                }), 404
            
            if task.status != TaskStatus.COMPLETED:
                return jsonify({
                    'success': False,
                    'error': f'Task not completed. Current status: {task.status.value}',
                    'current_status': task.status.value
                }), 400
            
            latest_result = task.results[-1] if task.results else {}
            
            return jsonify({
                'success': True,
                'task_id': task_id,
                'status': 'completed',
                'result': latest_result,
                'execution_summary': {
                    'total_attempts': task.current_attempts,
                    'agents_used': len(task.assigned_agents),
                    'quality_score': latest_result.get('quality_score', 0),
                    'qa_gates_passed': latest_result.get('qa_gates_passed', 0),
                    'synthesis_method': latest_result.get('synthesis_method', 'unknown'),
                    'execution_time': (datetime.utcnow() - task.created_at).total_seconds()
                },
                'quality_assurance': {
                    'gates_passed': latest_result.get('qa_gates_passed', 0),
                    'total_gates': latest_result.get('total_qa_gates', 5),
                    'recommendations': latest_result.get('recommendations', [])
                }
            })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/multi-agent/process/<task_id>', methods=['POST'])
    def process_task_immediately(task_id):
        """Process task immediately (for testing/priority tasks)"""
        try:
            task = multi_agent_orchestrator.active_tasks.get(task_id)
            
            if not task:
                return jsonify({
                    'success': False,
                    'error': 'Task not found'
                }), 404
            
            # Process task in async context
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                result = loop.run_until_complete(
                    multi_agent_orchestrator.process_task_parallel(task_id)
                )
                return jsonify({
                    'success': True,
                    'processing_result': result
                })
            finally:
                loop.close()
            
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/multi-agent/orchestrator/status', methods=['GET'])
    def get_orchestrator_status():
        """Get overall orchestrator status"""
        try:
            status = multi_agent_orchestrator.get_orchestrator_status()
            
            return jsonify({
                'success': True,
                'orchestrator_status': status,
                'capabilities': {
                    'parallel_processing': True,
                    'quality_assurance_pipeline': True,
                    'automatic_specialist_generation': True,
                    'multi_dimensional_validation': True,
                    'self_improving_feedback_loops': True
                },
                'performance_metrics': {
                    'avg_agents_per_task': '2-5',
                    'quality_gates': 5,
                    'success_threshold': '70%',
                    'specialist_generation_threshold': 3
                }
            })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/multi-agent/agents/pool', methods=['GET'])
    def get_agent_pool():
        """Get current agent pool status"""
        try:
            pool_status = []
            
            for agent_id, capability in multi_agent_orchestrator.agent_pool.items():
                pool_status.append({
                    'agent_id': agent_id,
                    'role': capability.role.value,
                    'specialization': capability.specialization,
                    'effectiveness_score': capability.effectiveness_score,
                    'current_load': capability.current_load,
                    'max_concurrent_tasks': capability.max_concurrent_tasks,
                    'success_rate': capability.success_rate,
                    'cost_per_task': capability.cost_per_task,
                    'availability': 'available' if capability.current_load < capability.max_concurrent_tasks else 'busy'
                })
            
            # Calculate pool statistics
            total_agents = len(pool_status)
            available_agents = len([a for a in pool_status if a['availability'] == 'available'])
            avg_effectiveness = sum([a['effectiveness_score'] for a in pool_status]) / max(total_agents, 1)
            specialist_count = len([a for a in pool_status if a['role'] == 'specialist'])
            
            return jsonify({
                'success': True,
                'agent_pool': pool_status,
                'pool_statistics': {
                    'total_agents': total_agents,
                    'available_agents': available_agents,
                    'avg_effectiveness': round(avg_effectiveness, 3),
                    'specialist_agents': specialist_count,
                    'utilization_rate': f"{((total_agents - available_agents) / max(total_agents, 1) * 100):.1f}%"
                }
            })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/multi-agent/tasks/active', methods=['GET'])
    def get_active_tasks():
        """Get all active tasks"""
        try:
            active_tasks = []
            
            for task_id, task in multi_agent_orchestrator.active_tasks.items():
                active_tasks.append({
                    'task_id': task_id,
                    'description': task.description[:100] + '...' if len(task.description) > 100 else task.description,
                    'status': task.status.value,
                    'priority': task.priority,
                    'complexity_score': task.complexity_score,
                    'attempts': task.current_attempts,
                    'max_attempts': task.max_attempts,
                    'assigned_agents': len(task.assigned_agents),
                    'created_at': task.created_at.isoformat(),
                    'deadline': task.deadline.isoformat() if task.deadline else None
                })
            
            # Sort by priority and creation time
            active_tasks.sort(key=lambda x: (-x['priority'], x['created_at']))
            
            return jsonify({
                'success': True,
                'active_tasks': active_tasks,
                'task_counts': {
                    'total': len(active_tasks),
                    'pending': len([t for t in active_tasks if t['status'] == 'pending']),
                    'in_progress': len([t for t in active_tasks if t['status'] == 'in_progress']),
                    'completed': len([t for t in active_tasks if t['status'] == 'completed']),
                    'failed': len([t for t in active_tasks if t['status'] == 'failed']),
                    'specialist_required': len([t for t in active_tasks if t['status'] == 'specialist_required'])
                }
            })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/multi-agent/auto-improve', methods=['POST'])
    def trigger_auto_improvement():
        """Trigger automatic system improvement"""
        try:
            data = request.get_json() or {}
            
            # Submit improvement task to the system
            improvement_task_id = multi_agent_orchestrator.submit_task(
                description="Analyze current system performance and suggest improvements",
                requirements=[
                    "Analyze agent performance metrics",
                    "Identify bottlenecks in parallel processing",
                    "Suggest optimizations for quality assurance pipeline",
                    "Recommend new specialist agent types needed"
                ],
                priority=9  # High priority for system improvements
            )
            
            return jsonify({
                'success': True,
                'improvement_task_id': improvement_task_id,
                'message': 'Auto-improvement analysis initiated',
                'features': [
                    'System performance analysis',
                    'Bottleneck identification',
                    'Quality pipeline optimization',
                    'New specialist recommendations'
                ]
            })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/multi-agent/demo', methods=['POST'])
    def run_demo_task():
        """Run a demonstration of multi-agent capabilities"""
        try:
            # Submit a complex demo task
            demo_task_id = multi_agent_orchestrator.submit_task(
                description="Create a comprehensive web application with AI integration, secure authentication, and real-time features",
                requirements=[
                    "Design responsive user interface",
                    "Implement secure user authentication",
                    "Add AI-powered features",
                    "Ensure real-time data synchronization",
                    "Implement comprehensive security measures",
                    "Add performance monitoring",
                    "Create comprehensive documentation"
                ],
                priority=8
            )
            
            return jsonify({
                'success': True,
                'demo_task_id': demo_task_id,
                'message': 'Demo task submitted for multi-agent processing',
                'expected_features': [
                    'Multiple agents working in parallel',
                    '5-level quality assurance pipeline',
                    'Automatic specialist generation if needed',
                    'Real-time progress tracking',
                    'Comprehensive result synthesis'
                ],
                'tracking_url': f'/api/multi-agent/status/{demo_task_id}'
            })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    return app