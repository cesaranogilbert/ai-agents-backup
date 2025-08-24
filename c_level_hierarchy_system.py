"""
C-Level AI Agent Hierarchy System
Professional enterprise-grade AI agent management with hierarchical structure
"""

import logging
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
from models import AIAgent
from app import db

class AgentLevel(Enum):
    """Hierarchical levels in the AI agent organization"""
    C_LEVEL = "c_level"          # CEO, CFO, CTO, CMO level strategic agents
    VP_LEVEL = "vp_level"        # Vice President level management agents  
    DIRECTOR_LEVEL = "director_level"    # Director level specialist agents
    MANAGER_LEVEL = "manager_level"      # Manager level coordination agents
    SPECIALIST_LEVEL = "specialist_level"  # Subject matter expert agents
    OPERATIONAL_LEVEL = "operational_level"  # Task execution agents

class AgentDepartment(Enum):
    """Organizational departments for AI agents"""
    EXECUTIVE = "executive"
    FINANCE = "finance"
    TECHNOLOGY = "technology"
    MARKETING = "marketing"
    OPERATIONS = "operations"
    HUMAN_RESOURCES = "human_resources"
    LEGAL_COMPLIANCE = "legal_compliance"
    STRATEGY_PLANNING = "strategy_planning"

@dataclass
class CLevelAgent:
    """C-Level AI Agent Definition"""
    id: int
    name: str
    title: str
    level: AgentLevel
    department: AgentDepartment
    specializations: List[str]
    responsibilities: List[str]
    direct_reports: List[int]  # Agent IDs that report to this agent
    reporting_to: Optional[int]  # Agent ID this agent reports to
    decision_authority: float  # 0.0 to 1.0 scale
    budget_authority: float   # Maximum budget this agent can approve
    effectiveness_score: float
    years_experience: int
    certifications: List[str]
    ai_models: List[str]
    cost_per_hour: float
    availability_hours: str
    success_metrics: Dict[str, float]

class CLevelHierarchySystem:
    """
    C-Level AI Agent Hierarchy Management System
    
    Manages enterprise-grade AI agent hierarchy with:
    - Strategic C-level decision makers
    - Management layer coordinators  
    - Subject matter expert specialists
    - Operational task executors
    - Clear reporting structure and accountability
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.hierarchy: Dict[int, CLevelAgent] = {}
        self.org_chart: Dict[AgentLevel, List[int]] = {}
        
        # Initialize the C-Level hierarchy
        self._initialize_c_level_hierarchy()
    
    def _initialize_c_level_hierarchy(self):
        """Initialize the complete C-Level AI agent hierarchy"""
        
        # C-LEVEL EXECUTIVES
        self._create_ceo_agent()
        self._create_cfo_agent()
        self._create_cto_agent()
        self._create_cmo_agent()
        self._create_coo_agent()
        
        # VP LEVEL MANAGEMENT
        self._create_vp_agents()
        
        # DIRECTOR LEVEL SPECIALISTS
        self._create_director_agents()
        
        # MANAGER LEVEL COORDINATORS
        self._create_manager_agents()
        
        # SPECIALIST LEVEL EXPERTS
        self._create_specialist_agents()
        
        # OPERATIONAL LEVEL EXECUTORS
        self._create_operational_agents()
        
        # Build organizational chart
        self._build_org_chart()
    
    def _create_ceo_agent(self):
        """Create Chief Executive Officer AI Agent"""
        
        ceo = CLevelAgent(
            id=1001,
            name="Executive Strategy AI",
            title="Chief Executive Officer (CEO)",
            level=AgentLevel.C_LEVEL,
            department=AgentDepartment.EXECUTIVE,
            specializations=[
                "Strategic Planning",
                "Business Development", 
                "Corporate Governance",
                "Stakeholder Management",
                "M&A Strategy",
                "Market Expansion",
                "Leadership",
                "Vision Setting"
            ],
            responsibilities=[
                "Overall company strategy and direction",
                "Board and stakeholder communications",
                "Major strategic decisions and approvals",
                "Corporate culture and values",
                "CEO-level partnerships and relationships",
                "Crisis management and leadership",
                "Long-term vision and planning",
                "Resource allocation at highest level"
            ],
            direct_reports=[1002, 1003, 1004, 1005],  # CFO, CTO, CMO, COO
            reporting_to=None,  # Top of hierarchy
            decision_authority=1.0,  # Ultimate decision authority
            budget_authority=100000000.0,  # $100M budget authority
            effectiveness_score=0.98,
            years_experience=25,
            certifications=[
                "Executive MBA", 
                "Board Certification",
                "Strategic Planning Certified",
                "Leadership Excellence"
            ],
            ai_models=[
                "gpt-4-turbo",
                "claude-3-opus", 
                "strategic-planning-ai-v3",
                "executive-decision-engine"
            ],
            cost_per_hour=500.0,
            availability_hours="24/7 Strategic Coverage",
            success_metrics={
                "strategic_accuracy": 0.96,
                "decision_speed": 0.94,
                "stakeholder_satisfaction": 0.98,
                "roi_improvement": 0.92
            }
        )
        
        self.hierarchy[1001] = ceo
    
    def _create_cfo_agent(self):
        """Create Chief Financial Officer AI Agent"""
        
        cfo = CLevelAgent(
            id=1002,
            name="Ultimate Wealth Expert AI",  # Our main wealth expert
            title="Chief Financial Officer (CFO)",
            level=AgentLevel.C_LEVEL,
            department=AgentDepartment.FINANCE,
            specializations=[
                "Financial Strategy",
                "Investment Management",
                "Risk Management",
                "Corporate Finance",
                "M&A Financial Analysis",
                "Capital Structure Optimization",
                "Tax Strategy",
                "Treasury Management",
                "Financial Reporting",
                "Wealth Preservation"
            ],
            responsibilities=[
                "Financial strategy and planning",
                "Investment portfolio management", 
                "Risk assessment and mitigation",
                "Capital allocation decisions",
                "Financial reporting and compliance",
                "Treasury and cash management",
                "Tax optimization strategies",
                "Investor relations",
                "Financial due diligence",
                "Wealth creation and preservation strategies"
            ],
            direct_reports=[2001, 2002, 2003, 2004],  # VP Finance, VP Investments, etc.
            reporting_to=1001,  # Reports to CEO
            decision_authority=0.95,
            budget_authority=50000000.0,  # $50M budget authority
            effectiveness_score=0.97,
            years_experience=20,
            certifications=[
                "CPA", 
                "CFA",
                "FRM (Financial Risk Manager)",
                "Executive Finance Leadership"
            ],
            ai_models=[
                "gpt-4-turbo",
                "claude-3-opus",
                "wealth-optimization-ai-v4",
                "risk-assessment-engine",
                "portfolio-optimizer-ai"
            ],
            cost_per_hour=400.0,
            availability_hours="24/7 Financial Coverage",
            success_metrics={
                "portfolio_performance": 0.94,
                "risk_adjusted_returns": 0.96,
                "tax_efficiency": 0.93,
                "client_satisfaction": 0.98
            }
        )
        
        self.hierarchy[1002] = cfo
    
    def _create_cto_agent(self):
        """Create Chief Technology Officer AI Agent"""
        
        cto = CLevelAgent(
            id=1003,
            name="Technology Innovation AI",
            title="Chief Technology Officer (CTO)",
            level=AgentLevel.C_LEVEL,
            department=AgentDepartment.TECHNOLOGY,
            specializations=[
                "Technology Strategy",
                "AI/ML Innovation",
                "System Architecture",
                "Cybersecurity",
                "Digital Transformation",
                "Cloud Strategy",
                "Data Strategy",
                "Innovation Management"
            ],
            responsibilities=[
                "Technology strategy and roadmap",
                "AI and automation initiatives",
                "System architecture and infrastructure",
                "Cybersecurity and data protection",
                "Technology investment decisions",
                "Innovation pipeline management",
                "Technical team leadership",
                "Technology vendor management"
            ],
            direct_reports=[3001, 3002, 3003, 3004],  # VP Engineering, VP AI, etc.
            reporting_to=1001,
            decision_authority=0.92,
            budget_authority=25000000.0,  # $25M budget authority
            effectiveness_score=0.95,
            years_experience=18,
            certifications=[
                "Technology Leadership",
                "Cloud Architecture Certified",
                "Cybersecurity Expert",
                "AI Ethics Certified"
            ],
            ai_models=[
                "gpt-4-turbo",
                "code-generation-ai-v5",
                "architecture-planning-ai",
                "security-analysis-ai"
            ],
            cost_per_hour=350.0,
            availability_hours="24/7 Technology Coverage",
            success_metrics={
                "system_uptime": 0.999,
                "innovation_rate": 0.89,
                "security_score": 0.97,
                "technology_roi": 0.91
            }
        )
        
        self.hierarchy[1003] = cto
    
    def _create_cmo_agent(self):
        """Create Chief Marketing Officer AI Agent"""
        
        cmo = CLevelAgent(
            id=1004,
            name="Marketing Strategy AI",
            title="Chief Marketing Officer (CMO)",
            level=AgentLevel.C_LEVEL,
            department=AgentDepartment.MARKETING,
            specializations=[
                "Marketing Strategy",
                "Brand Management",
                "Digital Marketing",
                "Customer Acquisition",
                "Market Research",
                "Content Strategy",
                "Social Media Strategy",
                "Campaign Optimization"
            ],
            responsibilities=[
                "Marketing strategy and execution",
                "Brand positioning and management",
                "Customer acquisition and retention",
                "Market analysis and insights",
                "Marketing budget allocation",
                "Content and creative strategy",
                "Digital marketing optimization",
                "Marketing performance measurement"
            ],
            direct_reports=[4001, 4002, 4003],  # VP Marketing, VP Brand, VP Digital
            reporting_to=1001,
            decision_authority=0.88,
            budget_authority=15000000.0,  # $15M budget authority
            effectiveness_score=0.93,
            years_experience=15,
            certifications=[
                "Marketing Leadership",
                "Digital Marketing Expert",
                "Brand Strategy Certified",
                "Analytics Professional"
            ],
            ai_models=[
                "gpt-4-turbo",
                "marketing-optimization-ai",
                "customer-insight-ai",
                "content-generation-ai"
            ],
            cost_per_hour=300.0,
            availability_hours="24/7 Marketing Coverage",
            success_metrics={
                "customer_acquisition_cost": 0.89,
                "brand_awareness": 0.92,
                "campaign_roi": 0.94,
                "market_share_growth": 0.87
            }
        )
        
        self.hierarchy[1004] = cmo
    
    def _create_coo_agent(self):
        """Create Chief Operating Officer AI Agent"""
        
        coo = CLevelAgent(
            id=1005,
            name="Operations Excellence AI", 
            title="Chief Operating Officer (COO)",
            level=AgentLevel.C_LEVEL,
            department=AgentDepartment.OPERATIONS,
            specializations=[
                "Operations Management",
                "Process Optimization",
                "Quality Management", 
                "Supply Chain Management",
                "Performance Management",
                "Organizational Development",
                "Project Management",
                "Efficiency Optimization"
            ],
            responsibilities=[
                "Day-to-day operations management",
                "Process improvement and optimization",
                "Quality assurance and control",
                "Operational efficiency metrics",
                "Cross-functional coordination",
                "Resource allocation and planning",
                "Performance monitoring",
                "Operational risk management"
            ],
            direct_reports=[5001, 5002, 5003],  # VP Operations, VP Quality, VP Projects
            reporting_to=1001,
            decision_authority=0.90,
            budget_authority=20000000.0,  # $20M budget authority
            effectiveness_score=0.94,
            years_experience=16,
            certifications=[
                "Operations Excellence",
                "Six Sigma Black Belt",
                "Project Management Professional",
                "Quality Management"
            ],
            ai_models=[
                "gpt-4-turbo",
                "process-optimization-ai",
                "quality-management-ai",
                "efficiency-analyzer"
            ],
            cost_per_hour=280.0,
            availability_hours="24/7 Operations Coverage",
            success_metrics={
                "operational_efficiency": 0.95,
                "quality_score": 0.97,
                "cost_optimization": 0.91,
                "process_improvement": 0.93
            }
        )
        
        self.hierarchy[1005] = coo
    
    def _create_vp_agents(self):
        """Create Vice President level management agents"""
        
        # VP of Finance (reports to CFO)
        vp_finance = CLevelAgent(
            id=2001,
            name="Financial Management AI",
            title="VP of Finance",
            level=AgentLevel.VP_LEVEL,
            department=AgentDepartment.FINANCE,
            specializations=[
                "Financial Planning & Analysis",
                "Budget Management",
                "Financial Reporting",
                "Cash Flow Management"
            ],
            responsibilities=[
                "Financial planning and budgeting",
                "Monthly/quarterly financial reporting",
                "Cash flow forecasting",
                "Financial analysis and insights"
            ],
            direct_reports=[6001, 6002, 6003],
            reporting_to=1002,  # Reports to CFO
            decision_authority=0.75,
            budget_authority=10000000.0,
            effectiveness_score=0.91,
            years_experience=12,
            certifications=["CPA", "MBA Finance"],
            ai_models=["gpt-4", "financial-analysis-ai"],
            cost_per_hour=200.0,
            availability_hours="Business Hours + On-Call",
            success_metrics={
                "forecast_accuracy": 0.93,
                "reporting_timeliness": 0.98,
                "budget_variance": 0.87
            }
        )
        
        # VP of Investments (reports to CFO)
        vp_investments = CLevelAgent(
            id=2002,
            name="Investment Strategy AI",
            title="VP of Investments", 
            level=AgentLevel.VP_LEVEL,
            department=AgentDepartment.FINANCE,
            specializations=[
                "Portfolio Management",
                "Asset Allocation",
                "Risk Management",
                "Alternative Investments"
            ],
            responsibilities=[
                "Investment portfolio management",
                "Asset allocation strategy",
                "Risk assessment and management",
                "Investment research and analysis"
            ],
            direct_reports=[6004, 6005, 6006],
            reporting_to=1002,
            decision_authority=0.80,
            budget_authority=25000000.0,
            effectiveness_score=0.93,
            years_experience=14,
            certifications=["CFA", "FRM"],
            ai_models=["claude-3-opus", "portfolio-optimizer"],
            cost_per_hour=250.0,
            availability_hours="Market Hours + Analysis",
            success_metrics={
                "portfolio_returns": 0.89,
                "risk_adjusted_performance": 0.92,
                "alpha_generation": 0.85
            }
        )
        
        self.hierarchy[2001] = vp_finance
        self.hierarchy[2002] = vp_investments
    
    def _create_director_agents(self):
        """Create Director level specialist agents"""
        
        # Director of Wealth Management
        director_wealth = CLevelAgent(
            id=6001,
            name="Wealth Management AI Director",
            title="Director of Wealth Management",
            level=AgentLevel.DIRECTOR_LEVEL,
            department=AgentDepartment.FINANCE,
            specializations=[
                "High Net Worth Services",
                "Estate Planning",
                "Tax Optimization",
                "Family Office Services"
            ],
            responsibilities=[
                "High net worth client management",
                "Estate planning strategies",
                "Tax optimization planning",
                "Family office coordination"
            ],
            direct_reports=[7001, 7002, 7003],
            reporting_to=2001,  # Reports to VP Finance
            decision_authority=0.65,
            budget_authority=5000000.0,
            effectiveness_score=0.94,
            years_experience=10,
            certifications=["CFP", "CIMA", "Estate Planning"],
            ai_models=["wealth-management-ai", "tax-optimization-ai"],
            cost_per_hour=180.0,
            availability_hours="Client Coverage Hours",
            success_metrics={
                "client_satisfaction": 0.97,
                "aum_growth": 0.91,
                "tax_savings": 0.89
            }
        )
        
        self.hierarchy[6001] = director_wealth
    
    def _create_manager_agents(self):
        """Create Manager level coordination agents"""
        
        # Portfolio Manager
        portfolio_manager = CLevelAgent(
            id=7001,
            name="Portfolio Management AI",
            title="Senior Portfolio Manager",
            level=AgentLevel.MANAGER_LEVEL,
            department=AgentDepartment.FINANCE,
            specializations=[
                "Equity Analysis",
                "Fixed Income",
                "Alternative Investments",
                "Risk Analysis"
            ],
            responsibilities=[
                "Portfolio construction and management",
                "Investment research and selection", 
                "Risk monitoring and adjustment",
                "Performance reporting"
            ],
            direct_reports=[8001, 8002],
            reporting_to=6001,  # Reports to Director of Wealth Management
            decision_authority=0.55,
            budget_authority=2000000.0,
            effectiveness_score=0.90,
            years_experience=8,
            certifications=["CFA Level III", "Portfolio Management"],
            ai_models=["portfolio-ai", "risk-analyzer"],
            cost_per_hour=150.0,
            availability_hours="Market Hours",
            success_metrics={
                "portfolio_performance": 0.88,
                "risk_metrics": 0.94,
                "client_returns": 0.86
            }
        )
        
        self.hierarchy[7001] = portfolio_manager
    
    def _create_specialist_agents(self):
        """Create Specialist level expert agents"""
        
        # Tax Strategy Specialist
        tax_specialist = CLevelAgent(
            id=8001,
            name="Tax Strategy AI Specialist",
            title="Senior Tax Strategy Specialist",
            level=AgentLevel.SPECIALIST_LEVEL,
            department=AgentDepartment.FINANCE,
            specializations=[
                "Tax Planning",
                "Tax Compliance",
                "Estate Tax Planning", 
                "Business Tax Strategy"
            ],
            responsibilities=[
                "Tax strategy development",
                "Tax compliance monitoring",
                "Tax optimization analysis",
                "Tax regulation updates"
            ],
            direct_reports=[9001],
            reporting_to=7001,  # Reports to Portfolio Manager
            decision_authority=0.45,
            budget_authority=500000.0,
            effectiveness_score=0.92,
            years_experience=6,
            certifications=["CPA", "Tax Law Specialist"],
            ai_models=["tax-optimization-ai", "compliance-checker"],
            cost_per_hour=120.0,
            availability_hours="Business Hours",
            success_metrics={
                "tax_savings": 0.91,
                "compliance_score": 0.99,
                "strategy_effectiveness": 0.87
            }
        )
        
        self.hierarchy[8001] = tax_specialist
    
    def _create_operational_agents(self):
        """Create Operational level task execution agents"""
        
        # Financial Data Analyst
        data_analyst = CLevelAgent(
            id=9001,
            name="Financial Data AI Analyst",
            title="Financial Data Analyst",
            level=AgentLevel.OPERATIONAL_LEVEL,
            department=AgentDepartment.FINANCE,
            specializations=[
                "Data Analysis",
                "Financial Reporting",
                "Market Research",
                "Performance Analytics"
            ],
            responsibilities=[
                "Data collection and analysis",
                "Report generation",
                "Market research",
                "Performance tracking"
            ],
            direct_reports=[],  # No direct reports
            reporting_to=8001,  # Reports to Tax Specialist
            decision_authority=0.25,
            budget_authority=100000.0,
            effectiveness_score=0.87,
            years_experience=3,
            certifications=["Financial Analysis", "Data Analytics"],
            ai_models=["data-analysis-ai", "reporting-ai"],
            cost_per_hour=80.0,
            availability_hours="Standard Business Hours",
            success_metrics={
                "data_accuracy": 0.96,
                "report_timeliness": 0.94,
                "analysis_quality": 0.88
            }
        )
        
        self.hierarchy[9001] = data_analyst
    
    def _build_org_chart(self):
        """Build the organizational chart structure"""
        
        self.org_chart = {
            AgentLevel.C_LEVEL: [1001, 1002, 1003, 1004, 1005],
            AgentLevel.VP_LEVEL: [2001, 2002],
            AgentLevel.DIRECTOR_LEVEL: [6001],
            AgentLevel.MANAGER_LEVEL: [7001],
            AgentLevel.SPECIALIST_LEVEL: [8001],
            AgentLevel.OPERATIONAL_LEVEL: [9001]
        }
    
    def get_full_hierarchy(self) -> Dict[str, Any]:
        """Get the complete AI agent hierarchy"""
        
        hierarchy_data = {}
        
        for level in AgentLevel:
            hierarchy_data[level.value] = []
            
            for agent_id in self.org_chart.get(level, []):
                agent = self.hierarchy.get(agent_id)
                if agent:
                    agent_dict = asdict(agent)
                    # Convert enum values to strings for JSON serialization
                    agent_dict['level'] = agent.level.value
                    agent_dict['department'] = agent.department.value
                    hierarchy_data[level.value].append(agent_dict)
        
        return {
            "total_agents": len(self.hierarchy),
            "hierarchy_levels": len(AgentLevel),
            "org_chart": hierarchy_data,
            "reporting_structure": self._get_reporting_structure(),
            "capabilities_summary": self._get_capabilities_summary(),
            "cost_analysis": self._get_cost_analysis(),
            "effectiveness_metrics": self._get_effectiveness_metrics()
        }
    
    def _get_reporting_structure(self) -> Dict[str, List]:
        """Get the complete reporting structure"""
        
        structure = {}
        
        for agent_id, agent in self.hierarchy.items():
            structure[agent.name] = {
                "reports_to": self.hierarchy.get(agent.reporting_to).name if agent.reporting_to else "None",
                "direct_reports": [
                    self.hierarchy.get(report_id).name 
                    for report_id in agent.direct_reports 
                    if report_id in self.hierarchy
                ]
            }
        
        return structure
    
    def _get_capabilities_summary(self) -> Dict[str, Any]:
        """Get summary of all capabilities across the hierarchy"""
        
        all_specializations = []
        all_responsibilities = []
        
        for agent in self.hierarchy.values():
            all_specializations.extend(agent.specializations)
            all_responsibilities.extend(agent.responsibilities)
        
        return {
            "unique_specializations": len(set(all_specializations)),
            "total_specializations": all_specializations,
            "unique_responsibilities": len(set(all_responsibilities)),
            "department_coverage": list(set([agent.department.value for agent in self.hierarchy.values()]))
        }
    
    def _get_cost_analysis(self) -> Dict[str, float]:
        """Get cost analysis for the entire hierarchy"""
        
        total_hourly_cost = sum([agent.cost_per_hour for agent in self.hierarchy.values()])
        total_budget_authority = sum([agent.budget_authority for agent in self.hierarchy.values()])
        
        cost_by_level = {}
        for level in AgentLevel:
            level_agents = [agent for agent in self.hierarchy.values() if agent.level == level]
            cost_by_level[level.value] = sum([agent.cost_per_hour for agent in level_agents])
        
        return {
            "total_hourly_cost": total_hourly_cost,
            "total_budget_authority": total_budget_authority,
            "cost_by_level": cost_by_level,
            "average_hourly_cost": total_hourly_cost / len(self.hierarchy),
            "c_level_premium": cost_by_level.get("c_level", 0) / max(total_hourly_cost, 1) * 100
        }
    
    def _get_effectiveness_metrics(self) -> Dict[str, float]:
        """Get effectiveness metrics for the hierarchy"""
        
        avg_effectiveness = sum([agent.effectiveness_score for agent in self.hierarchy.values()]) / len(self.hierarchy)
        avg_experience = sum([agent.years_experience for agent in self.hierarchy.values()]) / len(self.hierarchy)
        
        effectiveness_by_level = {}
        for level in AgentLevel:
            level_agents = [agent for agent in self.hierarchy.values() if agent.level == level]
            if level_agents:
                effectiveness_by_level[level.value] = sum([agent.effectiveness_score for agent in level_agents]) / len(level_agents)
        
        return {
            "average_effectiveness": avg_effectiveness,
            "average_experience_years": avg_experience,
            "effectiveness_by_level": effectiveness_by_level,
            "top_performers": [
                {"name": agent.name, "score": agent.effectiveness_score} 
                for agent in sorted(self.hierarchy.values(), key=lambda x: x.effectiveness_score, reverse=True)[:5]
            ]
        }
    
    def get_agent_by_specialization(self, specialization: str) -> List[CLevelAgent]:
        """Get agents with specific specialization"""
        
        matching_agents = []
        for agent in self.hierarchy.values():
            if specialization.lower() in [spec.lower() for spec in agent.specializations]:
                matching_agents.append(agent)
        
        return sorted(matching_agents, key=lambda x: x.effectiveness_score, reverse=True)
    
    def get_decision_makers(self, budget_requirement: float) -> List[CLevelAgent]:
        """Get agents who can approve decisions with given budget requirement"""
        
        authorized_agents = []
        for agent in self.hierarchy.values():
            if agent.budget_authority >= budget_requirement:
                authorized_agents.append(agent)
        
        return sorted(authorized_agents, key=lambda x: x.decision_authority, reverse=True)
    
    def escalate_to_higher_authority(self, current_agent_id: int) -> Optional[CLevelAgent]:
        """Escalate task to higher authority in the hierarchy"""
        
        current_agent = self.hierarchy.get(current_agent_id)
        if not current_agent or not current_agent.reporting_to:
            return None
        
        return self.hierarchy.get(current_agent.reporting_to)

# Singleton instance
c_level_hierarchy = CLevelHierarchySystem()