import re
import json
import os
import logging
from datetime import datetime
from app import db
from models import ReplitApp, AIAgent, AppCredential
from services.replit_service import ReplitService

class AIAgentService:
    def __init__(self):
        self.replit_service = ReplitService()
        self.patterns = self._load_agent_patterns()
        
    def _load_agent_patterns(self):
        """Load AI agent detection patterns from config"""
        try:
            config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'agent_patterns.json')
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Error loading agent patterns: {str(e)}")
            return self._get_default_patterns()
    
    def _get_default_patterns(self):
        """Default agent detection patterns"""
        return {
            "openai": {
                "imports": [
                    r"import\s+openai",
                    r"from\s+openai\s+import",
                    r"import\s+langchain",
                    r"from\s+langchain\s+import"
                ],
                "api_calls": [
                    r"openai\.ChatCompletion",
                    r"openai\.Completion",
                    r"client\.chat\.completions",
                    r"OpenAI\(",
                    r"ChatOpenAI\("
                ],
                "models": [
                    r"gpt-3\.5-turbo", r"gpt-4", r"text-davinci", 
                    r"text-ada", r"text-babbage", r"text-curie"
                ]
            },
            "anthropic": {
                "imports": [
                    r"import\s+anthropic",
                    r"from\s+anthropic\s+import"
                ],
                "api_calls": [
                    r"anthropic\.Client",
                    r"anthropic\.Anthropic",
                    r"messages\.create"
                ],
                "models": [
                    r"claude-3", r"claude-2", r"claude-instant"
                ]
            },
            "huggingface": {
                "imports": [
                    r"from\s+transformers\s+import",
                    r"import\s+transformers",
                    r"from\s+huggingface_hub\s+import"
                ],
                "api_calls": [
                    r"AutoTokenizer", r"AutoModel", r"pipeline\(",
                    r"HuggingFaceHub", r"load_dataset"
                ],
                "models": [
                    r"bert-", r"gpt2", r"t5-", r"roberta-",
                    r"distilbert", r"xlnet"
                ]
            },
            "local": {
                "imports": [
                    r"import\s+torch",
                    r"import\s+tensorflow",
                    r"from\s+sklearn\s+import",
                    r"import\s+numpy",
                    r"import\s+pandas"
                ],
                "patterns": [
                    r"\.fit\(", r"\.predict\(", r"\.train\(",
                    r"model\.save", r"torch\.load", r"tf\.keras"
                ]
            }
        }
    
    def analyze_all_apps(self):
        """Analyze all active apps for AI agents"""
        try:
            apps = ReplitApp.query.filter_by(is_active=True).all()
            analyzed_count = 0
            
            for app in apps:
                try:
                    self.analyze_app_for_agents(app)
                    analyzed_count += 1
                except Exception as e:
                    logging.error(f"Error analyzing app {app.name}: {str(e)}")
                    continue
                    
            db.session.commit()
            logging.info(f"Analyzed {analyzed_count} apps for AI agents")
            return analyzed_count
            
        except Exception as e:
            logging.error(f"Error in analyze_all_apps: {str(e)}")
            return 0
    
    def analyze_app_for_agents(self, app):
        """Analyze a specific app for AI agents"""
        try:
            # Get app files
            files = self.replit_service.get_app_files(app.repl_id)
            
            if not files:
                logging.warning(f"No files found for app {app.name}")
                return
            
            # Clear existing agents for this app
            AIAgent.query.filter_by(app_id=app.id).delete()
            AppCredential.query.filter_by(app_id=app.id).delete()
            
            detected_agents = {}
            detected_credentials = set()
            
            for file in files:
                if file.get('content'):
                    content = file['content']
                    
                    # Detect AI agents
                    agents = self._detect_agents_in_content(content, file['path'])
                    for agent in agents:
                        agent_key = f"{agent['type']}_{agent['name']}"
                        if agent_key not in detected_agents:
                            detected_agents[agent_key] = agent
                        else:
                            # Merge features and endpoints
                            detected_agents[agent_key]['features'].extend(agent['features'])
                            detected_agents[agent_key]['api_endpoints'].extend(agent['api_endpoints'])
                    
                    # Detect credentials
                    credentials = self._detect_credentials_in_content(content)
                    detected_credentials.update(credentials)
            
            # Save detected agents
            for agent_data in detected_agents.values():
                new_agent = AIAgent(
                    app_id=app.id,
                    agent_type=agent_data['type'],
                    agent_name=agent_data['name'],
                    model_name=agent_data.get('model'),
                    role_description=agent_data.get('role', 'Detected AI agent'),
                    features_used=list(set(agent_data['features'])),
                    api_endpoints=list(set(agent_data['api_endpoints']))
                )
                db.session.add(new_agent)
            
            # Save detected credentials
            for cred in detected_credentials:
                new_credential = AppCredential(
                    app_id=app.id,
                    credential_type='api_key',
                    service_name=cred
                )
                db.session.add(new_credential)
                
        except Exception as e:
            logging.error(f"Error analyzing app {app.id} for agents: {str(e)}")
            raise
    
    def _detect_agents_in_content(self, content, file_path):
        """Detect AI agents in file content"""
        detected = []
        
        for agent_type, patterns in self.patterns.items():
            agent_detected = False
            features = []
            api_endpoints = []
            model_name = None
            
            # Check imports
            if 'imports' in patterns:
                for import_pattern in patterns['imports']:
                    if re.search(import_pattern, content, re.IGNORECASE):
                        agent_detected = True
                        features.append(f"imports_{agent_type}")
                        break
            
            # Check API calls
            if 'api_calls' in patterns:
                for api_pattern in patterns['api_calls']:
                    matches = re.findall(api_pattern, content, re.IGNORECASE)
                    if matches:
                        agent_detected = True
                        features.append(f"api_calls_{agent_type}")
                        api_endpoints.extend(matches)
            
            # Check models
            if 'models' in patterns:
                for model_pattern in patterns['models']:
                    match = re.search(model_pattern, content, re.IGNORECASE)
                    if match:
                        agent_detected = True
                        model_name = match.group(0)
                        features.append(f"model_{model_name}")
                        break
            
            # Check general patterns for local models
            if 'patterns' in patterns:
                for pattern in patterns['patterns']:
                    if re.search(pattern, content, re.IGNORECASE):
                        agent_detected = True
                        features.append(f"pattern_{pattern}")
            
            if agent_detected:
                detected.append({
                    'type': agent_type,
                    'name': f"{agent_type.title()} Agent in {os.path.basename(file_path)}",
                    'model': model_name,
                    'role': self._infer_agent_role(content, agent_type),
                    'features': features,
                    'api_endpoints': api_endpoints
                })
        
        return detected
    
    def _detect_credentials_in_content(self, content):
        """Detect API credentials and services in content"""
        credentials = set()
        
        # Common API key patterns
        patterns = {
            'openai': [r'OPENAI_API_KEY', r'openai.*key'],
            'anthropic': [r'ANTHROPIC_API_KEY', r'anthropic.*key'],
            'google': [r'GOOGLE_API_KEY', r'GOOGLE_.*_KEY'],
            'aws': [r'AWS_ACCESS_KEY', r'AWS_SECRET_KEY'],
            'azure': [r'AZURE_.*_KEY', r'AZURE_.*_SECRET'],
            'huggingface': [r'HUGGINGFACE_.*_TOKEN', r'HF_TOKEN'],
            'telegram': [r'TELEGRAM_.*_TOKEN', r'BOT_TOKEN'],
            'github': [r'GITHUB_TOKEN', r'GH_TOKEN'],
            'stripe': [r'STRIPE_.*_KEY'],
            'sendgrid': [r'SENDGRID_API_KEY'],
            'twilio': [r'TWILIO_.*'],
        }
        
        for service, service_patterns in patterns.items():
            for pattern in service_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    credentials.add(service)
                    break
        
        return credentials
    
    def _infer_agent_role(self, content, agent_type):
        """Infer the role of an AI agent based on content analysis"""
        roles = {
            'chat': [r'chat', r'conversation', r'dialogue', r'message'],
            'completion': [r'complete', r'generate', r'text.*generation'],
            'classification': [r'classify', r'categorize', r'sentiment'],
            'summarization': [r'summarize', r'summary', r'abstract'],
            'translation': [r'translate', r'translation', r'language'],
            'qa': [r'question.*answer', r'qa', r'ask'],
            'embedding': [r'embed', r'vector', r'similarity'],
            'image': [r'image', r'vision', r'visual', r'dall'],
            'code': [r'code', r'programming', r'codex'],
            'analysis': [r'analyze', r'analysis', r'insight']
        }
        
        for role, patterns in roles.items():
            for pattern in patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    return f"{role.title()} Assistant"
        
        return f"{agent_type.title()} Assistant"
    
    def update_agent_usage(self, agent_id, response_time_ms=None, success=True, cost=0.0):
        """Update agent usage statistics"""
        try:
            agent = AIAgent.query.get(agent_id)
            if not agent:
                return False
            
            agent.usage_frequency += 1
            agent.last_used = datetime.utcnow()
            
            if cost > 0:
                agent.cost_estimate += cost
            
            # Update effectiveness score based on success rate
            if success:
                agent.effectiveness_score = min(1.0, agent.effectiveness_score + 0.01)
            else:
                agent.effectiveness_score = max(0.0, agent.effectiveness_score - 0.05)
            
            db.session.commit()
            return True
            
        except Exception as e:
            logging.error(f"Error updating agent usage: {str(e)}")
            return False
