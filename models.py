from app import db
from datetime import datetime
from sqlalchemy import JSON

class ReplitApp(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    repl_id = db.Column(db.String(100), unique=True, nullable=False)
    name = db.Column(db.String(200), nullable=False)
    url = db.Column(db.String(500))
    language = db.Column(db.String(50))
    description = db.Column(db.Text)
    file_count = db.Column(db.Integer, default=0)
    size_kb = db.Column(db.Integer, default=0)
    last_modified = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)
    
    # Relationships
    ai_agents = db.relationship('AIAgent', back_populates='app', cascade='all, delete-orphan')
    credentials = db.relationship('AppCredential', back_populates='app', cascade='all, delete-orphan')

class AIAgent(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    app_id = db.Column(db.Integer, db.ForeignKey('replit_app.id'), nullable=False)
    agent_type = db.Column(db.String(50), nullable=False)  # openai, anthropic, local, custom
    agent_name = db.Column(db.String(100), nullable=False)
    model_name = db.Column(db.String(100))
    role_description = db.Column(db.Text)
    usage_frequency = db.Column(db.Integer, default=0)
    last_used = db.Column(db.DateTime)
    effectiveness_score = db.Column(db.Float, default=0.0)
    cost_estimate = db.Column(db.Float, default=0.0)
    features_used = db.Column(JSON)  # List of features this agent implements
    api_endpoints = db.Column(JSON)  # API endpoints this agent uses
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    app = db.relationship('ReplitApp', back_populates='ai_agents')

class AppCredential(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    app_id = db.Column(db.Integer, db.ForeignKey('replit_app.id'), nullable=False)
    credential_type = db.Column(db.String(50), nullable=False)  # api_key, token, oauth
    service_name = db.Column(db.String(100), nullable=False)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    app = db.relationship('ReplitApp', back_populates='credentials')

class MatrixSnapshot(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    snapshot_date = db.Column(db.Date, nullable=False, unique=True)
    matrix_data = db.Column(JSON, nullable=False)  # Complete matrix data
    total_apps = db.Column(db.Integer, default=0)
    total_agents = db.Column(db.Integer, default=0)
    integration_opportunities = db.Column(JSON)  # List of recommended integrations
    optimization_tips = db.Column(JSON)  # List of optimization recommendations
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class AgentUsageLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    agent_id = db.Column(db.Integer, db.ForeignKey('ai_agent.id'), nullable=False)
    usage_date = db.Column(db.Date, nullable=False)
    usage_count = db.Column(db.Integer, default=1)
    response_time_ms = db.Column(db.Integer)
    success_rate = db.Column(db.Float, default=1.0)
    cost_incurred = db.Column(db.Float, default=0.0)
    
class TelegramNotification(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    notification_type = db.Column(db.String(50), nullable=False)
    message = db.Column(db.Text, nullable=False)
    sent_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_sent = db.Column(db.Boolean, default=False)
    chat_id = db.Column(db.String(100))

class SystemSettings(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    setting_key = db.Column(db.String(100), unique=True, nullable=False)
    setting_value = db.Column(db.Text)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class ExecutedOpportunity(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    opportunity_type = db.Column(db.String(50), nullable=False)  # 'integration' or 'optimization'
    opportunity_id = db.Column(db.String(100), nullable=False)  # unique identifier for the opportunity
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    replit_prompt = db.Column(db.Text)  # Generated prompt for implementation
    status = db.Column(db.String(20), default='executed')  # executed, automated, manual_required, done, failed
    executed_at = db.Column(db.DateTime, default=datetime.utcnow)
    completed_at = db.Column(db.DateTime)
    telegram_sent = db.Column(db.Boolean, default=False)
    automation_notes = db.Column(db.Text)  # Notes about automated vs manual implementation
    applied_changes = db.Column(JSON)  # List of changes that were automatically applied
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# Template Marketplace Models
class AppTemplate(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    slug = db.Column(db.String(200), unique=True, nullable=False)
    description = db.Column(db.Text, nullable=False)
    long_description = db.Column(db.Text)
    category_id = db.Column(db.Integer, db.ForeignKey('template_category.id'), nullable=False)
    
    # Pricing and access
    price = db.Column(db.Float, default=0.0)  # 0 = free template
    is_premium = db.Column(db.Boolean, default=False)
    is_featured = db.Column(db.Boolean, default=False)
    
    # Template content
    template_code = db.Column(db.Text)  # Main application code
    requirements_txt = db.Column(db.Text)  # Python requirements
    config_files = db.Column(JSON)  # Additional config files (package.json, etc.)
    environment_vars = db.Column(JSON)  # Required environment variables
    
    # Technical details
    tech_stack = db.Column(JSON)  # ['Python', 'Flask', 'PostgreSQL', 'Bootstrap']
    complexity_level = db.Column(db.String(20), default='beginner')  # beginner, intermediate, advanced
    estimated_dev_time = db.Column(db.String(50))  # "2-3 hours", "1-2 days"
    
    # Business metrics
    estimated_value = db.Column(db.Float, default=0.0)
    potential_revenue = db.Column(db.String(100))  # "$500-2000/month"
    target_market = db.Column(db.String(200))
    
    # Metadata
    author_name = db.Column(db.String(100), default='DevOpt.ai')
    demo_url = db.Column(db.String(500))
    github_url = db.Column(db.String(500))
    documentation_url = db.Column(db.String(500))
    
    # Stats
    download_count = db.Column(db.Integer, default=0)
    rating_avg = db.Column(db.Float, default=0.0)
    rating_count = db.Column(db.Integer, default=0)
    view_count = db.Column(db.Integer, default=0)
    
    # Status
    is_active = db.Column(db.Boolean, default=True)
    is_approved = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    category = db.relationship('TemplateCategory', back_populates='templates')
    purchases = db.relationship('TemplatePurchase', back_populates='template', cascade='all, delete-orphan')
    reviews = db.relationship('TemplateReview', back_populates='template', cascade='all, delete-orphan')
    tags = db.relationship('TemplateTag', secondary='template_tag_association', back_populates='templates')

class TemplateCategory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False)
    slug = db.Column(db.String(100), unique=True, nullable=False)
    description = db.Column(db.Text)
    icon = db.Column(db.String(100), default='fas fa-folder')
    color = db.Column(db.String(7), default='#007bff')  # Hex color
    sort_order = db.Column(db.Integer, default=0)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    templates = db.relationship('AppTemplate', back_populates='category', cascade='all, delete-orphan')

class TemplatePurchase(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    template_id = db.Column(db.Integer, db.ForeignKey('app_template.id'), nullable=False)
    customer_email = db.Column(db.String(200), nullable=False)
    customer_name = db.Column(db.String(200))
    
    # Payment details
    amount_paid = db.Column(db.Float, nullable=False)
    currency = db.Column(db.String(3), default='USD')
    payment_method = db.Column(db.String(50))  # stripe, paypal, etc.
    transaction_id = db.Column(db.String(100))
    
    # Status
    status = db.Column(db.String(20), default='completed')  # pending, completed, refunded
    download_count = db.Column(db.Integer, default=0)
    max_downloads = db.Column(db.Integer, default=5)
    
    purchased_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_download = db.Column(db.DateTime)
    
    # Relationships
    template = db.relationship('AppTemplate', back_populates='purchases')

class TemplateReview(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    template_id = db.Column(db.Integer, db.ForeignKey('app_template.id'), nullable=False)
    reviewer_name = db.Column(db.String(100), nullable=False)
    reviewer_email = db.Column(db.String(200))
    
    rating = db.Column(db.Integer, nullable=False)  # 1-5 stars
    title = db.Column(db.String(200))
    review_text = db.Column(db.Text)
    
    # Usage details
    implementation_time = db.Column(db.String(50))  # "Completed in 2 hours"
    difficulty_rating = db.Column(db.Integer)  # 1-5 (1=very easy, 5=very hard)
    would_recommend = db.Column(db.Boolean, default=True)
    
    # Metadata
    is_verified_purchase = db.Column(db.Boolean, default=False)
    is_approved = db.Column(db.Boolean, default=True)
    helpful_count = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    template = db.relationship('AppTemplate', back_populates='reviews')

class TemplateTag(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), unique=True, nullable=False)
    slug = db.Column(db.String(50), unique=True, nullable=False)
    description = db.Column(db.String(200))
    color = db.Column(db.String(7), default='#6c757d')  # Hex color
    usage_count = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    templates = db.relationship('AppTemplate', secondary='template_tag_association', back_populates='tags')

# Association table for many-to-many relationship between templates and tags
from sqlalchemy import Table
template_tag_association = Table('template_tag_association', db.Model.metadata,
    db.Column('template_id', db.Integer, db.ForeignKey('app_template.id'), primary_key=True),
    db.Column('tag_id', db.Integer, db.ForeignKey('template_tag.id'), primary_key=True)
)
