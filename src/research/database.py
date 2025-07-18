"""
Database models and session management for DeerFlow research tasks
"""

import os
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float, Boolean, Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.sql import func
from datetime import datetime
from enum import Enum as PyEnum
import logging

logger = logging.getLogger(__name__)

# Database configuration
DATABASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "database")
os.makedirs(DATABASE_DIR, exist_ok=True)
DATABASE_URL = f"sqlite:///{os.path.join(DATABASE_DIR, 'research_tasks.db')}"

# Create engine and session
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # SQLite specific
    echo=False  # Set to True for SQL debugging
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class ResearchStatusEnum(PyEnum):
    """Research task status states - matches task_manager.py"""
    PENDING = "pending"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"
    SUBMITTED_TO_MKTAGENT = "submitted_to_mktagent"

class ResearchTaskDB(Base):
    """
    SQLAlchemy model for research tasks.
    Persistent storage version of ResearchTask dataclass.
    """
    __tablename__ = "research_tasks"

    # Primary fields
    id = Column(String, primary_key=True, index=True)
    query = Column(Text, nullable=False)
    status = Column(Enum(ResearchStatusEnum), nullable=False, default=ResearchStatusEnum.PENDING)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Results
    final_report = Column(Text, nullable=True)
    error = Column(Text, nullable=True)
    processing_time = Column(Float, nullable=True)
    sources_analyzed = Column(Integer, nullable=True)
    
    # Summary and insights (extracted from final_report)
    summary = Column(Text, nullable=True)
    insights = Column(Text, nullable=True)  # JSON string of key_points list
    
    # Configuration parameters
    max_plan_iterations = Column(Integer, default=1)
    max_step_num = Column(Integer, default=3)
    enable_background_investigation = Column(Boolean, default=True)
    
    # MktAgent integration
    mktagent_campaign_id = Column(Integer, nullable=True)
    mktagent_content_id = Column(Integer, nullable=True)
    submit_to_mktagent = Column(Boolean, default=False)

    def to_dict(self):
        """Convert to dictionary for API responses"""
        import json
        
        # Parse insights JSON if available
        insights_list = []
        if self.insights:
            try:
                insights_list = json.loads(self.insights)
            except json.JSONDecodeError:
                insights_list = []
        
        return {
            "research_id": self.id,
            "query": self.query,
            "status": self.status.value,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "final_report": self.final_report,
            "summary": self.summary,
            "insights": insights_list,
            "error": self.error,
            "processing_time": self.processing_time,
            "sources_analyzed": self.sources_analyzed,
            "mktagent_campaign_id": self.mktagent_campaign_id,
            "mktagent_content_id": self.mktagent_content_id,
            "submit_to_mktagent": self.submit_to_mktagent
        }

def get_db() -> Session:
    """
    Create database session.
    Similar to mktagent's database session pattern.
    """
    db = SessionLocal()
    try:
        return db
    except Exception as e:
        logger.error(f"Error creating database session: {e}")
        db.close()
        raise

def init_database():
    """
    Initialize database tables.
    Called on application startup.
    """
    try:
        Base.metadata.create_all(bind=engine)
        logger.info(f"Database initialized at: {DATABASE_URL}")
        
        # Log database stats
        with get_db() as db:
            total_tasks = db.query(ResearchTaskDB).count()
            logger.info(f"Database contains {total_tasks} research tasks")
            
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise

class ResearchTaskRepository:
    """
    Repository pattern for research task database operations.
    Provides clean interface for CRUD operations.
    """
    
    def __init__(self):
        self.db = get_db()
    
    def create_task(
        self,
        task_id: str,
        query: str,
        max_plan_iterations: int = 1,
        max_step_num: int = 3,
        enable_background_investigation: bool = True,
        mktagent_campaign_id: int = None,
        submit_to_mktagent: bool = False
    ) -> ResearchTaskDB:
        """Create a new research task in database"""
        try:
            task = ResearchTaskDB(
                id=task_id,
                query=query,
                status=ResearchStatusEnum.PENDING,
                max_plan_iterations=max_plan_iterations,
                max_step_num=max_step_num,
                enable_background_investigation=enable_background_investigation,
                mktagent_campaign_id=mktagent_campaign_id,
                submit_to_mktagent=submit_to_mktagent
            )
            
            self.db.add(task)
            self.db.commit()
            self.db.refresh(task)
            
            logger.info(f"Created research task {task_id} in database")
            return task
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to create task {task_id}: {e}")
            raise
    
    def get_task(self, task_id: str) -> ResearchTaskDB:
        """Get task by ID"""
        return self.db.query(ResearchTaskDB).filter(ResearchTaskDB.id == task_id).first()
    
    def update_task_status(
        self,
        task_id: str,
        status: ResearchStatusEnum,
        **kwargs
    ) -> bool:
        """Update task status and other fields"""
        try:
            task = self.get_task(task_id)
            if not task:
                logger.error(f"Task {task_id} not found for update")
                return False
            
            # Update status
            old_status = task.status
            task.status = status
            task.updated_at = datetime.now()
            
            # Handle insights field specially (convert list to JSON string)
            import json
            if 'insights' in kwargs and isinstance(kwargs['insights'], list):
                kwargs['insights'] = json.dumps(kwargs['insights'], ensure_ascii=False)
            
            # Update additional fields
            for key, value in kwargs.items():
                if hasattr(task, key) and value is not None:
                    setattr(task, key, value)
            
            # Set completion time for terminal states
            if status in [ResearchStatusEnum.COMPLETED, ResearchStatusEnum.FAILED]:
                task.completed_at = datetime.now()
                if task.created_at and not task.processing_time:
                    task.processing_time = (task.completed_at - task.created_at).total_seconds()
            
            self.db.commit()
            logger.info(f"Updated task {task_id}: {old_status.value} -> {status.value}")
            return True
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to update task {task_id}: {e}")
            return False
    
    def list_tasks(
        self,
        status_filter: ResearchStatusEnum = None,
        limit: int = 100,
        offset: int = 0
    ) -> tuple[list[ResearchTaskDB], int]:
        """List tasks with filtering and pagination"""
        try:
            query = self.db.query(ResearchTaskDB)
            
            if status_filter:
                query = query.filter(ResearchTaskDB.status == status_filter)
            
            # Get total count before pagination
            total = query.count()
            
            # Apply pagination and ordering
            tasks = query.order_by(ResearchTaskDB.created_at.desc()).offset(offset).limit(limit).all()
            
            return tasks, total
            
        except Exception as e:
            logger.error(f"Failed to list tasks: {e}")
            return [], 0
    
    def delete_task(self, task_id: str) -> bool:
        """Delete a task"""
        try:
            task = self.get_task(task_id)
            if not task:
                return False
            
            self.db.delete(task)
            self.db.commit()
            logger.info(f"Deleted task {task_id}")
            return True
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to delete task {task_id}: {e}")
            return False
    
    def cleanup_old_tasks(self, max_age_hours: int = 24) -> int:
        """Clean up old completed/failed tasks"""
        try:
            from datetime import timedelta
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            
            deleted = self.db.query(ResearchTaskDB).filter(
                ResearchTaskDB.completed_at < cutoff_time,
                ResearchTaskDB.status.in_([
                    ResearchStatusEnum.COMPLETED,
                    ResearchStatusEnum.FAILED
                ])
            ).delete()
            
            self.db.commit()
            
            if deleted > 0:
                logger.info(f"Cleaned up {deleted} old tasks")
            
            return deleted
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to cleanup old tasks: {e}")
            return 0
    
    def get_statistics(self) -> dict:
        """Get task statistics"""
        try:
            total_tasks = self.db.query(ResearchTaskDB).count()
            
            # Count by status
            status_counts = {}
            for status in ResearchStatusEnum:
                count = self.db.query(ResearchTaskDB).filter(
                    ResearchTaskDB.status == status
                ).count()
                status_counts[status.value] = count
            
            # Average processing time for completed tasks
            completed_tasks = self.db.query(ResearchTaskDB).filter(
                ResearchTaskDB.status == ResearchStatusEnum.COMPLETED,
                ResearchTaskDB.processing_time.isnot(None)
            ).all()
            
            avg_processing_time = (
                sum(t.processing_time for t in completed_tasks) / len(completed_tasks)
                if completed_tasks else 0
            )
            
            return {
                "total_tasks": total_tasks,
                "status_counts": status_counts,
                "average_processing_time_seconds": round(avg_processing_time, 2),
                "completed_tasks": len(completed_tasks)
            }
            
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}
    
    def close(self):
        """Close database session"""
        if self.db:
            self.db.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

# Global repository instance
research_repository = ResearchTaskRepository()